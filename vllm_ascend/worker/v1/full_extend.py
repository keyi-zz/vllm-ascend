# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-extend graph planning for vllm-ascend.

Full-extend captures ONE whole-forward ACL graph per token bucket for
non-uniform (prefill / mixed) batches, following the design popularized by
SGLang PR #27988:

* token axis: pad the batch up to the nearest prefill capture bucket;
* request axis: a fixed number of request slots; unused slots are filled
  with zero-query-length sentinel requests whose KV length is 1 and whose
  block-table row points at block 0;
* attention runs inside the graph and its variable host-side parameters
  (``actual_seq_lengths_q`` / ``seq_lens_list`` / ``block_table``) are
  rebound on every replay through the UpdatableGraph task-update channel
  (the Ascend equivalent of SGLang's out-of-graph metadata contract).

This module is deliberately free of torch imports so the planning logic can
be unit-tested on a plain CPU environment.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

from vllm.config import VllmConfig
from vllm.forward_context import BatchDescriptor
from vllm.logger import init_logger

logger = init_logger(__name__)


def generate_prefill_capture_sizes(max_tokens: int) -> list[int]:
    """Generate the prefill token-bucket ladder.

    The ladder follows SGLang's ``generate_prefill_cuda_graph_batch_sizes``
    spacing: dense at small token counts (where launch overhead dominates)
    and progressively sparser towards the chunked-prefill budget. The exact
    ``max_tokens`` is always included so that fully-filled chunked-prefill
    steps replay without padding waste.
    """
    if max_tokens < 4:
        return []
    sizes = (
        list(range(4, 33, 4))
        + list(range(48, 257, 16))
        + list(range(288, 513, 32))
        + list(range(576, 1025, 64))
        + list(range(1280, 4097, 256))
        + list(range(4608, max_tokens + 1, 512))
    )
    sizes = [s for s in sizes if s <= max_tokens]
    if not sizes or sizes[-1] != max_tokens:
        sizes.append(max_tokens)
    return sorted(set(sizes))


def default_req_slots(chunked_prefill_size: int | None, max_num_seqs: int) -> int:
    """Fixed request-slot count per full-extend graph.

    Mirrors SGLang's ``full_prefill_max_req`` default:
    ``max(chunked_prefill_size // 512, 1)``, clamped to ``max_num_seqs``.
    """
    budget = chunked_prefill_size if chunked_prefill_size else 4096
    slots = max(budget // 512, 1)
    return min(slots, max_num_seqs)


@dataclass
class FullExtendPlanner:
    """Owns the full-extend key table, bucket ladder and dispatch gates."""

    capture_sizes: list[int]
    req_slots: int
    padding_factor: float
    max_context_size: int
    chunked_prefill_size: int
    # ``context_cap`` is the per-request synthetic query cap used at capture;
    # it also bounds the KV length a replaying batch may carry.
    context_cap: int = 0
    _sorted_sizes: list[int] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._sorted_sizes = sorted(self.capture_sizes)
        self._desc_by_tokens = {
            t: BatchDescriptor(num_tokens=t, num_reqs=self.req_slots, uniform=False, has_lora=False)
            for t in self.capture_sizes
        }
        # Every bucket must be coverable by the request slots under the
        # per-request context cap, otherwise a bucket cannot be captured.
        min_slots_needed = -(-max(self.capture_sizes) // self.context_cap) if self.capture_sizes else 0
        if min_slots_needed > self.req_slots:
            raise ValueError(
                f"full_extend: req_slots={self.req_slots} cannot cover the largest "
                f"capture bucket {max(self.capture_sizes)} with per-request context "
                f"cap {self.context_cap}; need at least {min_slots_needed} slots."
            )

    def descriptor_for(self, num_tokens: int) -> BatchDescriptor | None:
        """Capture key for an exact bucket token count, or None."""
        return self._desc_by_tokens.get(num_tokens)

    # ------------------------------------------------------------------
    # capture-side helpers
    # ------------------------------------------------------------------
    def capture_descriptors(self) -> list[BatchDescriptor]:
        """Capture keys, largest token count first (graph-pool friendly)."""
        return [
            BatchDescriptor(
                num_tokens=t, num_reqs=self.req_slots, uniform=False, has_lora=False
            )
            for t in sorted(self.capture_sizes, reverse=True)
        ]

    def build_capture_scheduled_tokens(self, num_tokens: int) -> list[int]:
        """Synthetic per-slot query lengths for capturing ``num_tokens``.

        Synthetic requests each carry at most ``context_cap`` tokens and
        exactly cover ``num_tokens``; the remaining slots are zero-length
        sentinels (query length 0, KV length 1 at metadata-build time).
        """
        lens: list[int] = []
        start = 0
        while start < num_tokens:
            take = min(self.context_cap, num_tokens - start)
            lens.append(take)
            start += take
        assert sum(lens) == num_tokens
        assert len(lens) <= self.req_slots
        lens += [0] * (self.req_slots - len(lens))
        return lens

    # ------------------------------------------------------------------
    # replay-side dispatch
    # ------------------------------------------------------------------
    def pad_to_bucket(self, num_tokens: int) -> int | None:
        """Smallest captured bucket >= ``num_tokens``, or None if too large."""
        idx = bisect.bisect_left(self._sorted_sizes, num_tokens)
        if idx >= len(self._sorted_sizes):
            return None
        return self._sorted_sizes[idx]

    def try_dispatch(
        self,
        num_tokens: int,
        num_reqs: int,
        max_query_len: int,
        max_kv_len: int,
    ) -> BatchDescriptor | None:
        """Gates for replaying a real batch through a full-extend graph.

        Returns the capture key on success, or None when the batch should
        fall through to the regular PIECEWISE/eager path.
        """
        if num_reqs > self.req_slots:
            return None
        if max_query_len > self.context_cap:
            return None
        if max_kv_len > self.max_context_size:
            return None
        padded = self.pad_to_bucket(num_tokens)
        if padded is None:
            return None
        # Padding waste gate: padding rows are real GEMM work.
        if padded > num_tokens * self.padding_factor:
            return None
        return BatchDescriptor(
            num_tokens=padded, num_reqs=self.req_slots, uniform=False, has_lora=False
        )

    @classmethod
    def maybe_create(
        cls,
        vllm_config: VllmConfig,
        full_extend_config,
    ) -> FullExtendPlanner | None:
        """Build a planner from configs, or None when disabled/unsupported.

        v1 restrictions (each logged once): speculative decoding, pipeline /
        data / context / sequence parallelism, encoder-decoder, multimodal,
        LoRA and hybrid (Mamba) models are out of scope.
        """
        if not full_extend_config.enabled:
            return None

        scheduler_config = vllm_config.scheduler_config
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config

        reasons: list[str] = []
        if vllm_config.speculative_config is not None:
            reasons.append("speculative decoding")
        if parallel_config.pipeline_parallel_size > 1:
            reasons.append("pipeline parallelism")
        if parallel_config.data_parallel_size > 1:
            reasons.append("data parallelism (uncoordinated dispatch)")
        if parallel_config.context_parallel_size > 1:
            reasons.append("context parallelism")
        if model_config.is_encoder_decoder:
            reasons.append("encoder-decoder model")
        if getattr(model_config, "multimodal_config", None) is not None:
            reasons.append("multimodal model")
        if vllm_config.lora_config is not None:
            reasons.append("LoRA")
        if getattr(model_config, "is_hybrid", False):
            reasons.append("hybrid (Mamba) model")
        if reasons:
            logger.warning_once(
                "full_extend is disabled: unsupported configuration (%s).",
                "; ".join(reasons),
            )
            return None

        chunked_prefill_size = scheduler_config.chunked_prefill_size
        max_num_seqs = scheduler_config.max_num_seqs
        max_model_len = model_config.max_model_len

        max_capture_tokens = full_extend_config.max_capture_tokens
        if max_capture_tokens is None:
            max_capture_tokens = min(chunked_prefill_size or 8192, 8192)
        max_capture_tokens = min(max_capture_tokens, scheduler_config.max_num_batched_tokens)
        if max_capture_tokens < 4:
            logger.warning_once(
                "full_extend is disabled: max_capture_tokens=%d is too small.",
                max_capture_tokens,
            )
            return None

        req_slots = full_extend_config.req_slots
        if req_slots is None:
            req_slots = default_req_slots(chunked_prefill_size, max_num_seqs)
        req_slots = min(req_slots, max_num_seqs)

        max_context_size = full_extend_config.max_context_size
        if max_context_size is None:
            max_context_size = max_model_len
        context_cap = min(max_context_size, max_model_len)

        capture_sizes = generate_prefill_capture_sizes(max_capture_tokens)
        try:
            planner = cls(
                capture_sizes=capture_sizes,
                req_slots=req_slots,
                padding_factor=full_extend_config.padding_factor,
                max_context_size=max_context_size,
                chunked_prefill_size=chunked_prefill_size or 0,
                context_cap=context_cap,
            )
        except ValueError as e:
            logger.warning_once("full_extend is disabled: %s", e)
            return None

        logger.info(
            "full_extend planner enabled: %d buckets [%d..%d], req_slots=%d, "
            "context_cap=%d, padding_factor=%.1f",
            len(capture_sizes),
            capture_sizes[0],
            capture_sizes[-1],
            req_slots,
            context_cap,
            full_extend_config.padding_factor,
        )
        return planner
