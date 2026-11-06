#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

"""Pure-logic tests for the full-extend planner (no NPU required)."""

import pytest

from vllm_ascend.worker.v1.full_extend import (
    FullExtendPlanner,
    default_req_slots,
    generate_prefill_capture_sizes,
)


def make_planner(
    max_capture_tokens: int = 8192,
    req_slots: int = 16,
    context_cap: int = 4096,
    max_context_size: int = 32768,
    padding_factor: float = 2.0,
) -> FullExtendPlanner:
    return FullExtendPlanner(
        capture_sizes=generate_prefill_capture_sizes(max_capture_tokens),
        req_slots=req_slots,
        padding_factor=padding_factor,
        max_context_size=max_context_size,
        chunked_prefill_size=8192,
        context_cap=context_cap,
    )


class TestLadder:
    def test_small_max(self):
        assert generate_prefill_capture_sizes(3) == []
        assert generate_prefill_capture_sizes(4) == [4]

    def test_segments_and_ordering(self):
        sizes = generate_prefill_capture_sizes(8192)
        assert sizes == sorted(set(sizes))
        for expected in (4, 8, 32, 48, 256, 288, 512, 576, 1024, 1280, 4096, 4608, 8192):
            assert expected in sizes, f"{expected} missing from ladder"
        # Exact max is always included even off-ladder.
        assert generate_prefill_capture_sizes(8191)[-1] == 8191
        assert 8191 in generate_prefill_capture_sizes(8191)

    def test_all_within_budget(self):
        for budget in (64, 512, 2048, 8192):
            assert all(s <= budget for s in generate_prefill_capture_sizes(budget))


class TestReqSlots:
    def test_default(self):
        assert default_req_slots(8192, 128) == 16
        assert default_req_slots(4096, 128) == 8

    def test_minimum_one(self):
        assert default_req_slots(256, 128) == 1

    def test_none_budget(self):
        assert default_req_slots(None, 128) == 8

    def test_clamped_to_max_num_seqs(self):
        assert default_req_slots(8192, 8) == 8


class TestPlannerCapture:
    def test_descriptors_descending_non_uniform(self):
        planner = make_planner()
        descs = planner.capture_descriptors()
        tokens = [d.num_tokens for d in descs]
        assert tokens == sorted(tokens, reverse=True)
        assert all(d.num_reqs == 16 and d.uniform is False for d in descs)

    def test_descriptor_for_roundtrip(self):
        planner = make_planner()
        for t in planner.capture_sizes:
            desc = planner.descriptor_for(t)
            assert desc is not None
            assert desc.num_tokens == t
            assert desc.num_reqs == planner.req_slots
        assert planner.descriptor_for(7) is None or 7 in planner.capture_sizes

    def test_scheduled_tokens_cover_exactly(self):
        planner = make_planner(req_slots=16, context_cap=4096)
        for t in planner.capture_sizes:
            lens = planner.build_capture_scheduled_tokens(t)
            assert len(lens) == planner.req_slots
            assert sum(lens) == t
            assert all(0 < x <= 4096 for x in lens if x > 0)
            # sentinels are strictly trailing zeros
            assert lens == sorted(lens, reverse=True)

    def test_slots_must_cover_largest_bucket(self):
        with pytest.raises(ValueError, match="req_slots"):
            make_planner(req_slots=1, context_cap=4096)


class TestDispatch:
    def test_exact_bucket_hit(self):
        planner = make_planner()
        desc = planner.try_dispatch(
            num_tokens=1024, num_reqs=4, max_query_len=2048, max_kv_len=8192
        )
        assert desc is not None
        assert desc.num_tokens == 1024
        assert desc.num_reqs == 16
        assert desc.uniform is False

    def test_pad_up_to_next_bucket(self):
        planner = make_planner()
        desc = planner.try_dispatch(
            num_tokens=1025, num_reqs=1, max_query_len=1025, max_kv_len=100
        )
        assert desc is not None
        assert desc.num_tokens == 1280

    def test_reject_too_many_reqs(self):
        planner = make_planner(req_slots=4)
        assert (
            planner.try_dispatch(num_tokens=64, num_reqs=5, max_query_len=16, max_kv_len=16)
            is None
        )

    def test_reject_query_over_context_cap(self):
        planner = make_planner(context_cap=1024)
        assert (
            planner.try_dispatch(num_tokens=512, num_reqs=1, max_query_len=2048, max_kv_len=16)
            is None
        )

    def test_reject_kv_over_max_context(self):
        planner = make_planner(max_context_size=4096)
        assert (
            planner.try_dispatch(num_tokens=512, num_reqs=1, max_query_len=8, max_kv_len=8192)
            is None
        )

    def test_reject_over_largest_bucket(self):
        planner = make_planner(max_capture_tokens=1024, context_cap=1024)
        assert (
            planner.try_dispatch(num_tokens=2000, num_reqs=1, max_query_len=8, max_kv_len=16)
            is None
        )

    def test_padding_waste_gate(self):
        # bucket 48 for 33 tokens is 45% padding: accepted at factor 2.0,
        # rejected at a tight factor.
        loose = make_planner(padding_factor=2.0)
        assert (
            loose.try_dispatch(num_tokens=33, num_reqs=1, max_query_len=33, max_kv_len=16)
            is not None
        )
        tight = make_planner(padding_factor=1.05)
        assert (
            tight.try_dispatch(num_tokens=33, num_reqs=1, max_query_len=33, max_kv_len=16)
            is None
        )


class TestPadToBucket:
    def test_lookup(self):
        planner = make_planner()
        assert planner.pad_to_bucket(4) == 4
        assert planner.pad_to_bucket(33) == 48
        assert planner.pad_to_bucket(planner.capture_sizes[-1]) == planner.capture_sizes[-1]
        assert planner.pad_to_bucket(planner.capture_sizes[-1] + 1) is None
