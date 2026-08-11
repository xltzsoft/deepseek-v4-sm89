"""CPU regressions for DeepSeek-V4 compression-state reuse boundaries."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _state_pool(*, ratio: int, ring_size: int = 8) -> CompressStatePool:
    return CompressStatePool(
        size=32,
        ring_size=ring_size,
        overlap=ratio == 4,
        head_dim=4,
        dtype=torch.float32,
        device="cpu",
        enable_memory_saver=False,
        ratio=ratio,
        swa_page_size=256,
    )


def _assert_empty_rows(test: unittest.TestCase, pool, rows) -> None:
    test.assertTrue(
        (pool.kv_score_buffer.kv_score[rows, : pool.last_dim // 2] == 0).all()
    )
    test.assertTrue(
        torch.isneginf(pool.kv_score_buffer.kv_score[rows, pool.last_dim // 2 :]).all()
    )


class TestDeepseekV4StateLifecycle(unittest.TestCase):
    def test_req_slot_reset_hook_runs_only_for_new_assignments(self):
        pool = ReqToTokenPool(
            size=1,
            max_context_len=8,
            device="cpu",
            enable_memory_saver=False,
        )
        calls = []
        pool.register_slot_reset_hook(lambda slots: calls.append(tuple(slots)))

        req = SimpleNamespace(
            req_pool_idx=None,
            inflight_middle_chunks=0,
            kv_committed_len=0,
        )
        self.assertEqual(pool.alloc([req]), [1])
        self.assertEqual(calls, [])
        pool.drain_slot_reset_hooks()
        self.assertEqual(calls, [(1,)])

        req.inflight_middle_chunks = 1
        self.assertEqual(pool.alloc([req]), [1])
        self.assertEqual(calls, [(1,)])

        pool.free(req)
        replacement = SimpleNamespace(
            req_pool_idx=None,
            inflight_middle_chunks=0,
            kv_committed_len=0,
        )
        self.assertEqual(pool.alloc([replacement]), [1])
        self.assertEqual(calls, [(1,)])
        pool.drain_slot_reset_hooks()
        self.assertEqual(calls, [(1,), (1,)])

    def test_c128_reset_isolated_to_reused_request_slot(self):
        state = _state_pool(ratio=128, ring_size=4)
        state.kv_score_buffer.kv_score.fill_(7)
        token_pool = object.__new__(DeepSeekV4TokenToKVPool)
        token_pool.device = "cpu"
        token_pool.compress_state_pools = [state]
        token_pool.online_c128_mtp_pending_seq_lens = None

        token_pool.clear_c128_req_state(1)

        _assert_empty_rows(self, state, slice(4, 8))
        self.assertTrue((state.kv_score_buffer.kv_score[:4] == 7).all())
        self.assertTrue((state.kv_score_buffer.kv_score[8:12] == 7).all())

    def test_online_c128_reset_clears_all_draft_banks_and_pending_marker(self):
        state = CompressStatePool(
            size=4,
            ring_size=1,
            overlap=False,
            head_dim=4,
            dtype=torch.float32,
            device="cpu",
            enable_memory_saver=False,
            ratio=128,
            online=True,
            online_mtp_max_draft_tokens=2,
        )
        state.kv_score_buffer.kv_score.fill_(7)
        token_pool = object.__new__(DeepSeekV4TokenToKVPool)
        token_pool.device = "cpu"
        token_pool.compress_state_pools = [state]
        token_pool.online_c128_mtp_pending_seq_lens = torch.full((4,), 99)

        with patch("sglang.srt.mem_cache.deepseek_v4_memory_pool.ONLINE_C128", True):
            token_pool.clear_c128_req_states((1, 3))

        reset_rows = torch.tensor([1, 3, 7, 9, 13, 15])
        head_dim = state.last_dim // 3
        reset = state.kv_score_buffer.kv_score[reset_rows]
        self.assertTrue(torch.isneginf(reset[:, :head_dim]).all())
        self.assertTrue((reset[:, head_dim:] == 0).all())
        self.assertTrue((state.kv_score_buffer.kv_score[[0, 2]] == 7).all())
        self.assertEqual(
            token_pool.online_c128_mtp_pending_seq_lens.tolist(), [99, -1, 99, -1]
        )

    def test_c4_reset_isolated_to_released_swa_page(self):
        attention_state = _state_pool(ratio=4)
        indexer_state = _state_pool(ratio=4)
        attention_state.kv_score_buffer.kv_score.fill_(7)
        indexer_state.kv_score_buffer.kv_score.fill_(9)
        token_pool = object.__new__(DeepSeekV4TokenToKVPool)
        token_pool.swa_page_size = 256
        token_pool.compress_state_pools = [attention_state]
        token_pool.indexer_compress_state_pools = [indexer_state]

        token_pool.clear_swa_page_state(torch.tensor([256, 300, 511]))

        for state in (attention_state, indexer_state):
            _assert_empty_rows(self, state, slice(8, 16))
        self.assertTrue((attention_state.kv_score_buffer.kv_score[:8] == 7).all())
        self.assertTrue((attention_state.kv_score_buffer.kv_score[16:24] == 7).all())
        self.assertTrue((indexer_state.kv_score_buffer.kv_score[:8] == 9).all())
        self.assertTrue((indexer_state.kv_score_buffer.kv_score[16:24] == 9).all())

    def test_clear_all_swa_page_state_resets_only_c4_pools(self):
        attention_state = _state_pool(ratio=4)
        indexer_state = _state_pool(ratio=4)
        c128_state = _state_pool(ratio=128, ring_size=4)
        attention_state.kv_score_buffer.kv_score.fill_(7)
        indexer_state.kv_score_buffer.kv_score.fill_(9)
        c128_state.kv_score_buffer.kv_score.fill_(11)
        token_pool = object.__new__(DeepSeekV4TokenToKVPool)
        token_pool.compress_state_pools = [attention_state, c128_state]
        token_pool.indexer_compress_state_pools = [indexer_state]

        token_pool.clear_all_swa_page_state()

        _assert_empty_rows(self, attention_state, slice(None))
        _assert_empty_rows(self, indexer_state, slice(None))
        self.assertTrue((c128_state.kv_score_buffer.kv_score == 11).all())

    def test_swa_allocator_clears_state_before_returning_physical_page(self):
        allocator = object.__new__(SWATokenToKVPoolAllocator)
        allocator.page_size = 256
        allocator.full_to_swa_index_mapping = torch.zeros(769, dtype=torch.int64)
        allocator.full_to_swa_index_mapping[256:512] = torch.arange(512, 768)
        allocator.full_to_swa_index_mapping[-1] = -1
        calls = []
        allocator._kvcache = SimpleNamespace(
            clear_swa_page_state=lambda indices: calls.append(indices.clone())
        )
        allocator.swa_attn_allocator = SimpleNamespace(
            free=lambda indices: calls.append(indices.clone())
        )

        allocator.free_swa(torch.tensor([300]))

        self.assertEqual(len(calls), 2)
        torch.testing.assert_close(calls[0], torch.arange(512, 768))
        torch.testing.assert_close(calls[1], torch.arange(512, 768))
        self.assertTrue((allocator.full_to_swa_index_mapping[256:512] == 0).all())


if __name__ == "__main__":
    unittest.main()
