"""CPU regressions for DeepSeek-V4 compression-state initialization."""

import unittest

import torch
from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDeepseekV4CompressStateInit(unittest.TestCase):
    def _assert_empty_state(self, *, ratio: int, overlap: bool) -> None:
        pool = CompressStatePool(
            size=16,
            ring_size=8,
            overlap=overlap,
            head_dim=4,
            dtype=torch.float32,
            device="cpu",
            enable_memory_saver=False,
            ratio=ratio,
        )

        torch.testing.assert_close(
            pool.kv_score_buffer.kv,
            torch.zeros_like(pool.kv_score_buffer.kv),
        )
        self.assertTrue(torch.isneginf(pool.kv_score_buffer.score).all())

    def test_c4_initializes_every_ring_slot(self):
        self._assert_empty_state(ratio=4, overlap=True)

    def test_c128_initializes_every_ring_slot(self):
        self._assert_empty_state(ratio=128, overlap=False)


if __name__ == "__main__":
    unittest.main()
