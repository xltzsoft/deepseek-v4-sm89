"""CPU tests for DeepSeek-V4 SM89 tensor-parallel head layout."""

import unittest
from unittest.mock import patch

import torch

import sglang.srt.models.deepseek_v4 as deepseek_v4
from sglang.srt.models.deepseek_v4 import MqaAttentionBase
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDeepseekV4Sm89HeadPadding(unittest.TestCase):
    @staticmethod
    def _make_tp8_layer(rank: int = 3) -> MqaAttentionBase:
        layer = object.__new__(MqaAttentionBase)
        torch.nn.Module.__init__(layer)
        layer.attn_tp_size = 8
        layer.attn_tp_rank = rank
        layer.n_heads = 64
        layer.n_local_heads = 8
        layer.head_dim = 512
        layer.attn_sink = torch.nn.Parameter(torch.arange(64, dtype=torch.float32))
        layer._attn_sink_local = None
        return layer

    def test_sm89_tp8_keeps_native_query_heads(self):
        layer = self._make_tp8_layer()
        x = torch.empty(4, 32)

        with patch.object(deepseek_v4, "_is_sm89", True):
            tp_slice, q_padded, q_out = layer._prepare_tp_query_buffer(x)

        self.assertEqual(tp_slice, slice(None))
        self.assertIsNone(q_padded)
        self.assertIsNone(q_out)

    def test_sm89_tp8_sink_is_the_exact_rank_local_slice(self):
        layer = self._make_tp8_layer(rank=3)

        with patch.object(deepseek_v4, "_is_sm89", True):
            sink = layer._local_attn_sink()

        self.assertEqual(tuple(sink.shape), (8,))
        torch.testing.assert_close(sink, torch.arange(24, 32, dtype=torch.float32))

    def test_non_sm89_preserves_flashmla_head64_layout(self):
        layer = self._make_tp8_layer(rank=3)
        x = torch.empty(4, 32)

        with (
            patch.object(deepseek_v4, "_is_sm89", False),
            patch.object(deepseek_v4, "_is_gfx942_supported", True),
        ):
            tp_slice, q_padded, q_out = layer._prepare_tp_query_buffer(x)
            sink = layer._local_attn_sink()

        self.assertEqual(tp_slice, slice(0, 8))
        self.assertEqual(tuple(q_padded.shape), (4, 64, 512))
        self.assertEqual(tuple(q_out.shape), (4, 8, 512))
        self.assertEqual(q_out.data_ptr(), q_padded.data_ptr())
        self.assertEqual(tuple(sink.shape), (64,))
        torch.testing.assert_close(sink[:8], torch.arange(24, 32, dtype=torch.float32))
        torch.testing.assert_close(sink[8:], torch.zeros(56, dtype=torch.float32))


if __name__ == "__main__":
    unittest.main()
