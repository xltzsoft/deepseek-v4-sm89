"""CPU tests for DeepSeek-V4 SM89 tensor-parallel head layout."""

import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import sglang.srt.models.deepseek_v4 as deepseek_v4
import sglang.srt.models.deepseek_v4_dspark as deepseek_v4_dspark
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.deepseek_v4 import MqaAttentionBase
from sglang.srt.models.deepseek_v4_dspark import DSparkAttention
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


class _PassthroughProjection(torch.nn.Module):
    def forward(self, x):
        return x, None


class _RecordingAttentionBackend:
    def __init__(self):
        self.q_shape = None
        self.attn_sink_shape = None

    def forward(self, **kwargs):
        q = kwargs["q"]
        self.q_shape = tuple(q.shape)
        self.attn_sink_shape = tuple(kwargs["attn_sink"].shape)
        return torch.zeros_like(q)


class TestDeepseekV4DSparkSm89HeadPadding(unittest.TestCase):
    @staticmethod
    def _make_tp8_layer():
        layer = object.__new__(DSparkAttention)
        torch.nn.Module.__init__(layer)
        layer.attn_tp_size = 8
        layer.attn_tp_rank = 3
        layer.n_heads = 64
        layer.n_local_heads = 8
        layer.head_dim = 4
        layer.rope_head_dim = 2
        layer.n_local_groups = 1
        layer.o_lora_rank = 1
        layer.layer_id = 0
        layer.alt_streams = None
        layer._multi_stream_bs_limit = 64
        layer._use_fast_kernel = True
        layer.attn = SimpleNamespace()
        layer.freqs_cis = torch.empty(0)
        layer.attn_sink = torch.nn.Parameter(torch.arange(64, dtype=torch.float32))
        layer._attn_sink_local = None
        layer.wo_a = torch.nn.Linear(32, 1, bias=False)
        layer.wo_b = _PassthroughProjection()

        q_out_shapes = []

        def kv_proj_only(self, hidden_states):
            return hidden_states.new_zeros(hidden_states.shape[0], self.head_dim)

        def store_block_kv(self, **kwargs):
            return None

        def compute_q(self, hidden_states, positions, q_out=None):
            del positions
            q = hidden_states.new_zeros(
                hidden_states.shape[0], self.n_local_heads, self.head_dim
            )
            q_out_shapes.append(None if q_out is None else tuple(q_out.shape))
            if q_out is not None:
                q_out.copy_(q)
                return q_out
            return q

        layer.kv_proj_only = MethodType(kv_proj_only, layer)
        layer._store_block_kv = MethodType(store_block_kv, layer)
        layer._compute_q = MethodType(compute_q, layer)
        return layer, q_out_shapes

    def _run_forward(self, *, sm89: bool, capture_mode: bool):
        layer, q_out_shapes = self._make_tp8_layer()
        backend = _RecordingAttentionBackend()
        num_tokens = 3
        hidden_states = torch.zeros(num_tokens, 16)
        positions = torch.arange(num_tokens)
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)

        with (
            patch.object(deepseek_v4, "_is_sm89", sm89),
            patch.object(
                deepseek_v4_dspark,
                "get_is_capture_mode",
                return_value=capture_mode,
            ),
            patch.object(
                deepseek_v4_dspark,
                "_resolve_dspark_pool",
                return_value=object(),
            ),
            patch.object(deepseek_v4_dspark, "fused_rope_inplace"),
            patch(
                "sglang.srt.model_executor.forward_context.get_attn_backend",
                return_value=backend,
            ),
        ):
            output = layer(positions, hidden_states, forward_batch)

        self.assertEqual(tuple(output.shape), (num_tokens, 1))
        return backend, q_out_shapes

    def test_sm89_target_verify_capture_uses_native_tp8_heads(self):
        backend, q_out_shapes = self._run_forward(sm89=True, capture_mode=True)

        self.assertEqual(q_out_shapes, [None])
        self.assertEqual(backend.q_shape, (3, 8, 4))
        self.assertEqual(backend.attn_sink_shape, (8,))

    def test_non_sm89_keeps_flashmla_head64_contract(self):
        backend, q_out_shapes = self._run_forward(sm89=False, capture_mode=False)

        self.assertEqual(q_out_shapes, [(3, 8, 4)])
        self.assertEqual(backend.q_shape, (3, 64, 4))
        self.assertEqual(backend.attn_sink_shape, (64,))


if __name__ == "__main__":
    unittest.main()
