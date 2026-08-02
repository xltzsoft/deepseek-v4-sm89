"""CPU regressions for bundled DeepSeek-V4 DSpark weight loading."""

import unittest
from types import SimpleNamespace

import torch
from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _RecordingExpertParam:
    def __init__(self):
        self.calls = []

    def weight_loader(
        self,
        param,
        loaded_weight,
        param_name,
        *,
        shard_id=None,
        expert_id=None,
    ):
        self.calls.append(
            (param, loaded_weight.clone(), param_name, shard_id, expert_id)
        )


class TestDeepseekV4DSparkWeightLoading(unittest.TestCase):
    @staticmethod
    def _make_loader_model(*, num_fused_shared_experts: int, params):
        remapper = SimpleNamespace(confidence_head=None)
        return SimpleNamespace(
            config=SimpleNamespace(n_routed_experts=256),
            num_fused_shared_experts=num_fused_shared_experts,
            named_parameters=lambda: list(params.items()),
            _remap_dspark_weight_name=lambda name: (
                DeepseekV4ForCausalLMDSpark._remap_dspark_weight_name(remapper, name)
            ),
            _assert_confidence_head_loaded=lambda **_kwargs: None,
        )

    def test_bundled_shared_expert_keys_map_to_stage_mlp(self):
        remapper = SimpleNamespace(confidence_head=None)

        mapped = DeepseekV4ForCausalLMDSpark._remap_dspark_weight_name(
            remapper, "mtp.2.ffn.shared_experts.w3.scale"
        )

        self.assertEqual(
            mapped,
            "stages.2.mlp.shared_experts.up_proj.weight_scale_inv",
        )

    def test_fused_shared_expert_loads_into_expert_256(self):
        recorder = _RecordingExpertParam()
        params = {
            "stages.0.mlp.experts.w13_weight": recorder,
            "stages.0.mlp.experts.w13_weight_scale_inv": recorder,
            "stages.0.mlp.experts.w2_weight": recorder,
            "stages.0.mlp.experts.w2_weight_scale_inv": recorder,
        }
        model = self._make_loader_model(
            num_fused_shared_experts=1,
            params=params,
        )
        suffixes = [
            "w1.weight",
            "w1.scale",
            "w2.weight",
            "w2.scale",
            "w3.weight",
            "w3.scale",
        ]
        weights = [
            (
                f"mtp.0.ffn.shared_experts.{suffix}",
                torch.tensor([index], dtype=torch.float32),
            )
            for index, suffix in enumerate(suffixes)
        ]

        DeepseekV4ForCausalLMDSpark.load_weights(model, weights)

        self.assertEqual(len(recorder.calls), 6)
        self.assertEqual(
            [call[2:] for call in recorder.calls],
            [
                ("stages.0.mlp.experts.w13_weight", "w1", 256),
                ("stages.0.mlp.experts.w13_weight_scale_inv", "w1", 256),
                ("stages.0.mlp.experts.w2_weight", "w2", 256),
                ("stages.0.mlp.experts.w2_weight_scale_inv", "w2", 256),
                ("stages.0.mlp.experts.w13_weight", "w3", 256),
                ("stages.0.mlp.experts.w13_weight_scale_inv", "w3", 256),
            ],
        )
        for index, call in enumerate(recorder.calls):
            torch.testing.assert_close(
                call[1], torch.tensor([index], dtype=torch.float32)
            )


if __name__ == "__main__":
    unittest.main()
