"""CPU regressions for the SM89 sparse decode launch configuration."""

import unittest

from sglang.kernels.ops.attention.flash_mla_sm120_triton import (
    _get_sparse_decode_configs,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestFlashMlaSm89Config(unittest.TestCase):
    def test_sm89_uses_offline_tuned_single_config(self):
        configs = _get_sparse_decode_configs((8, 9))

        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].kwargs, {"BLOCK_T": 32})
        self.assertEqual(configs[0].num_warps, 8)
        self.assertEqual(configs[0].num_stages, 2)

    def test_other_architectures_keep_runtime_autotuning(self):
        configs = _get_sparse_decode_configs((12, 0))

        self.assertEqual(len(configs), 3)
        self.assertEqual(
            [(config.kwargs["BLOCK_T"], config.num_warps) for config in configs],
            [(16, 4), (16, 8), (32, 8)],
        )


if __name__ == "__main__":
    unittest.main()
