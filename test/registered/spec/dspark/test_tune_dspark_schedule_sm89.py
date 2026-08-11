from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[4]
TUNER_PATH = ROOT / "benchmark/kernels/deepseek/tune_dspark_schedule_sm89.py"
TUNER_SPEC = importlib.util.spec_from_file_location(
    "tune_dspark_schedule_sm89", TUNER_PATH
)
assert TUNER_SPEC is not None and TUNER_SPEC.loader is not None
tuner = importlib.util.module_from_spec(TUNER_SPEC)
sys.modules[TUNER_SPEC.name] = tuner
TUNER_SPEC.loader.exec_module(tuner)


class TestCandidateSpace(unittest.TestCase):
    def test_default_space_is_cartesian_and_contains_production_shape(self):
        candidates = tuner.generate_candidates()
        self.assertEqual(len(candidates), 36)
        self.assertEqual(len({candidate.candidate_id for candidate in candidates}), 36)
        self.assertIn("bc64-bcp256-w4", {c.candidate_id for c in candidates})

    def test_invalid_candidate_axes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "BLOCK_C"):
            tuner.validate_candidate(tuner.ScheduleCandidate(24, 128, 4))
        with self.assertRaisesRegex(ValueError, "num_warps"):
            tuner.validate_candidate(tuner.ScheduleCandidate(32, 128, 3))

    def test_representative_cases_cover_shapes_budgets_and_patterns(self):
        cases = tuner.select_cases("representative")
        self.assertEqual(len(cases), 12)
        self.assertEqual(
            {case.batch_size for case in cases},
            {1, 4, 8, 16, 32, 48, 64, 128, 192, 256},
        )
        self.assertIn(0, {case.budget for case in cases})
        self.assertTrue(any(case.budget >= 768 for case in cases))
        self.assertEqual(
            {case.pattern for case in cases}, {"random", "quantized", "sparse", "ties"}
        )


class TestProductionSourceInspection(unittest.TestCase):
    def test_extracts_required_symbols_and_defaults_without_importing_runtime(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "dspark_schedule.py"
            path.write_text(
                """
def _schedule_topk_prep_kernel(): pass
def _schedule_topk_selected_extra_kernel(): pass
def _schedule_topk_finalize_kernel(): pass
def schedule_verify_lens_topk_triton():
    BLOCK_C = 64
    BLOCK_CP = 256
    BLOCK = 256
""",
                encoding="utf-8",
            )
            info = tuner.inspect_production_source(path)
        self.assertEqual(info["production_defaults"]["BLOCK_C"], 64)
        self.assertEqual(info["production_defaults"]["BLOCK_CP"], 256)
        self.assertEqual(info["production_defaults"]["BLOCK"], 256)
        self.assertEqual(info["api_variant"], "split_select_finalize")
        self.assertEqual(len(info["sha256"]), 64)


class TestRuntimeLoading(unittest.TestCase):
    def test_loads_nested_dspark_module_from_requested_source(self):
        with tempfile.TemporaryDirectory() as raw:
            source = Path(raw) / "sglang"
            source_python = source / "python"
            schedule_path = source_python / tuner.SCHEDULE_RELATIVE_PATH.removeprefix(
                "python/"
            )
            planner_path = (
                source_python
                / "sglang/srt/speculative/dspark_components/dspark_planner.py"
            )
            schedule_path.parent.mkdir(parents=True)
            planner_path.parent.mkdir(parents=True)
            schedule_path.write_text(
                """
def _schedule_topk_prep_kernel(): pass
def _schedule_topk_selected_extra_kernel(): pass
def _schedule_topk_finalize_kernel(): pass
def schedule_verify_lens_topk_triton():
    _schedule_topk_selected_extra_kernel()
    _schedule_topk_finalize_kernel()
""",
                encoding="utf-8",
            )
            planner_path.touch()

            schedule = types.SimpleNamespace(
                __file__=str(schedule_path),
                __name__=tuner.DSPARK_SCHEDULE_MODULE,
                _schedule_topk_prep_kernel=object(),
                _schedule_topk_selected_extra_kernel=object(),
                _schedule_topk_finalize_kernel=object(),
                schedule_verify_lens_topk_triton=object(),
            )
            planner = types.SimpleNamespace(
                __file__=str(planner_path),
                __name__=tuner.DSPARK_PLANNER_MODULE,
                DSparkScheduleConfig=object(),
            )
            modules = {
                "torch": object(),
                "triton": object(),
                tuner.DSPARK_SCHEDULE_MODULE: schedule,
                tuner.DSPARK_PLANNER_MODULE: planner,
            }
            old_path = list(tuner.sys.path)
            old_sglang_modules = {
                name: module
                for name, module in tuner.sys.modules.items()
                if name == "sglang" or name.startswith("sglang.")
            }
            self.addCleanup(setattr, tuner.sys, "path", old_path)

            def restore_sglang_modules():
                for name in tuple(tuner.sys.modules):
                    if name == "sglang" or name.startswith("sglang."):
                        del tuner.sys.modules[name]
                tuner.sys.modules.update(old_sglang_modules)

            self.addCleanup(restore_sglang_modules)
            with mock.patch.object(
                tuner.importlib,
                "import_module",
                side_effect=lambda name: modules[name],
            ) as import_module:
                runtime = tuner._load_runtime(source)

        self.assertIs(runtime.schedule_module, schedule)
        self.assertEqual(runtime.api_variant, "split_select_finalize")
        self.assertEqual(
            [call.args[0] for call in import_module.call_args_list],
            [
                "torch",
                "triton",
                tuner.DSPARK_SCHEDULE_MODULE,
                tuner.DSPARK_PLANNER_MODULE,
            ],
        )

    def test_inspection_uses_wrapper_api_when_both_apis_exist(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "dspark_schedule.py"
            path.write_text(
                """
def _schedule_topk_prep_kernel(): pass
def _schedule_topk_selected_extra_kernel(): pass
def _schedule_topk_finalize_kernel(): pass
def _schedule_topk_select_and_finalize_kernel(): pass
def _schedule_topk_all_valid_finalize_kernel(): pass
def schedule_verify_lens_topk_triton():
    _schedule_topk_selected_extra_kernel()
    _schedule_topk_finalize_kernel()
""",
                encoding="utf-8",
            )
            info = tuner.inspect_production_source(path)

        self.assertEqual(info["api_variant"], "split_select_finalize")


class _FakeTensor:
    def __init__(self, shape, *, dtype="float32", device="cuda:0"):
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.dtype = dtype
        self.device = device

    def contiguous(self):
        return self


class _FakeKernel:
    def __init__(self):
        self.grids = []
        self.calls = []

    def __getitem__(self, grid):
        self.grids.append(grid)

        def invoke(*args, **kwargs):
            self.calls.append((args, kwargs))

        return invoke


class TestCandidateLauncher(unittest.TestCase):
    def test_launcher_forwards_candidate_meta_to_fused_production_kernel(self):
        prep = _FakeKernel()
        fused = _FakeKernel()
        all_valid = _FakeKernel()

        class FakeTorch:
            int32 = "int32"
            float32 = "float32"

            @staticmethod
            def empty(shape, *, dtype, device):
                return _FakeTensor(shape, dtype=dtype, device=device)

        schedule = types.SimpleNamespace(
            torch=FakeTorch,
            _schedule_topk_prep_kernel=prep,
            _schedule_topk_select_and_finalize_kernel=fused,
            _schedule_topk_all_valid_finalize_kernel=all_valid,
        )
        triton = types.SimpleNamespace(
            cdiv=lambda value, divisor: (value + divisor - 1) // divisor,
            next_power_of_2=lambda value: 1 << (value - 1).bit_length(),
        )
        cfg = types.SimpleNamespace(
            min_verify_len=1,
            survival_eps=1e-6,
            resolved_max_verify_len=lambda: 6,
        )
        candidate = tuner.ScheduleCandidate(16, 128, 8)
        got = tuner.launch_schedule_candidate(
            confidence=_FakeTensor((8, 5)),
            budget=8,
            cfg=cfg,
            candidate=candidate,
            schedule_module=schedule,
            triton_module=triton,
        )

        self.assertEqual(got.shape, (8,))
        self.assertEqual(prep.grids, [(8,)])
        self.assertEqual(fused.grids, [(8,)])
        self.assertEqual(fused.calls[0][1]["BLOCK_C"], 16)
        self.assertEqual(fused.calls[0][1]["BLOCK_CP"], 128)
        self.assertEqual(fused.calls[0][1]["num_warps"], 8)
        self.assertEqual(all_valid.calls, [])

    def test_launcher_forwards_candidate_meta_to_production_kernel(self):
        prep = _FakeKernel()
        selected = _FakeKernel()
        finalize = _FakeKernel()

        class FakeTorch:
            int32 = "int32"
            float32 = "float32"

            @staticmethod
            def empty(shape, *, dtype, device):
                return _FakeTensor(shape, dtype=dtype, device=device)

        schedule = types.SimpleNamespace(
            torch=FakeTorch,
            _schedule_topk_prep_kernel=prep,
            _schedule_topk_selected_extra_kernel=selected,
            _schedule_topk_finalize_kernel=finalize,
        )
        triton = types.SimpleNamespace(
            cdiv=lambda value, divisor: (value + divisor - 1) // divisor,
            next_power_of_2=lambda value: 1 << (value - 1).bit_length(),
        )
        confidence = _FakeTensor((8, 5))
        cfg = types.SimpleNamespace(
            min_verify_len=1,
            survival_eps=1e-6,
            resolved_max_verify_len=lambda: 6,
        )
        candidate = tuner.ScheduleCandidate(32, 128, 8)
        got = tuner.launch_schedule_candidate(
            confidence=confidence,
            budget=8,
            cfg=cfg,
            candidate=candidate,
            schedule_module=schedule,
            triton_module=triton,
        )
        self.assertEqual(got.shape, (8,))
        self.assertEqual(selected.grids, [(2,)])
        self.assertEqual(selected.calls[0][1]["BLOCK_C"], 32)
        self.assertEqual(selected.calls[0][1]["BLOCK_CP"], 128)
        self.assertEqual(selected.calls[0][1]["num_warps"], 8)
        self.assertEqual(prep.grids, [(8,)])
        self.assertEqual(finalize.grids, [(1,)])

    def test_launcher_skips_selection_kernel_for_zero_budget(self):
        prep = _FakeKernel()
        selected = _FakeKernel()
        finalize = _FakeKernel()

        class FakeTorch:
            int32 = "int32"
            float32 = "float32"

            @staticmethod
            def empty(shape, *, dtype, device):
                return _FakeTensor(shape, dtype=dtype, device=device)

            @staticmethod
            def full(shape, value, *, dtype, device):
                return _FakeTensor(shape, dtype=dtype, device=device)

        schedule = types.SimpleNamespace(
            torch=FakeTorch,
            _schedule_topk_prep_kernel=prep,
            _schedule_topk_selected_extra_kernel=selected,
            _schedule_topk_finalize_kernel=finalize,
        )
        triton = types.SimpleNamespace(
            cdiv=lambda value, divisor: 1, next_power_of_2=lambda value: 8
        )
        cfg = types.SimpleNamespace(
            min_verify_len=1,
            survival_eps=1e-6,
            resolved_max_verify_len=lambda: 6,
        )
        tuner.launch_schedule_candidate(
            confidence=_FakeTensor((1, 5)),
            budget=0,
            cfg=cfg,
            candidate=tuner.ScheduleCandidate(16, 64, 2),
            schedule_module=schedule,
            triton_module=triton,
        )
        self.assertEqual(selected.calls, [])
        self.assertEqual(prep.calls, [])

    def test_launcher_rejects_incomplete_fused_api(self):
        class FakeTorch:
            int32 = "int32"
            float32 = "float32"

            @staticmethod
            def empty(shape, *, dtype, device):
                return _FakeTensor(shape, dtype=dtype, device=device)

        schedule = types.SimpleNamespace(
            torch=FakeTorch,
            _schedule_topk_prep_kernel=_FakeKernel(),
            _schedule_topk_select_and_finalize_kernel=_FakeKernel(),
        )
        triton = types.SimpleNamespace(next_power_of_2=lambda value: 8)
        cfg = types.SimpleNamespace(
            min_verify_len=1,
            survival_eps=1e-6,
            resolved_max_verify_len=lambda: 6,
        )
        with self.assertRaisesRegex(RuntimeError, "incomplete fused"):
            tuner.launch_schedule_candidate(
                confidence=_FakeTensor((8, 5)),
                budget=8,
                cfg=cfg,
                candidate=tuner.ScheduleCandidate(8, 256, 4),
                schedule_module=schedule,
                triton_module=triton,
            )


class TestRankingAndPlan(unittest.TestCase):
    def _row(
        self,
        *,
        case,
        candidate_id,
        variant,
        median,
        valid=True,
        speedup=None,
        speedup_samples=None,
        **meta,
    ):
        if speedup is not None and speedup_samples is None:
            speedup_samples = [speedup] * 4
        return {
            "case_name": case.name,
            "variant": variant,
            "candidate_id": candidate_id,
            "status": "ok",
            "valid": valid,
            "median_us": median,
            "paired_speedup_geomean": speedup,
            "paired_speedup_samples": speedup_samples,
            "BLOCK_C": meta.get("BLOCK_C"),
            "BLOCK_CP": meta.get("BLOCK_CP"),
            "num_warps": meta.get("num_warps"),
        }

    def test_invalid_fast_candidate_is_not_eligible(self):
        cases = [
            tuner.ScheduleCase("a", 8, 5, 1, 0, 8, "random", 1),
            tuner.ScheduleCase("b", 32, 5, 1, 0, 80, "random", 2),
        ]
        rows = []
        for case in cases:
            rows.append(
                self._row(
                    case=case,
                    candidate_id="production-default",
                    variant="production_default",
                    median=10.0,
                )
            )
            rows.append(
                self._row(
                    case=case,
                    candidate_id="bc32-bcp128-w4",
                    variant="candidate",
                    median=5.0,
                    speedup=2.0,
                    BLOCK_C=32,
                    BLOCK_CP=128,
                    num_warps=4,
                )
            )
            rows.append(
                self._row(
                    case=case,
                    candidate_id="bc64-bcp256-w8",
                    variant="candidate",
                    median=1.0,
                    speedup=10.0,
                    valid=case.name == "a",
                    BLOCK_C=64,
                    BLOCK_CP=256,
                    num_warps=8,
                )
            )
        ranked = tuner._rank_candidates(rows=rows, cases=cases)
        self.assertEqual(ranked[0]["candidate_id"], "bc32-bcp128-w4")
        self.assertTrue(ranked[0]["eligible"])
        self.assertTrue(ranked[0]["promotable"])
        self.assertFalse(ranked[1]["eligible"])

    def test_all_slower_candidates_produce_no_promotable_result(self):
        cases = [
            tuner.ScheduleCase("a", 8, 5, 1, 0, 8, "random", 1),
            tuner.ScheduleCase("b", 32, 5, 1, 0, 80, "random", 2),
        ]
        rows = []
        for case in cases:
            rows.append(
                self._row(
                    case=case,
                    candidate_id="bc32-bcp256-w2",
                    variant="candidate",
                    median=10.2,
                    speedup=0.98,
                    BLOCK_C=32,
                    BLOCK_CP=256,
                    num_warps=2,
                )
            )
        ranked = tuner._rank_candidates(rows=rows, cases=cases)
        self.assertTrue(ranked[0]["eligible"])
        self.assertFalse(ranked[0]["promotable"])
        self.assertLess(ranked[0]["geomean_speedup"], 1.0)

    def test_noisy_paired_samples_cannot_collapse_to_zero_width_ci(self):
        cases = [
            tuner.ScheduleCase("a", 8, 5, 1, 0, 8, "random", 1),
            tuner.ScheduleCase("b", 32, 5, 1, 0, 80, "random", 2),
        ]
        rows = [
            self._row(
                case=case,
                candidate_id="bc32-bcp128-w4",
                variant="candidate",
                median=9.6,
                speedup=1.039,
                speedup_samples=[0.9, 1.2],
                BLOCK_C=32,
                BLOCK_CP=128,
                num_warps=4,
            )
            for case in cases
        ]

        ranked = tuner._rank_candidates(rows=rows, cases=cases)

        self.assertTrue(ranked[0]["eligible"])
        self.assertTrue(ranked[0]["raw_samples_complete"])
        self.assertGreater(ranked[0]["geomean_speedup"], 1.01)
        self.assertLess(ranked[0]["speedup_ci_low"], 1.0)
        self.assertFalse(ranked[0]["promotable"])

    def test_missing_raw_samples_never_produces_winner(self):
        case = tuner.ScheduleCase("a", 8, 5, 1, 0, 8, "random", 1)
        row = self._row(
            case=case,
            candidate_id="bc32-bcp128-w4",
            variant="candidate",
            median=5.0,
            speedup=2.0,
            speedup_samples=[],
            BLOCK_C=32,
            BLOCK_CP=128,
            num_warps=4,
        )

        ranked = tuner._rank_candidates(rows=[row], cases=[case])

        self.assertFalse(ranked[0]["eligible"])
        self.assertFalse(ranked[0]["raw_samples_complete"])
        self.assertFalse(ranked[0]["promotable"])

    def test_log_speedup_summary_reports_finite_confidence_interval(self):
        summary = tuner.log_speedup_summary([1.02, 1.03, 1.01, 1.04, 1.02])
        self.assertEqual(summary["samples"], 5)
        self.assertLess(summary["ci_low"], summary["geomean"])
        self.assertGreater(summary["ci_high"], summary["geomean"])

    def test_plan_only_writes_cpu_plan_and_never_loads_runtime(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "sglang"
            schedule_path = source / tuner.SCHEDULE_RELATIVE_PATH
            schedule_path.parent.mkdir(parents=True)
            schedule_path.write_text(
                """
def _schedule_topk_prep_kernel(): pass
def _schedule_topk_selected_extra_kernel(): pass
def _schedule_topk_finalize_kernel(): pass
def schedule_verify_lens_topk_triton():
    BLOCK_C = 64
    BLOCK_CP = 256
    BLOCK = 256
""",
                encoding="utf-8",
            )
            output = root / "out"
            with (
                mock.patch.object(
                    tuner, "_load_runtime", side_effect=AssertionError("runtime loaded")
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                code = tuner.main(
                    [
                        "--source",
                        str(source),
                        "--output",
                        str(output),
                        "--case-set",
                        "smoke",
                        "--plan-only",
                    ]
                )
            plan = json.loads(
                (output / "dspark_schedule_plan.json").read_text(encoding="utf-8")
            )
        self.assertEqual(code, 0)
        self.assertEqual(plan["candidate_count"], 36)
        self.assertEqual(len(plan["cases"]), 3)
        self.assertFalse((output / "dspark_schedule_measurements.csv").exists())
        self.assertFalse(plan["timed_path"]["torch_reference_timed"])
        self.assertTrue(plan["timing_policy"]["paired_ab_ba"])
        self.assertTrue(
            plan["promotion_policy"]["independent_confirmation_required_before_install"]
        )


class TestPairedTiming(unittest.TestCase):
    def test_measurement_alternates_ab_ba_with_common_repeat_count(self):
        baseline = object()
        candidate = object()
        torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(synchronize=lambda: None)
        )
        order = []

        def fake_batch(_torch, launcher, repeats):
            order.append((launcher, repeats))
            return 1.0 if launcher is baseline else 0.9

        with (
            mock.patch.object(
                tuner,
                "_calibrate_inner_repeats",
                side_effect=[4, 8],
            ),
            mock.patch.object(tuner, "_event_batch_time_ms", side_effect=fake_batch),
        ):
            result = tuner.measure_paired_launchers(
                torch,
                baseline,
                candidate,
                warmup=0,
                iters=4,
                timing_min_ms=5.0,
                timing_max_ms=20.0,
                candidate_first=True,
            )

        self.assertEqual(
            [launcher for launcher, _ in order],
            [
                candidate,
                baseline,
                baseline,
                candidate,
                candidate,
                baseline,
                baseline,
                candidate,
            ],
        )
        self.assertTrue(all(repeats == 8 for _, repeats in order))
        self.assertEqual(len(result.speedup_samples), 4)
        self.assertGreater(result.speedup_geomean, 1.0)

    def test_measurement_rejects_odd_pair_count(self):
        torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(synchronize=lambda: None)
        )
        with self.assertRaisesRegex(ValueError, "even sample count"):
            tuner.measure_paired_launchers(
                torch,
                object(),
                object(),
                warmup=0,
                iters=3,
                timing_min_ms=5.0,
                timing_max_ms=20.0,
                candidate_first=True,
            )


if __name__ == "__main__":
    unittest.main()
