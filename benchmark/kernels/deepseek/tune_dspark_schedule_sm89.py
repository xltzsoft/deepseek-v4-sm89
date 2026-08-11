#!/usr/bin/env python3
"""Independent SM89 DSpark schedule-kernel benchmark and tuner.

The harness imports the production Triton kernels from SGLang and varies only
their launch meta parameters. Torch is used for exact output validation outside
the timed region; no Torch reference path is included in latency measurements.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib
import itertools
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


SCHEDULE_RELATIVE_PATH = (
    "python/sglang/kernels/ops/speculative/dspark/dspark_schedule.py"
)
COMMON_PRODUCTION_SYMBOLS = (
    "_schedule_topk_prep_kernel",
    "schedule_verify_lens_topk_triton",
)
FUSED_PRODUCTION_SYMBOL = "_schedule_topk_select_and_finalize_kernel"
ALL_VALID_PRODUCTION_SYMBOL = "_schedule_topk_all_valid_finalize_kernel"
SPLIT_PRODUCTION_SYMBOLS = (
    "_schedule_topk_selected_extra_kernel",
    "_schedule_topk_finalize_kernel",
)
DEFAULT_BLOCK_C = (8, 16, 32, 64)
DEFAULT_BLOCK_CP = (64, 128, 256)
DEFAULT_NUM_WARPS = (2, 4, 8)
MAX_INNER_REPEATS = 1 << 20
CALIBRATION_STEPS = 8
DEFAULT_PROMOTION_MIN_SPEEDUP = 1.01
DEFAULT_PROMOTION_MIN_CI = 1.0
DEFAULT_PROMOTION_MIN_WORST_SPEEDUP = 0.98
DSPARK_SCHEDULE_MODULE = "sglang.kernels.ops.speculative.dspark.dspark_schedule"
DSPARK_PLANNER_MODULE = "sglang.srt.speculative.dspark_components.dspark_planner"


@dataclass(frozen=True)
class ScheduleCandidate:
    block_c: int
    block_cp: int
    num_warps: int

    @property
    def candidate_id(self) -> str:
        return f"bc{self.block_c}-bcp{self.block_cp}-w{self.num_warps}"

    def as_json(self) -> dict[str, int | str]:
        return {
            "candidate_id": self.candidate_id,
            "BLOCK_C": self.block_c,
            "BLOCK_CP": self.block_cp,
            "num_warps": self.num_warps,
        }


@dataclass(frozen=True)
class ScheduleCase:
    name: str
    batch_size: int
    gamma: int
    min_verify_len: int
    max_verify_len: int
    budget: int
    pattern: str
    seed: int

    def as_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["resolved_max_verify_len"] = self.max_verify_len or self.gamma + 1
        payload["candidate_slots"] = self.batch_size * min(
            self.gamma, payload["resolved_max_verify_len"]
        )
        payload["tunable_budget"] = self.budget > 0 and payload["candidate_slots"] > 0
        return payload


@dataclass(frozen=True)
class TimingStats:
    samples: int
    median_us: float
    p20_us: float
    p80_us: float
    mean_us: float
    stdev_us: float
    inner_repeats: int


@dataclass(frozen=True)
class PairedTimingStats:
    baseline: TimingStats
    candidate: TimingStats
    pair_samples: int
    speedup_samples: tuple[float, ...]
    speedup_median: float
    speedup_geomean: float
    speedup_ci_low: float
    speedup_ci_high: float


@dataclass(frozen=True)
class Runtime:
    torch: Any
    triton: Any
    schedule_module: Any
    config_type: Any
    api_variant: str


REPRESENTATIVE_CASES: tuple[ScheduleCase, ...] = (
    ScheduleCase("g5_bs1_budget1", 1, 5, 1, 0, 1, "random", 1001),
    ScheduleCase("g5_bs4_budget5", 4, 5, 1, 0, 5, "random", 1002),
    ScheduleCase("g5_bs8_budget10", 8, 5, 1, 0, 10, "quantized", 1003),
    ScheduleCase("g5_bs16_budget20", 16, 5, 1, 0, 20, "sparse", 1004),
    ScheduleCase("g5_bs32_budget40", 32, 5, 1, 0, 40, "random", 1005),
    ScheduleCase("g5_bs48_budget120", 48, 5, 1, 0, 120, "quantized", 1006),
    ScheduleCase("g5_bs64_budget0_ties", 64, 5, 1, 0, 0, "ties", 1007),
    ScheduleCase("g5_bs64_budget7_quantized", 64, 5, 1, 0, 7, "quantized", 1008),
    ScheduleCase("g7_bs64_budget224", 64, 7, 1, 0, 224, "sparse", 1009),
    ScheduleCase("g7_bs128_budget384", 128, 7, 1, 0, 384, "random", 1010),
    ScheduleCase("g7_bs192_budget320", 192, 7, 1, 0, 320, "quantized", 1011),
    ScheduleCase("g7_bs256_budget768", 256, 7, 1, 0, 768, "random", 1012),
)
SMOKE_CASES: tuple[ScheduleCase, ...] = (
    REPRESENTATIVE_CASES[0],
    REPRESENTATIVE_CASES[4],
    REPRESENTATIVE_CASES[6],
)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def validate_candidate(candidate: ScheduleCandidate) -> None:
    if not _is_power_of_two(candidate.block_c) or not 8 <= candidate.block_c <= 256:
        raise ValueError("BLOCK_C must be a power of two in [8, 256]")
    if not _is_power_of_two(candidate.block_cp) or not 16 <= candidate.block_cp <= 1024:
        raise ValueError("BLOCK_CP must be a power of two in [16, 1024]")
    if candidate.num_warps not in (1, 2, 4, 8, 16):
        raise ValueError("num_warps must be one of 1, 2, 4, 8, 16")


def generate_candidates(
    block_c: Sequence[int] = DEFAULT_BLOCK_C,
    block_cp: Sequence[int] = DEFAULT_BLOCK_CP,
    num_warps: Sequence[int] = DEFAULT_NUM_WARPS,
) -> list[ScheduleCandidate]:
    candidates: list[ScheduleCandidate] = []
    seen: set[tuple[int, int, int]] = set()
    for values in itertools.product(block_c, block_cp, num_warps):
        candidate = ScheduleCandidate(*map(int, values))
        validate_candidate(candidate)
        key = (candidate.block_c, candidate.block_cp, candidate.num_warps)
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)
    if not candidates:
        raise ValueError("candidate search space is empty")
    return candidates


def select_cases(case_set: str, names: Sequence[str] = ()) -> list[ScheduleCase]:
    if case_set not in {"smoke", "representative"}:
        raise ValueError(f"unknown case set: {case_set}")
    source = SMOKE_CASES if case_set == "smoke" else REPRESENTATIVE_CASES
    if not names:
        return list(source)
    by_name = {case.name: case for case in source}
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ValueError(f"unknown schedule case(s): {', '.join(unknown)}")
    return [by_name[name] for name in names]


def _constant_assignments(function: ast.FunctionDef) -> dict[str, int | str]:
    values: dict[str, int | str] = {}
    for node in ast.walk(function):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not isinstance(node.value, ast.Constant) or not isinstance(
            node.value.value, int
        ):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                values[target.id] = int(node.value.value)
    return values


def _constant_call_keywords(function: ast.FunctionDef) -> dict[str, int]:
    values: dict[str, int] = {}
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if (
                keyword.arg
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, int)
            ):
                values[keyword.arg] = int(keyword.value.value)
    return values


def _select_api_variant(
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
) -> tuple[str, tuple[str, ...]]:
    wrapper = functions["schedule_verify_lens_topk_triton"]
    referenced_names = {
        node.id for node in ast.walk(wrapper) if isinstance(node, ast.Name)
    }
    fused_defined = all(
        name in functions
        for name in (FUSED_PRODUCTION_SYMBOL, ALL_VALID_PRODUCTION_SYMBOL)
    )
    split_defined = all(name in functions for name in SPLIT_PRODUCTION_SYMBOLS)
    fused_referenced = bool(
        referenced_names & {FUSED_PRODUCTION_SYMBOL, ALL_VALID_PRODUCTION_SYMBOL}
    )
    split_referenced = bool(referenced_names & set(SPLIT_PRODUCTION_SYMBOLS))

    if fused_referenced and split_referenced:
        raise ValueError("production wrapper references both fused and split APIs")
    if fused_referenced:
        if not fused_defined:
            raise ValueError("production wrapper references an incomplete fused API")
        variant = "fused_select_finalize"
        required = (FUSED_PRODUCTION_SYMBOL, ALL_VALID_PRODUCTION_SYMBOL)
    elif split_referenced:
        if not split_defined:
            raise ValueError("production wrapper references an incomplete split API")
        variant = "split_select_finalize"
        required = SPLIT_PRODUCTION_SYMBOLS
    elif fused_defined and not split_defined:
        variant = "fused_select_finalize"
        required = (FUSED_PRODUCTION_SYMBOL, ALL_VALID_PRODUCTION_SYMBOL)
    elif split_defined and not fused_defined:
        variant = "split_select_finalize"
        required = SPLIT_PRODUCTION_SYMBOLS
    elif fused_defined and split_defined:
        raise ValueError(
            "production source defines both APIs but the wrapper selects neither"
        )
    else:
        raise ValueError(
            "production schedule source has neither the fused nor split selection API"
        )
    return variant, (*COMMON_PRODUCTION_SYMBOLS, *required)


def inspect_production_source(source: Path) -> dict[str, Any]:
    """Inspect and fingerprint the production schedule module without importing it."""
    source = source.resolve()
    if not source.is_file():
        raise ValueError(f"production schedule source does not exist: {source}")
    text = source.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(source))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing = [name for name in COMMON_PRODUCTION_SYMBOLS if name not in functions]
    if missing:
        raise ValueError(f"production schedule source is missing symbols: {missing}")
    api_variant, required_symbols = _select_api_variant(functions)
    wrapper_defaults = _constant_assignments(
        functions["schedule_verify_lens_topk_triton"]
    )
    for name, value in _constant_call_keywords(
        functions["schedule_verify_lens_topk_triton"]
    ).items():
        wrapper_defaults.setdefault(name, value)
    if api_variant == "fused_select_finalize":
        wrapper_defaults.setdefault("BLOCK_C", "next_power_of_2(cols)")
        wrapper_defaults.setdefault("BLOCK_R", "next_power_of_2(num_requests)")
    return {
        "path": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "api_variant": api_variant,
        "required_symbols": list(required_symbols),
        "production_defaults": {
            "BLOCK_C": wrapper_defaults.get("BLOCK_C"),
            "BLOCK_CP": wrapper_defaults.get("BLOCK_CP"),
            "BLOCK": wrapper_defaults.get("BLOCK"),
            "BLOCK_R": wrapper_defaults.get("BLOCK_R"),
            "num_warps": None,
        },
    }


def _schedule_source(source: Path) -> Path:
    return (source / SCHEDULE_RELATIVE_PATH).resolve()


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source.resolve()
    source_info = inspect_production_source(_schedule_source(source))
    cases = select_cases(args.case_set, args.case)
    candidates = generate_candidates(args.block_c, args.block_cp, args.num_warps)
    return {
        "schema_version": 1,
        "runner": "dspark_schedule_sm89",
        "target": {
            "device_name": args.device_name,
            "compute_capability": [8, 9],
            "source_root": str(source),
        },
        "production_source": source_info,
        "candidate_axes": {
            "BLOCK_C": [int(value) for value in args.block_c],
            "BLOCK_CP": [int(value) for value in args.block_cp],
            "num_warps": [int(value) for value in args.num_warps],
        },
        "candidate_count": len(candidates),
        "candidates": [candidate.as_json() for candidate in candidates],
        "case_set": args.case_set,
        "seed_offset": args.seed,
        "cases": [case.as_json() for case in cases],
        "timed_path": {
            "kind": "production_triton_only",
            "api_variant": source_info["api_variant"],
            "candidate_launcher": "launch_schedule_candidate",
            "production_baseline": "ScheduleVerifyLensTopk.triton",
            "torch_reference_timed": False,
        },
        "reference_policy": {
            "path": "ScheduleVerifyLensTopk.torch",
            "comparison": "torch.equal_exact",
            "used_for_latency": False,
        },
        "timing_policy": {
            "single_process_serial": True,
            "paired_ab_ba": True,
            "candidate_order": "deterministic_shuffle_per_case",
            "one_cuda_event_pair_per_variant_per_sample": True,
            "warmup": args.warmup,
            "samples": args.iters,
            "timing_min_ms": args.timing_min_ms,
            "timing_max_ms": args.timing_max_ms,
        },
        "promotion_policy": {
            "min_geomean_speedup": args.promotion_min_speedup,
            "min_confidence_bound": args.promotion_min_ci,
            "min_worst_case_speedup": args.promotion_min_worst_speedup,
            "confidence_level": 0.95,
            "independent_confirmation_required_before_install": True,
        },
    }


def _load_runtime(source: Path) -> Runtime:
    source_python = (source / "python").resolve()
    if not source_python.is_dir():
        raise ValueError(f"SGLang Python source does not exist: {source_python}")
    source_python_str = str(source_python)
    sys.path[:] = [path for path in sys.path if path != source_python_str]
    sys.path.insert(0, source_python_str)

    # The queue runs beside editable SGLang installs from older source snapshots.
    # Reload the package deterministically from the source requested by this run.
    for name in tuple(sys.modules):
        if name == "sglang" or name.startswith("sglang."):
            del sys.modules[name]
    importlib.invalidate_caches()

    torch = importlib.import_module("torch")
    triton = importlib.import_module("triton")
    dspark_schedule = importlib.import_module(DSPARK_SCHEDULE_MODULE)
    planner_module = importlib.import_module(DSPARK_PLANNER_MODULE)
    DSparkScheduleConfig = planner_module.DSparkScheduleConfig

    for module in (dspark_schedule, planner_module):
        origin = Path(module.__file__).resolve()
        try:
            origin.relative_to(source_python)
        except ValueError as exc:
            raise RuntimeError(
                f"loaded {module.__name__} from {origin}, expected {source_python}"
            ) from exc

    source_info = inspect_production_source(Path(dspark_schedule.__file__))
    for name in source_info["required_symbols"]:
        if not hasattr(dspark_schedule, name):
            raise RuntimeError(f"loaded schedule module lacks production symbol {name}")
    return Runtime(
        torch=torch,
        triton=triton,
        schedule_module=dspark_schedule,
        config_type=DSparkScheduleConfig,
        api_variant=source_info["api_variant"],
    )


def launch_schedule_candidate(
    *,
    confidence: Any,
    budget: int,
    cfg: Any,
    candidate: ScheduleCandidate,
    schedule_module: Any | None = None,
    triton_module: Any | None = None,
    api_variant: str | None = None,
) -> Any:
    """Invoke the production schedule kernels with candidate launch metadata.

    This is intentionally importable for external profilers and tests. The
    kernel bodies remain in SGLang; this function only mirrors the production
    wrapper so the middle kernel can receive searched constexpr values.
    """
    if schedule_module is None:
        schedule_module = importlib.import_module(DSPARK_SCHEDULE_MODULE)
    if triton_module is None:
        import triton as triton_module

    torch = schedule_module.torch
    num_requests, gamma = confidence.shape
    max_len = cfg.resolved_max_verify_len()
    device = confidence.device
    cols = min(max_len, gamma)
    n = num_requests * cols
    if num_requests == 0:
        return torch.empty(0, dtype=torch.int32, device=device)
    lower_bound = max(int(cfg.min_verify_len), 1)
    if budget <= 0 or n <= 0:
        return torch.full(
            (num_requests,),
            min(lower_bound, int(max_len)),
            dtype=torch.int32,
            device=device,
        )

    prep = schedule_module._schedule_topk_prep_kernel
    survival = torch.empty((num_requests, cols), dtype=torch.float32, device=device)
    has_fused = hasattr(schedule_module, FUSED_PRODUCTION_SYMBOL)
    has_all_valid = hasattr(schedule_module, ALL_VALID_PRODUCTION_SYMBOL)
    if has_fused != has_all_valid:
        raise RuntimeError(
            "loaded schedule module has an incomplete fused selection API"
        )
    has_split = all(hasattr(schedule_module, name) for name in SPLIT_PRODUCTION_SYMBOLS)
    if api_variant is None:
        if has_fused and has_split:
            raise RuntimeError(
                "loaded schedule module exposes both APIs; api_variant is required"
            )
        if has_fused:
            api_variant = "fused_select_finalize"
        elif has_split:
            api_variant = "split_select_finalize"
        else:
            raise RuntimeError("loaded schedule module has no complete selection API")
    if api_variant not in {"fused_select_finalize", "split_select_finalize"}:
        raise ValueError(f"unknown schedule api_variant: {api_variant}")
    if api_variant == "fused_select_finalize":
        if not has_fused:
            raise RuntimeError("loaded schedule module lacks the fused selection API")
        # The optimized production API fuses selection and finalization. Its
        # per-row BLOCK_C must cover the whole candidate window for exactness.
        required_block_c = triton_module.next_power_of_2(max(cols, 1))
        if candidate.block_c < required_block_c:
            raise ValueError(
                f"BLOCK_C={candidate.block_c} is smaller than cols={cols} "
                f"(requires >= {required_block_c})"
            )
        valid_counts = torch.empty(num_requests, dtype=torch.int32, device=device)
        prep[(num_requests,)](
            confidence.contiguous(),
            survival,
            valid_counts,
            gamma,
            cols,
            float(cfg.survival_eps),
            G_P2=triton_module.next_power_of_2(max(gamma, 1)),
        )
        verify_lens = torch.empty(num_requests, dtype=torch.int32, device=device)
        if budget >= n:
            block = 256
            schedule_module._schedule_topk_all_valid_finalize_kernel[
                (triton_module.cdiv(num_requests, block),)
            ](
                valid_counts,
                verify_lens,
                int(cfg.min_verify_len),
                lower_bound,
                int(max_len),
                num_requests,
                BLOCK=block,
            )
        else:
            schedule_module._schedule_topk_select_and_finalize_kernel[(num_requests,)](
                survival,
                verify_lens,
                int(budget),
                cols,
                n,
                float(cfg.survival_eps),
                int(cfg.min_verify_len),
                lower_bound,
                int(max_len),
                BLOCK_C=candidate.block_c,
                BLOCK_CP=candidate.block_cp,
                num_warps=candidate.num_warps,
            )
        return verify_lens

    # Compatibility path for the earlier split production wrapper.
    if not has_split:
        raise RuntimeError("loaded schedule module lacks the split selection API")
    selected_extra = torch.empty(num_requests, dtype=torch.int32, device=device)
    prep[(num_requests,)](
        confidence.contiguous(),
        survival,
        selected_extra,
        gamma,
        cols,
        G_P2=triton_module.next_power_of_2(max(gamma, 1)),
    )
    if budget > 0 and n > 0:
        schedule_module._schedule_topk_selected_extra_kernel[
            (triton_module.cdiv(n, candidate.block_c),)
        ](
            survival,
            selected_extra,
            int(budget),
            cols,
            n,
            float(cfg.survival_eps),
            BLOCK_C=candidate.block_c,
            BLOCK_CP=candidate.block_cp,
            num_warps=candidate.num_warps,
        )
    verify_lens = torch.empty(num_requests, dtype=torch.int32, device=device)
    block = 256
    schedule_module._schedule_topk_finalize_kernel[
        (triton_module.cdiv(num_requests, block),)
    ](
        selected_extra,
        verify_lens,
        int(cfg.min_verify_len),
        max(int(cfg.min_verify_len), 1),
        int(max_len),
        num_requests,
        BLOCK=block,
    )
    return verify_lens


def launch_production_schedule(
    *, schedule_module: Any, confidence: Any, budget: int, cfg: Any
) -> Any:
    """Call the unmodified production wrapper as the baseline path."""
    return schedule_module.ScheduleVerifyLensTopk.triton(
        confidence=confidence, budget=budget, cfg=cfg
    )


def _make_confidence(torch: Any, case: ScheduleCase, *, seed_offset: int = 0) -> Any:
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(case.seed + seed_offset)
    shape = (case.batch_size, case.gamma)
    if case.pattern == "ties":
        return torch.full(shape, 0.5, dtype=torch.float32, device=device)
    base = torch.rand(shape, dtype=torch.float32, device=device, generator=generator)
    if case.pattern == "quantized":
        return (base * 4.0).floor() / 4.0
    if case.pattern == "sparse":
        return torch.where(base < 0.35, torch.zeros_like(base), base)
    if case.pattern == "random":
        return base.clamp_min(0.02)
    raise ValueError(f"unknown confidence pattern: {case.pattern}")


def _make_config(runtime: Runtime, case: ScheduleCase) -> Any:
    config = runtime.config_type(
        gamma=case.gamma,
        min_verify_len=case.min_verify_len,
        max_verify_len=case.max_verify_len,
        survival_eps=1e-6,
    )
    validate = getattr(config, "validate", None)
    if validate is not None:
        validate()
    return config


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute percentile of an empty sample set")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def choose_inner_repeats(
    *,
    current_repeats: int,
    elapsed_ms: float,
    timing_min_ms: float,
    timing_max_ms: float,
    max_repeats: int = MAX_INNER_REPEATS,
) -> int:
    if current_repeats <= 0 or max_repeats <= 0:
        raise ValueError("repeat counts must be positive")
    if timing_min_ms <= 0 or timing_max_ms <= timing_min_ms:
        raise ValueError("timing window must satisfy 0 < min < max")
    if timing_min_ms <= elapsed_ms <= timing_max_ms:
        return min(current_repeats, max_repeats)
    if elapsed_ms <= 0:
        return min(max_repeats, max(current_repeats + 1, current_repeats * 16))
    target_ms = math.sqrt(timing_min_ms * timing_max_ms)
    estimate = max(1, math.ceil(current_repeats * target_ms / elapsed_ms))
    if elapsed_ms < timing_min_ms:
        estimate = max(current_repeats + 1, estimate)
    elif current_repeats > 1:
        estimate = min(current_repeats - 1, estimate)
    return min(max_repeats, estimate)


def _event_batch_time_ms(torch: Any, launch: Callable[[], Any], repeats: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        launch()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _calibrate_inner_repeats(
    torch: Any,
    launch: Callable[[], Any],
    *,
    timing_min_ms: float,
    timing_max_ms: float,
) -> int:
    repeats = 1
    for _ in range(CALIBRATION_STEPS):
        elapsed_ms = _event_batch_time_ms(torch, launch, repeats)
        adjusted = choose_inner_repeats(
            current_repeats=repeats,
            elapsed_ms=elapsed_ms,
            timing_min_ms=timing_min_ms,
            timing_max_ms=timing_max_ms,
        )
        if adjusted == repeats:
            break
        repeats = adjusted
    return repeats


def timing_stats(samples: Sequence[float], inner_repeats: int) -> TimingStats:
    if not samples:
        raise ValueError("no timing samples were collected")
    return TimingStats(
        samples=len(samples),
        median_us=statistics.median(samples),
        p20_us=percentile(samples, 0.2),
        p80_us=percentile(samples, 0.8),
        mean_us=statistics.fmean(samples),
        stdev_us=statistics.pstdev(samples),
        inner_repeats=inner_repeats,
    )


def measure_launcher(
    torch: Any,
    launch: Callable[[], Any],
    *,
    warmup: int,
    iters: int,
    timing_min_ms: float,
    timing_max_ms: float,
) -> TimingStats:
    if warmup < 0 or iters <= 0:
        raise ValueError("warmup must be non-negative and iters must be positive")
    for _ in range(warmup):
        launch()
    if warmup:
        torch.cuda.synchronize()
    repeats = _calibrate_inner_repeats(
        torch,
        launch,
        timing_min_ms=timing_min_ms,
        timing_max_ms=timing_max_ms,
    )
    samples = [
        _event_batch_time_ms(torch, launch, repeats) * 1000.0 / repeats
        for _ in range(iters)
    ]
    return timing_stats(samples, repeats)


_T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def log_speedup_summary(values: Sequence[float]) -> dict[str, float | int]:
    ratios = [float(value) for value in values]
    if not ratios or any(value <= 0 or not math.isfinite(value) for value in ratios):
        raise ValueError("speedup samples must be finite and positive")
    logs = [math.log(value) for value in ratios]
    mean_log = statistics.fmean(logs)
    if len(logs) == 1:
        ci_low_log = ci_high_log = mean_log
    else:
        critical = _T_CRITICAL_95.get(len(logs) - 1, 1.96)
        half_width = critical * statistics.stdev(logs) / math.sqrt(len(logs))
        ci_low_log = mean_log - half_width
        ci_high_log = mean_log + half_width
    return {
        "samples": len(ratios),
        "median": statistics.median(ratios),
        "geomean": math.exp(mean_log),
        "ci_low": math.exp(ci_low_log),
        "ci_high": math.exp(ci_high_log),
    }


def measure_paired_launchers(
    torch: Any,
    baseline: Callable[[], Any],
    candidate: Callable[[], Any],
    *,
    warmup: int,
    iters: int,
    timing_min_ms: float,
    timing_max_ms: float,
    candidate_first: bool,
) -> PairedTimingStats:
    if warmup < 0 or iters <= 1 or iters % 2:
        raise ValueError(
            "paired timing requires non-negative warmup and an even sample count >= 2"
        )
    for index in range(warmup):
        run_candidate_first = (index % 2 == 0) == candidate_first
        first, second = (
            (candidate, baseline) if run_candidate_first else (baseline, candidate)
        )
        first()
        second()
    torch.cuda.synchronize()

    baseline_repeats = _calibrate_inner_repeats(
        torch,
        baseline,
        timing_min_ms=timing_min_ms,
        timing_max_ms=timing_max_ms,
    )
    candidate_repeats = _calibrate_inner_repeats(
        torch,
        candidate,
        timing_min_ms=timing_min_ms,
        timing_max_ms=timing_max_ms,
    )
    repeats = max(baseline_repeats, candidate_repeats)
    baseline_samples: list[float] = []
    candidate_samples: list[float] = []
    speedups: list[float] = []
    for index in range(iters):
        run_candidate_first = (index % 2 == 0) == candidate_first
        if run_candidate_first:
            candidate_us = (
                _event_batch_time_ms(torch, candidate, repeats) * 1000.0 / repeats
            )
            baseline_us = (
                _event_batch_time_ms(torch, baseline, repeats) * 1000.0 / repeats
            )
        else:
            baseline_us = (
                _event_batch_time_ms(torch, baseline, repeats) * 1000.0 / repeats
            )
            candidate_us = (
                _event_batch_time_ms(torch, candidate, repeats) * 1000.0 / repeats
            )
        baseline_samples.append(baseline_us)
        candidate_samples.append(candidate_us)
        speedups.append(baseline_us / candidate_us)

    speedup = log_speedup_summary(speedups)
    return PairedTimingStats(
        baseline=timing_stats(baseline_samples, repeats),
        candidate=timing_stats(candidate_samples, repeats),
        pair_samples=int(speedup["samples"]),
        speedup_samples=tuple(speedups),
        speedup_median=float(speedup["median"]),
        speedup_geomean=float(speedup["geomean"]),
        speedup_ci_low=float(speedup["ci_low"]),
        speedup_ci_high=float(speedup["ci_high"]),
    )


def validate_exact(
    torch: Any, launch: Callable[[], Any], reference: Any
) -> dict[str, Any]:
    got = launch()
    torch.cuda.synchronize()
    shape_ok = tuple(got.shape) == tuple(reference.shape)
    dtype_ok = got.dtype == reference.dtype
    exact = shape_ok and dtype_ok and bool(torch.equal(got, reference))
    mismatch_count = None
    if shape_ok:
        mismatch_count = int((got != reference).to(torch.int64).sum().item())
    return {
        "valid": exact,
        "shape_ok": shape_ok,
        "dtype_ok": dtype_ok,
        "exact": exact,
        "mismatch_count": mismatch_count,
    }


def _is_expected_failure(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    return (
        "outofresources" in name
        or "out of resource" in message
        or "resource" in message
    )


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _validation_row(
    *,
    torch: Any,
    launch: Callable[[], Any],
    reference: Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    result = validate_exact(torch, launch, reference)
    result["validation_wall_seconds"] = time.perf_counter() - started
    return result


def _rank_candidates(
    *,
    rows: Sequence[dict[str, Any]],
    cases: Sequence[ScheduleCase],
    min_geomean_speedup: float = DEFAULT_PROMOTION_MIN_SPEEDUP,
    min_confidence_bound: float = DEFAULT_PROMOTION_MIN_CI,
    min_worst_case_speedup: float = DEFAULT_PROMOTION_MIN_WORST_SPEEDUP,
) -> list[dict[str, Any]]:
    rankings: list[dict[str, Any]] = []
    tunable_cases = [case for case in cases if case.budget > 0]
    for candidate_id in sorted(
        {row["candidate_id"] for row in rows if row["variant"] == "candidate"}
    ):
        candidate_rows = {
            row["case_name"]: row
            for row in rows
            if row["variant"] == "candidate" and row["candidate_id"] == candidate_id
        }
        first_row = next(iter(candidate_rows.values()), {})
        valid_count = sum(bool(row.get("valid")) for row in candidate_rows.values())
        case_speedups: list[float] = []
        paired_speedups: list[float] = []
        raw_samples_complete = True
        for case in tunable_cases:
            candidate_row = candidate_rows.get(case.name)
            if (
                candidate_row
                and candidate_row.get("status") == "ok"
                and candidate_row.get("valid")
                and float(candidate_row.get("paired_speedup_geomean", 0.0)) > 0
            ):
                case_speedups.append(float(candidate_row["paired_speedup_geomean"]))
                raw = candidate_row.get("paired_speedup_samples")
                if not isinstance(raw, (list, tuple)) or not raw:
                    raw_samples_complete = False
                else:
                    try:
                        samples = [float(value) for value in raw]
                    except (TypeError, ValueError):
                        raw_samples_complete = False
                    else:
                        if any(
                            value <= 0 or not math.isfinite(value) for value in samples
                        ):
                            raw_samples_complete = False
                        else:
                            paired_speedups.extend(samples)
            else:
                raw_samples_complete = False
        eligible = (
            bool(tunable_cases)
            and len(case_speedups) == len(tunable_cases)
            and valid_count == len(cases)
            and raw_samples_complete
        )
        point_aggregate = log_speedup_summary(case_speedups) if case_speedups else None
        uncertainty_aggregate = (
            log_speedup_summary(paired_speedups)
            if raw_samples_complete and paired_speedups
            else None
        )
        geomean = (
            float(uncertainty_aggregate["geomean"])
            if uncertainty_aggregate
            else (float(point_aggregate["geomean"]) if point_aggregate else None)
        )
        ci_low = (
            float(uncertainty_aggregate["ci_low"]) if uncertainty_aggregate else None
        )
        ci_high = (
            float(uncertainty_aggregate["ci_high"]) if uncertainty_aggregate else None
        )
        worst = min(case_speedups) if case_speedups else None
        promotable = bool(
            eligible
            and geomean is not None
            and ci_low is not None
            and worst is not None
            and geomean >= min_geomean_speedup
            and ci_low >= min_confidence_bound
            and worst >= min_worst_case_speedup
        )
        ranking = {
            "candidate_id": candidate_id,
            "BLOCK_C": first_row.get("BLOCK_C"),
            "BLOCK_CP": first_row.get("BLOCK_CP"),
            "num_warps": first_row.get("num_warps"),
            "valid_cases": valid_count,
            "total_cases": len(cases),
            "tunable_cases": len(tunable_cases),
            "speedup_samples": len(paired_speedups),
            "case_speedup_samples": len(case_speedups),
            "raw_samples_complete": raw_samples_complete,
            "eligible": eligible,
            "promotable": promotable,
            "geomean_speedup": geomean,
            "speedup_ci_low": ci_low,
            "speedup_ci_high": ci_high,
            "worst_speedup": worst,
        }
        rankings.append(ranking)
    return sorted(
        rankings,
        key=lambda row: (
            not row["eligible"],
            -(row["geomean_speedup"] or float("-inf")),
            row["candidate_id"],
        ),
    )


def run_gpu(args: argparse.Namespace, plan: dict[str, Any]) -> dict[str, Any]:
    total_started = time.perf_counter()
    runtime = _load_runtime(args.source)
    planned_variant = plan["production_source"]["api_variant"]
    if runtime.api_variant != planned_variant:
        raise RuntimeError(
            f"production schedule API changed after planning: "
            f"{planned_variant} -> {runtime.api_variant}"
        )
    loaded_source = inspect_production_source(Path(runtime.schedule_module.__file__))
    planned_sha256 = plan["production_source"]["sha256"]
    if loaded_source["sha256"] != planned_sha256:
        raise RuntimeError(
            "production schedule source changed after planning: "
            f"{planned_sha256} -> {loaded_source['sha256']}"
        )
    torch = runtime.torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.device_count() != 1 and not args.allow_multiple_visible_gpus:
        raise RuntimeError("exactly one CUDA device must be visible")
    torch.cuda.set_device(0)
    capability = tuple(torch.cuda.get_device_capability(0))
    if capability != (8, 9) and not args.allow_non_sm89:
        raise RuntimeError(f"SM89 is required, found compute capability {capability}")

    cases = select_cases(args.case_set, args.case)
    candidates = generate_candidates(args.block_c, args.block_cp, args.num_warps)
    output = args.output.resolve()
    winner_path = output / "dspark_schedule_winner.json"
    winner_path.unlink(missing_ok=True)
    rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        confidence = _make_confidence(torch, case, seed_offset=args.seed)
        cfg = _make_config(runtime, case)
        reference_started = time.perf_counter()
        reference = runtime.schedule_module.ScheduleVerifyLensTopk.torch(
            confidence=confidence, budget=case.budget, cfg=cfg
        )
        torch.cuda.synchronize()
        reference_wall_seconds = time.perf_counter() - reference_started
        candidate_slots = case.batch_size * min(
            case.gamma, case.max_verify_len or case.gamma + 1
        )

        def baseline_launcher(confidence=confidence, cfg=cfg, budget=case.budget):
            return launch_production_schedule(
                schedule_module=runtime.schedule_module,
                confidence=confidence,
                budget=budget,
                cfg=cfg,
            )

        def new_row(
            variant: str, candidate: ScheduleCandidate | None
        ) -> dict[str, Any]:
            return {
                "case_index": case_index,
                "case_name": case.name,
                "batch_size": case.batch_size,
                "gamma": case.gamma,
                "min_verify_len": case.min_verify_len,
                "max_verify_len": case.max_verify_len,
                "budget": case.budget,
                "pattern": case.pattern,
                "candidate_slots": candidate_slots,
                "tunable_case": case.budget > 0 and candidate_slots > 0,
                "variant": variant,
                "candidate_id": (
                    "production-default"
                    if candidate is None
                    else candidate.candidate_id
                ),
                "BLOCK_C": None if candidate is None else candidate.block_c,
                "BLOCK_CP": None if candidate is None else candidate.block_cp,
                "num_warps": None if candidate is None else candidate.num_warps,
                "reference_timed": False,
                "reference_wall_seconds": reference_wall_seconds,
            }

        baseline_row = new_row("production_default", None)
        baseline_validation = _validation_row(
            torch=torch, launch=baseline_launcher, reference=reference
        )
        baseline_row.update(baseline_validation)
        if not baseline_validation["valid"]:
            raise RuntimeError(f"production parity failed for case {case.name}")
        baseline_stats = measure_launcher(
            torch,
            baseline_launcher,
            warmup=args.warmup,
            iters=args.iters,
            timing_min_ms=args.timing_min_ms,
            timing_max_ms=args.timing_max_ms,
        )
        baseline_row.update({"status": "ok", **asdict(baseline_stats)})
        baseline_row["candidate_slots_per_second"] = (
            candidate_slots * 1_000_000.0 / baseline_stats.median_us
            if baseline_stats.median_us > 0
            else None
        )
        rows.append(baseline_row)

        # Validate every candidate before timing so JIT compilation cannot bias
        # the first measured variants. Candidate order is then shuffled with a
        # reproducible per-case seed and each pair alternates AB/BA order.
        validated: list[
            tuple[ScheduleCandidate, Callable[[], Any], dict[str, Any]]
        ] = []
        for candidate in candidates:

            def launcher(
                candidate=candidate,
                confidence=confidence,
                cfg=cfg,
                budget=case.budget,
            ):
                return launch_schedule_candidate(
                    confidence=confidence,
                    budget=budget,
                    cfg=cfg,
                    candidate=candidate,
                    schedule_module=runtime.schedule_module,
                    triton_module=runtime.triton,
                    api_variant=runtime.api_variant,
                )

            row = new_row("candidate", candidate)
            try:
                validation = _validation_row(
                    torch=torch, launch=launcher, reference=reference
                )
                row.update(validation)
                if not validation["valid"]:
                    row["status"] = "invalid"
                    rows.append(row)
                    continue
                validated.append((candidate, launcher, row))
            except Exception as exc:
                row.update(
                    {
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "expected_failure": _is_expected_failure(exc),
                    }
                )
                rows.append(row)

        random.Random(args.seed + case.seed * 1009 + case_index).shuffle(validated)
        for order_index, (candidate, launcher, row) in enumerate(validated):
            if case.budget <= 0 or candidate_slots <= 0:
                row.update(status="ok", measurement_skipped="non_tunable_case")
                rows.append(row)
                continue
            measured_started = time.perf_counter()
            try:
                paired = measure_paired_launchers(
                    torch,
                    baseline_launcher,
                    launcher,
                    warmup=args.warmup,
                    iters=args.iters,
                    timing_min_ms=args.timing_min_ms,
                    timing_max_ms=args.timing_max_ms,
                    candidate_first=order_index % 2 == 0,
                )
                row.update(
                    {
                        "status": "ok",
                        **asdict(paired.candidate),
                        "baseline_paired_median_us": paired.baseline.median_us,
                        "paired_samples": paired.pair_samples,
                        "paired_speedup_samples": list(paired.speedup_samples),
                        "paired_speedup_median": paired.speedup_median,
                        "paired_speedup_geomean": paired.speedup_geomean,
                        "paired_speedup_ci_low": paired.speedup_ci_low,
                        "paired_speedup_ci_high": paired.speedup_ci_high,
                        "measurement_order": (
                            "candidate-first"
                            if order_index % 2 == 0
                            else "baseline-first"
                        ),
                    }
                )
                row["candidate_slots_per_second"] = (
                    candidate_slots * 1_000_000.0 / paired.candidate.median_us
                    if paired.candidate.median_us > 0
                    else None
                )
                row["measurement_wall_seconds"] = time.perf_counter() - measured_started
            except Exception as exc:
                row.update(
                    {
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "expected_failure": _is_expected_failure(exc),
                    }
                )
            rows.append(row)

    _write_csv(output / "dspark_schedule_measurements.csv", rows)
    rankings = _rank_candidates(
        rows=rows,
        cases=cases,
        min_geomean_speedup=args.promotion_min_speedup,
        min_confidence_bound=args.promotion_min_ci,
        min_worst_case_speedup=args.promotion_min_worst_speedup,
    )
    _write_csv(output / "dspark_schedule_ranking.csv", rankings)
    screening_leader = next((row for row in rankings if row["eligible"]), None)
    winner = next((row for row in rankings if row["promotable"]), None)
    tunable_case_count = sum(case.budget > 0 for case in cases)
    if tunable_case_count == 0:
        decision_status = "no_tunable_cases"
    elif screening_leader is None:
        decision_status = "invalid"
    elif winner is None:
        decision_status = "no_improvement"
    else:
        decision_status = "winner"
    baseline_rows = [row for row in rows if row["variant"] == "production_default"]
    device = {
        "name": torch.cuda.get_device_name(0),
        "capability": list(capability),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    decision = {
        "schema_version": 1,
        "status": decision_status,
        "candidate": winner,
        "screening_leader": screening_leader,
        "production_source": plan["production_source"],
        "device": device,
        "case_set": args.case_set,
        "seed": args.seed,
        "promotion_policy": plan["promotion_policy"],
        "independent_confirmation_required_before_install": winner is not None,
    }
    summary = {
        **plan,
        "completed_at_epoch": time.time(),
        "device": device,
        "wall_seconds": time.perf_counter() - total_started,
        "baseline": {
            "valid_cases": sum(bool(row.get("valid")) for row in baseline_rows),
            "total_cases": len(cases),
        },
        "winner": winner,
        "decision": decision,
        "artifacts": {
            "measurements_csv": str(output / "dspark_schedule_measurements.csv"),
            "ranking_csv": str(output / "dspark_schedule_ranking.csv"),
            "summary_json": str(output / "dspark_schedule_summary.json"),
            "decision_json": str(output / "dspark_schedule_decision.json"),
            "winner_json": str(winner_path) if winner is not None else None,
        },
    }
    _atomic_json(output / "dspark_schedule_decision.json", decision)
    _atomic_json(output / "dspark_schedule_summary.json", summary)
    if winner is not None:
        _atomic_json(
            winner_path,
            {
                "schema_version": 1,
                "status": "winner",
                "candidate": winner,
                "production_source": plan["production_source"],
                "device": device,
                "case_set": args.case_set,
                "seed": args.seed,
                "promotion_policy": plan["promotion_policy"],
                "independent_confirmation_required_before_install": True,
            },
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independent SM89 DSpark production-Triton schedule tuner."
    )
    parser.add_argument("--source", type=Path, required=True, help="SGLang source root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--case-set", choices=("smoke", "representative"), default="representative"
    )
    parser.add_argument(
        "--case", action="append", default=[], help="Case name filter; repeatable"
    )
    parser.add_argument("--block-c", type=int, nargs="+", default=list(DEFAULT_BLOCK_C))
    parser.add_argument(
        "--block-cp", type=int, nargs="+", default=list(DEFAULT_BLOCK_CP)
    )
    parser.add_argument(
        "--num-warps", type=int, nargs="+", default=list(DEFAULT_NUM_WARPS)
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--timing-min-ms", type=float, default=5.0)
    parser.add_argument("--timing-max-ms", type=float, default=20.0)
    parser.add_argument(
        "--promotion-min-speedup",
        type=float,
        default=DEFAULT_PROMOTION_MIN_SPEEDUP,
    )
    parser.add_argument(
        "--promotion-min-ci", type=float, default=DEFAULT_PROMOTION_MIN_CI
    )
    parser.add_argument(
        "--promotion-min-worst-speedup",
        type=float,
        default=DEFAULT_PROMOTION_MIN_WORST_SPEEDUP,
    )
    parser.add_argument("--device-name", default="NVIDIA GeForce RTX 4090")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--allow-non-sm89", action="store_true")
    parser.add_argument("--allow-multiple-visible-gpus", action="store_true")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iters <= 1 or args.iters % 2:
        parser.error("--iters must be an even number of at least 2 for paired timing")
    if args.timing_min_ms <= 0 or args.timing_max_ms <= args.timing_min_ms:
        parser.error("timing window must satisfy 0 < min < max")
    if (
        args.promotion_min_speedup <= 0
        or args.promotion_min_ci <= 0
        or args.promotion_min_worst_speedup <= 0
    ):
        parser.error("promotion thresholds must be positive")
    try:
        generate_candidates(args.block_c, args.block_cp, args.num_warps)
        select_cases(args.case_set, args.case)
    except ValueError as exc:
        parser.error(str(exc))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    plan = build_plan(args)
    if args.plan_only:
        _atomic_json(args.output / "dspark_schedule_plan.json", plan)
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    summary = run_gpu(args, plan)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 2 if summary["decision"]["status"] == "invalid" else 0


if __name__ == "__main__":
    raise SystemExit(main())
