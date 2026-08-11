#!/usr/bin/env python3
"""Parity and CUDA-event timing for the production DSpark schedule kernel.

This entry deliberately keeps ``ScheduleVerifyLensTopk.torch`` outside the
timed region.  The adaptive queue's ``profile_dspark_kernels.py`` invokes
``TestDsparkKernelParity._parity``, which evaluates both ``cls.triton`` and
``cls.torch``.  The latter reaches the stable ``torch.argsort`` reference in
``dspark_schedule.py``; consequently, ``aten::sort`` in that combined trace is
not evidence of a production DSpark hotspot.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Iterable

import torch

from sglang.kernels.ops.speculative.dspark.dspark_schedule import (
    ScheduleVerifyLensTopk,
)
from sglang.srt.speculative.dspark_components.dspark_planner import (
    DSparkScheduleConfig,
)


MODES = ("partial", "all-effective", "all-candidates")


def _csv_ints(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def _csv_modes(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(values) - set(MODES))
    if not values or unknown:
        raise argparse.ArgumentTypeError(
            f"expected a comma-separated subset of {MODES}; unknown={unknown}"
        )
    return values


def _make_case(
    *, batch_size: int, gamma: int, mode: str, cfg: DSparkScheduleConfig
) -> tuple[torch.Tensor, int, int]:
    generator = torch.Generator(device="cuda").manual_seed(
        20260806 + batch_size * 101 + gamma
    )
    confidence = torch.rand(
        (batch_size, gamma), device="cuda", generator=generator, dtype=torch.float32
    )
    confidence = confidence * 0.19 + 0.80
    if mode == "all-effective":
        confidence[:, max(1, gamma // 2) :] = 0.0

    cols = min(cfg.resolved_max_verify_len(), gamma)
    survival = torch.cumprod(confidence, dim=1)[:, :cols]
    effective_candidates = int((survival >= cfg.survival_eps).sum().item())
    if mode == "partial":
        budget = max(effective_candidates // 2, 1)
    elif mode == "all-effective":
        budget = effective_candidates
    elif mode == "all-candidates":
        budget = batch_size * cols
    else:
        raise ValueError(f"unknown mode: {mode}")
    return confidence, budget, effective_candidates


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[min(int((len(ordered) - 1) * fraction), len(ordered) - 1)]


def _time_triton_only(
    *,
    schedule,
    confidence: torch.Tensor,
    budget: int,
    cfg: DSparkScheduleConfig,
    warmup: int,
    iters: int,
) -> list[float]:
    for _ in range(warmup):
        schedule.triton(confidence=confidence, budget=budget, cfg=cfg)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends):
        start.record()
        schedule.triton(confidence=confidence, budget=budget, cfg=cfg)
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]


def _time_one_triton_launch(*, schedule, confidence, budget, cfg) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    schedule.triton(confidence=confidence, budget=budget, cfg=cfg)
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) * 1000.0)


def _time_paired_triton_only(
    *,
    schedule,
    legacy_schedule,
    confidence: torch.Tensor,
    budget: int,
    cfg: DSparkScheduleConfig,
    warmup: int,
    iters: int,
) -> tuple[list[float], list[float], list[float]]:
    if iters <= 1 or iters % 2:
        raise ValueError("paired timing requires an even sample count >= 2")
    for index in range(warmup):
        first, second = (
            (schedule, legacy_schedule)
            if index % 2 == 0
            else (legacy_schedule, schedule)
        )
        first.triton(confidence=confidence, budget=budget, cfg=cfg)
        second.triton(confidence=confidence, budget=budget, cfg=cfg)
    torch.cuda.synchronize()

    schedule_samples: list[float] = []
    legacy_samples: list[float] = []
    speedups: list[float] = []
    for index in range(iters):
        if index % 2 == 0:
            schedule_us = _time_one_triton_launch(
                schedule=schedule, confidence=confidence, budget=budget, cfg=cfg
            )
            legacy_us = _time_one_triton_launch(
                schedule=legacy_schedule,
                confidence=confidence,
                budget=budget,
                cfg=cfg,
            )
        else:
            legacy_us = _time_one_triton_launch(
                schedule=legacy_schedule,
                confidence=confidence,
                budget=budget,
                cfg=cfg,
            )
            schedule_us = _time_one_triton_launch(
                schedule=schedule, confidence=confidence, budget=budget, cfg=cfg
            )
        schedule_samples.append(schedule_us)
        legacy_samples.append(legacy_us)
        speedups.append(legacy_us / schedule_us)
    return schedule_samples, legacy_samples, speedups


def _load_legacy_schedule(path: Path):
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"legacy schedule source not found: {path}")
    module_name = f"_dspark_schedule_legacy_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load legacy schedule source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.ScheduleVerifyLensTopk


def run(
    *,
    batch_sizes: Iterable[int],
    gamma: int,
    modes: Iterable[str],
    warmup: int,
    iters: int,
    legacy_schedule=None,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DSpark Triton parity and timing")

    cfg = DSparkScheduleConfig(gamma=gamma)
    rows = []
    for batch_size in batch_sizes:
        for mode in modes:
            confidence, budget, effective_candidates = _make_case(
                batch_size=batch_size, gamma=gamma, mode=mode, cfg=cfg
            )

            # The reference is a correctness oracle only and is never timed.
            reference = ScheduleVerifyLensTopk.torch(
                confidence=confidence, budget=budget, cfg=cfg
            )
            actual = ScheduleVerifyLensTopk.triton(
                confidence=confidence, budget=budget, cfg=cfg
            )
            torch.cuda.synchronize()
            if not torch.equal(actual, reference):
                raise AssertionError(
                    f"parity failure: bs={batch_size} gamma={gamma} mode={mode} "
                    f"budget={budget} actual={actual.tolist()} "
                    f"reference={reference.tolist()}"
                )

            samples_us = _time_triton_only(
                schedule=ScheduleVerifyLensTopk,
                confidence=confidence,
                budget=budget,
                cfg=cfg,
                warmup=warmup,
                iters=iters,
            )
            row = {
                "batch_size": batch_size,
                "gamma": gamma,
                "mode": mode,
                "budget": budget,
                "effective_candidates": effective_candidates,
                "latency_us_median": statistics.median(samples_us),
                "latency_us_p90": _percentile(samples_us, 0.90),
                "latency_us_min": min(samples_us),
                "parity": "exact",
            }
            if legacy_schedule is not None:
                legacy_actual = legacy_schedule.triton(
                    confidence=confidence, budget=budget, cfg=cfg
                )
                torch.cuda.synchronize()
                if not torch.equal(legacy_actual, reference):
                    raise AssertionError(
                        f"legacy parity failure: bs={batch_size} gamma={gamma} "
                        f"mode={mode} budget={budget}"
                    )
                paired_samples_us, legacy_samples_us, paired_speedups = (
                    _time_paired_triton_only(
                        schedule=ScheduleVerifyLensTopk,
                        legacy_schedule=legacy_schedule,
                        confidence=confidence,
                        budget=budget,
                        cfg=cfg,
                        warmup=warmup,
                        iters=iters,
                    )
                )
                paired_median = statistics.median(paired_samples_us)
                legacy_median = statistics.median(legacy_samples_us)
                row.update(
                    {
                        "latency_us_median": paired_median,
                        "latency_us_p90": _percentile(paired_samples_us, 0.90),
                        "latency_us_min": min(paired_samples_us),
                        "legacy_latency_us_median": legacy_median,
                        "legacy_latency_us_p90": _percentile(legacy_samples_us, 0.90),
                        "legacy_latency_us_min": min(legacy_samples_us),
                        "speedup_vs_legacy": legacy_median / paired_median,
                        "paired_speedup_vs_legacy_geomean": math.exp(
                            statistics.fmean(
                                math.log(value) for value in paired_speedups
                            )
                        ),
                        "legacy_pair_order": "alternating AB/BA",
                        "legacy_parity": "exact",
                    }
                )
            rows.append(row)

    major, minor = torch.cuda.get_device_capability()
    return {
        "device": torch.cuda.get_device_name(),
        "device_capability": f"sm{major}{minor}",
        "torch": torch.__version__,
        "timed_scope": "ScheduleVerifyLensTopk.triton only",
        "reference_in_timed_region": False,
        "legacy_baseline": legacy_schedule is not None,
        "warmup": warmup,
        "iterations": iters,
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=_csv_ints, default=_csv_ints("1,8,32,64"))
    parser.add_argument("--gamma", type=int, default=5)
    parser.add_argument("--modes", type=_csv_modes, default=list(MODES))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--legacy-source", type=Path)
    args = parser.parse_args()
    if args.gamma <= 0 or args.warmup < 0 or args.iters <= 0:
        parser.error("gamma and iters must be positive; warmup must be non-negative")
    if args.iters % 2:
        parser.error("--iters must be even for balanced paired legacy timing")

    legacy_schedule = (
        _load_legacy_schedule(args.legacy_source) if args.legacy_source else None
    )
    result = run(
        batch_sizes=args.batch_sizes,
        gamma=args.gamma,
        modes=args.modes,
        warmup=args.warmup,
        iters=args.iters,
        legacy_schedule=legacy_schedule,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
