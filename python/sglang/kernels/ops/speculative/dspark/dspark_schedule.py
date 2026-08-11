from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.speculative.dspark.dispatch import (
    inputs_on_cuda,
)

if TYPE_CHECKING:
    from sglang.srt.speculative.dspark_components.dspark_planner import (
        DSparkScheduleConfig,
    )


class ScheduleVerifyLensTopk:
    @classmethod
    def execute(cls, *args, **kwargs) -> torch.Tensor:
        if inputs_on_cuda(*args, **kwargs):
            return cls.triton(*args, **kwargs)
        return cls.torch(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        confidence: torch.Tensor,
        budget: int,
        cfg: DSparkScheduleConfig,
    ) -> torch.Tensor:
        return schedule_verify_lens_topk(confidence=confidence, budget=budget, cfg=cfg)

    @classmethod
    def triton(
        cls,
        *,
        confidence: torch.Tensor,
        budget: int,
        cfg: DSparkScheduleConfig,
    ) -> torch.Tensor:
        return schedule_verify_lens_topk_triton(
            confidence=confidence, budget=budget, cfg=cfg
        )


def compute_sort_survival(confidence: torch.Tensor) -> torch.Tensor:
    return torch.cumprod(confidence.to(torch.float32), dim=1)


def schedule_verify_lens_topk(
    *,
    confidence: torch.Tensor,
    budget: int,
    cfg: DSparkScheduleConfig,
) -> torch.Tensor:
    return schedule_verify_lens_topk_from_survival(
        survival_probs=compute_sort_survival(confidence), budget=budget, cfg=cfg
    )


def schedule_verify_lens_topk_from_survival(
    *,
    survival_probs: torch.Tensor,
    budget: int,
    cfg: DSparkScheduleConfig,
) -> torch.Tensor:
    num_requests, _gamma = survival_probs.shape
    max_len = cfg.resolved_max_verify_len()
    device = survival_probs.device

    selected_extra = torch.zeros(num_requests, dtype=torch.int64, device=device)
    if budget > 0:
        candidate_window = survival_probs[:, :max_len]
        num_candidates = candidate_window.numel()
        if num_candidates > 0:
            request_index = (
                torch.arange(num_requests, device=device)
                .view(num_requests, 1)
                .expand_as(candidate_window)
            )
            position_index = (
                torch.arange(candidate_window.shape[1], device=device)
                .view(1, candidate_window.shape[1])
                .expand_as(candidate_window)
            )
            valid = candidate_window >= cfg.survival_eps

            # Once the budget covers every effective candidate, ranking cannot
            # change the result.  Keep this branch in the torch reference too:
            # it is the executable semantic model for the corresponding Triton
            # fast path, including the survival-eps boundary.
            if _budget_covers_all_valid_candidates(valid=valid, budget=budget):
                selected_extra = valid.to(torch.int64).sum(dim=1)
            else:
                flat_prob = candidate_window.reshape(-1).to(torch.float64)
                flat_request = request_index.reshape(-1)
                flat_position = position_index.reshape(-1)
                flat_valid = valid.reshape(-1)

                order = _value_independent_descending_order(
                    probs=flat_prob,
                    positions=flat_position,
                    requests=flat_request,
                    valid=flat_valid,
                )

                take = min(int(budget), num_candidates)
                chosen = order[:take]
                chosen_requests = flat_request[chosen]
                chosen_valid = flat_valid[chosen].to(torch.int64)
                selected_extra.scatter_add_(0, chosen_requests, chosen_valid)

    min_len = torch.full(
        (num_requests,), cfg.min_verify_len, dtype=torch.int64, device=device
    )
    verify_lens = min_len + selected_extra
    lower_bound = max(cfg.min_verify_len, 1)
    verify_lens = torch.clamp(verify_lens, min=lower_bound, max=max_len)
    return verify_lens.to(torch.int32)


def _budget_covers_all_valid_candidates(*, valid: torch.Tensor, budget: int) -> bool:
    """Return whether no candidate can be dropped by the global budget.

    This helper is intentionally scalar and lives in the reference path only.
    The CUDA wrapper avoids a host synchronization and therefore takes its
    no-ranking fast path only when the budget covers every candidate slot.
    ``valid`` is expected to be the mask after applying ``survival_eps`` and the
    configured candidate window.
    """
    return int(budget) >= int(valid.to(torch.int32).sum().item())


def _value_independent_descending_order(
    *,
    probs: torch.Tensor,
    positions: torch.Tensor,
    requests: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    masked_prob = torch.where(valid, probs, torch.full_like(probs, float("-inf")))
    num_candidates = masked_prob.numel()
    order = torch.arange(num_candidates, device=probs.device)
    order = order[torch.argsort(requests[order], stable=True)]
    order = order[torch.argsort(positions[order], stable=True)]
    order = order[torch.argsort(-masked_prob[order], stable=True)]
    return order


@triton.jit
def _schedule_topk_prep_kernel(
    confidence_ptr,
    survival_ptr,
    valid_counts_ptr,
    gamma,
    cols,
    survival_eps,
    G_P2: tl.constexpr,
):
    row = tl.program_id(0)
    g = tl.arange(0, G_P2)
    conf = tl.load(
        confidence_ptr + row.to(tl.int64) * gamma + g, mask=g < gamma, other=1.0
    ).to(tl.float32)
    surv = tl.cumprod(conf, axis=0)
    tl.store(survival_ptr + row.to(tl.int64) * cols + g, surv, mask=g < cols)
    valid = (g < cols) & (surv >= survival_eps)
    tl.store(valid_counts_ptr + row, tl.sum(valid.to(tl.int32), axis=0))


@triton.jit
def _schedule_topk_all_valid_finalize_kernel(
    valid_counts_ptr,
    out_ptr,
    min_verify_len,
    lower_bound,
    max_len,
    bs,
    BLOCK: tl.constexpr,
):
    """Finalize the no-ranking case when the budget covers every candidate slot."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < bs
    extra = tl.load(valid_counts_ptr + offs, mask=mask, other=0).to(tl.int32)
    lens = min_verify_len + extra
    lens = tl.maximum(lens, lower_bound)
    lens = tl.minimum(lens, max_len)
    tl.store(out_ptr + offs, lens, mask=mask)


@triton.jit
def _schedule_topk_select_and_finalize_kernel(
    survival_ptr,
    out_ptr,
    budget,
    cols,
    n,
    survival_eps,
    min_verify_len,
    lower_bound,
    max_len,
    BLOCK_C: tl.constexpr,
    BLOCK_CP: tl.constexpr,
):
    """Select the global top budget and finalize one request per program."""
    row = tl.program_id(0)
    p = tl.arange(0, BLOCK_C)
    c = row.to(tl.int64) * cols + p
    cmask = p < cols
    sp = tl.load(survival_ptr + c, mask=cmask, other=0.0)
    valid_c = cmask & (sp >= survival_eps)
    mp = tl.where(valid_c, sp, float("-inf"))
    rank = tl.zeros([BLOCK_C], dtype=tl.int32)
    for cp0 in range(0, n, BLOCK_CP):
        cp = cp0 + tl.arange(0, BLOCK_CP)
        cpmask = cp < n
        rp = cp // cols
        pp = cp % cols
        spp = tl.load(survival_ptr + cp, mask=cpmask, other=0.0)
        validp = cpmask & (spp >= survival_eps)
        mpp = tl.where(validp, spp, float("-inf"))
        gt = mpp[None, :] > mp[:, None]
        eq = mpp[None, :] == mp[:, None]
        pos_lt = pp[None, :] < p[:, None]
        pos_eq = pp[None, :] == p[:, None]
        req_lt = rp[None, :] < row
        before = gt | (eq & (pos_lt | (pos_eq & req_lt)))
        before = before & cpmask[None, :]
        rank += tl.sum(before.to(tl.int32), axis=1)
    extra = tl.sum((valid_c & (rank < budget)).to(tl.int32), axis=0)
    lens = min_verify_len + extra
    lens = tl.maximum(lens, lower_bound)
    lens = tl.minimum(lens, max_len)
    tl.store(out_ptr + row, lens)


def schedule_verify_lens_topk_triton(
    *,
    confidence: torch.Tensor,
    budget: int,
    cfg: DSparkScheduleConfig,
) -> torch.Tensor:
    num_requests, gamma = confidence.shape
    max_len = cfg.resolved_max_verify_len()
    device = confidence.device
    cols = min(max_len, gamma)
    n = num_requests * cols

    lower_bound = max(cfg.min_verify_len, 1)
    if num_requests == 0:
        return torch.empty(0, dtype=torch.int32, device=device)
    if budget <= 0 or n <= 0:
        base_len = min(lower_bound, max_len)
        return torch.full((num_requests,), base_len, dtype=torch.int32, device=device)

    valid_counts = torch.empty(num_requests, dtype=torch.int32, device=device)
    survival = torch.empty((num_requests, cols), dtype=torch.float32, device=device)
    confidence = confidence.contiguous()
    _schedule_topk_prep_kernel[(num_requests,)](
        confidence,
        survival,
        valid_counts,
        gamma,
        cols,
        float(cfg.survival_eps),
        G_P2=triton.next_power_of_2(max(gamma, 1)),
    )

    verify_lens = torch.empty(num_requests, dtype=torch.int32, device=device)
    if budget >= n:
        block = 256
        _schedule_topk_all_valid_finalize_kernel[(triton.cdiv(num_requests, block),)](
            valid_counts,
            verify_lens,
            int(cfg.min_verify_len),
            lower_bound,
            int(max_len),
            num_requests,
            BLOCK=block,
        )
    else:
        _schedule_topk_select_and_finalize_kernel[(num_requests,)](
            survival,
            verify_lens,
            int(budget),
            cols,
            n,
            float(cfg.survival_eps),
            int(cfg.min_verify_len),
            lower_bound,
            int(max_len),
            BLOCK_C=triton.next_power_of_2(max(cols, 1)),
            BLOCK_CP=256,
        )
    return verify_lens
