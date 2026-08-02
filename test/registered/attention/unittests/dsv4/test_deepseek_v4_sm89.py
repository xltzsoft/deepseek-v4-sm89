from __future__ import annotations

from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


def test_sm89_kernel_imports():
    from sglang.srt.layers.attention.dsv4.triton_paged_mqa_logits import (
        fp8_paged_mqa_logits_triton_sm89,
    )

    assert callable(fp8_paged_mqa_logits_triton_sm89)


def test_sm89_prefill_avoids_hopper_aot_sparse_kernel():
    from sglang.srt.layers.attention import deepseek_v4_backend as backend
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    with patch.object(backend, "_use_flashmla_fallback_path", True):
        assert not backend._should_use_flashmla_sparse_prefill(
            ForwardMode.EXTEND, num_queries=1024
        )

    with patch.object(backend, "_use_flashmla_fallback_path", False):
        assert backend._should_use_flashmla_sparse_prefill(
            ForwardMode.EXTEND, num_queries=1024
        )


def test_sm89_indexer_kernel_matches_torch_reference():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        return

    from sglang.srt.layers.attention.dsv4.indexer import (
        fp8_paged_mqa_logits_torch_sm120,
    )
    from sglang.srt.layers.attention.dsv4.triton_paged_mqa_logits import (
        fp8_paged_mqa_logits_triton_sm89,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    batch_size = 2
    num_pages = 6
    block_size = 64
    num_heads = 64
    head_dim = 128
    max_seq_len = 150

    q = (torch.randn(batch_size, 1, num_heads, head_dim, device=device) * 0.25).to(
        torch.float8_e4m3fn
    )
    weight = torch.rand(batch_size, num_heads, dtype=torch.bfloat16, device=device)
    weight /= weight.sum(dim=-1, keepdim=True)

    cache = torch.empty(
        num_pages,
        block_size,
        1,
        head_dim + 4,
        dtype=torch.uint8,
        device=device,
    )
    cache_flat = cache.view(num_pages, -1)
    value_bytes = block_size * head_dim
    values = (torch.randn(num_pages, block_size, head_dim, device=device) * 0.25).to(
        torch.float8_e4m3fn
    )
    scales = torch.rand(num_pages, block_size, dtype=torch.float32, device=device) + 0.5
    cache_flat[:, :value_bytes].copy_(values.view(torch.uint8).view(num_pages, -1))
    cache_flat[:, value_bytes:].copy_(scales.view(torch.uint8).view(num_pages, -1))

    seq_lens = torch.tensor([139, 75], dtype=torch.int32, device=device)
    page_table = torch.tensor(
        [[3, 1, 4], [0, 2, 5]], dtype=torch.int32, device=device
    )
    args = (q, cache, weight, seq_lens, page_table, None, max_seq_len, False)

    expected = fp8_paged_mqa_logits_torch_sm120(*args)
    actual = fp8_paged_mqa_logits_triton_sm89(*args)

    valid = torch.arange(max_seq_len, device=device)[None, :] < seq_lens[:, None]
    torch.testing.assert_close(actual[valid], expected[valid], rtol=2e-3, atol=2e-3)
    assert torch.isneginf(actual[~valid]).all()


def test_common_sm89_helper_is_strict():
    from sglang.srt.utils.common import is_cuda, is_sm89_supported

    is_sm89_supported.cache_clear()
    is_cuda.cache_clear()
    with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
        torch.cuda, "get_device_capability", return_value=(8, 9)
    ):
        assert is_sm89_supported()

    is_sm89_supported.cache_clear()
    is_cuda.cache_clear()
    with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
        torch.cuda, "get_device_capability", return_value=(8, 6)
    ):
        assert not is_sm89_supported()

    is_sm89_supported.cache_clear()
    is_cuda.cache_clear()
    with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
        torch.cuda, "get_device_capability", return_value=(9, 0)
    ):
        assert not is_sm89_supported()

    is_sm89_supported.cache_clear()
    is_cuda.cache_clear()
    with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
        torch.cuda, "get_device_capability", return_value=(12, 0)
    ):
        assert not is_sm89_supported()

    is_sm89_supported.cache_clear()
    is_cuda.cache_clear()
    with patch.object(torch.cuda, "is_available", return_value=False):
        assert not is_sm89_supported()


def test_sm89_deepseek_v4_defaults_disable_hopper_only_fast_paths():
    from sglang.srt.environ import envs
    from sglang.srt.server_args import apply_deepseek_v4_sm89_defaults

    fields = [
        envs.SGLANG_OPT_FP8_WO_A_GEMM,
        envs.SGLANG_OPT_USE_TOPK_V2,
        envs.SGLANG_OPT_USE_TILELANG_MHC_PRE,
        envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM,
        envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH,
        envs.SGLANG_SM120_FLASHMLA_BACKEND,
    ]
    for field in fields:
        field.clear()

    try:
        apply_deepseek_v4_sm89_defaults()

        assert not envs.SGLANG_OPT_FP8_WO_A_GEMM.get()
        assert not envs.SGLANG_OPT_USE_TOPK_V2.get()
        assert not envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.get()
        assert not envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get()
        assert not envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get()
        assert envs.SGLANG_SM120_FLASHMLA_BACKEND.get() == "triton"
    finally:
        for field in fields:
            field.clear()


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
