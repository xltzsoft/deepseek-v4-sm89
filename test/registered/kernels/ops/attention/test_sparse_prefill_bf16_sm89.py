import math

import pytest
import torch


def _is_sm89() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 9)


def _reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    kv = kv.squeeze(1).float()
    indices = indices.squeeze(1)
    num_queries, topk = indices.shape
    invalid = (indices < 0) | (indices >= kv.shape[0])
    invalid |= torch.arange(topk, device=q.device)[None, :] >= topk_length[:, None]
    selected = kv.index_select(0, indices.masked_fill(invalid, 0).flatten()).view(
        num_queries, topk, kv.shape[-1]
    )
    scores = torch.einsum("thd,tkd->thk", q.float(), selected) * softmax_scale
    scores.masked_fill_(invalid[:, None, :], float("-inf"))
    denominator_lse = torch.logaddexp(
        torch.logsumexp(scores, dim=-1), attn_sink[None, :]
    )
    probabilities = torch.exp(scores - denominator_lse[:, :, None])
    probabilities.masked_fill_(invalid[:, None, :], 0.0)
    return torch.einsum("thk,tkd->thd", probabilities, selected).to(torch.bfloat16)


@pytest.mark.skipif(not _is_sm89(), reason="SM89 CUDA GPU required")
@pytest.mark.parametrize("topk", [128, 256])
def test_sparse_prefill_bf16_sm89_matches_reference(topk: int):
    from sglang.kernels.ops.attention.dsv4.sparse_prefill_bf16_sm89 import (
        sparse_prefill_bf16_sm89,
    )

    torch.manual_seed(17)
    device = torch.device("cuda")
    num_queries, num_heads, num_kv, head_dim = 5, 64, 311, 512
    scale = head_dim**-0.5
    q = (torch.randn(num_queries, num_heads, head_dim, device=device) * 0.25).to(
        torch.bfloat16
    )
    kv = (torch.randn(num_kv, 1, head_dim, device=device) * 0.25).to(torch.bfloat16)
    indices = torch.randint(
        0, num_kv, (num_queries, 1, topk), dtype=torch.int32, device=device
    )
    topk_length = torch.tensor(
        [0, 1, topk // 3, topk - 1, topk], dtype=torch.int32, device=device
    )
    indices[2, 0, 1] = -1
    indices[3, 0, 2] = num_kv + 7
    attn_sink = torch.linspace(-2.0, 1.0, num_heads, device=device)

    actual = sparse_prefill_bf16_sm89(q, kv, indices, topk_length, attn_sink, scale)
    expected = _reference(q, kv, indices, topk_length, attn_sink, scale)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    assert torch.count_nonzero(actual[0]) == 0


@pytest.mark.skipif(not _is_sm89(), reason="SM89 CUDA GPU required")
def test_sparse_prefill_bf16_sm89_real_dsv4_geometry():
    from sglang.kernels.ops.attention.dsv4.sparse_prefill_bf16_sm89 import (
        sparse_prefill_bf16_sm89,
    )

    torch.manual_seed(23)
    device = torch.device("cuda")
    num_queries, num_heads, num_kv, topk = 32, 64, 2048, 640
    q = torch.randn(num_queries, num_heads, 512, device=device, dtype=torch.bfloat16)
    kv = torch.randn(num_kv, 1, 512, device=device, dtype=torch.bfloat16)
    indices = torch.randint(
        0, num_kv, (num_queries, 1, topk), dtype=torch.int32, device=device
    )
    topk_length = torch.randint(128, topk + 1, (num_queries,), device=device).to(
        torch.int32
    )
    positions = torch.arange(topk, device=device)[None, :]
    indices[:, 0].masked_fill_(positions >= topk_length[:, None], -1)
    attn_sink = torch.randn(num_heads, device=device, dtype=torch.float32)

    output = sparse_prefill_bf16_sm89(
        q,
        kv,
        indices,
        topk_length,
        attn_sink,
        1.0 / math.sqrt(512),
    )
    assert output.shape == q.shape
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
