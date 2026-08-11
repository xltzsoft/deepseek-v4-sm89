"""BF16 sparse MLA prefill for DeepSeek V4 on SM89.

The production DSV4 sparse-prefill path dequantizes its selected KV entries
into a flat BF16 workspace. FlashMLA's AOT sparse-prefill kernel cannot run on
Ada, so this kernel consumes that workspace directly with BF16 Tensor Cores.

The online-softmax body is adapted from the Triton unified-KV prefill kernel,
but this entry point keeps FlashMLA's dense per-query index rows. This avoids
building a compact CSR representation for every prefill chunk.
"""

import torch
import triton
import triton.language as tl


_DSV4_HEAD_DIM = 512
# SM89 launch meta selected by direct-kernel parity sweeps on RTX 4090.
_SM89_BLOCK_H = 16
_SM89_BLOCK_K = 64
_SM89_NUM_WARPS = 8
_SM89_NUM_STAGES = 2


@triton.jit
def _sparse_prefill_bf16_sm89_kernel(
    q_ptr,
    kv_ptr,
    indices_ptr,
    topk_length_ptr,
    attn_sink_ptr,
    out_ptr,
    num_kv: tl.int32,
    topk: tl.int32,
    softmax_scale: tl.constexpr,
    q_stride_t: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    kv_stride_t: tl.constexpr,
    kv_stride_d: tl.constexpr,
    indices_stride_t: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_d: tl.constexpr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_block_idx = tl.program_id(1)

    head_offsets = head_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    dim_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < H

    q = tl.load(
        q_ptr
        + token_idx * q_stride_t
        + head_offsets[:, None] * q_stride_h
        + dim_offsets[None, :] * q_stride_d,
        mask=head_mask[:, None],
        other=0.0,
    )

    neg_large = -3.4028234663852886e38
    max_score = tl.full((BLOCK_H,), neg_large, tl.float32)
    softmax_sum = tl.zeros((BLOCK_H,), tl.float32)
    accumulator = tl.zeros((BLOCK_H, BLOCK_D), tl.float32)

    valid_topk = tl.load(topk_length_ptr + token_idx).to(tl.int32)
    valid_topk = tl.maximum(0, tl.minimum(valid_topk, topk))
    key_offsets = tl.arange(0, BLOCK_K)

    for key_start in tl.range(0, valid_topk, BLOCK_K):
        positions = key_start + key_offsets
        in_length = positions < valid_topk
        kv_indices = tl.load(
            indices_ptr + token_idx * indices_stride_t + positions,
            mask=in_length,
            other=-1,
        )
        valid = in_length & (kv_indices >= 0) & (kv_indices < num_kv)
        safe_indices = tl.where(valid, kv_indices, 0)

        kv = tl.load(
            kv_ptr
            + safe_indices[:, None] * kv_stride_t
            + dim_offsets[None, :] * kv_stride_d,
            mask=valid[:, None],
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(kv)) * softmax_scale
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)

        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(max_score, block_max)
        alpha = tl.exp(max_score - new_max)
        probabilities = tl.exp(scores - new_max[:, None])
        probabilities = tl.where(
            head_mask[:, None] & valid[None, :], probabilities, 0.0
        )

        softmax_sum = softmax_sum * alpha + tl.sum(probabilities, axis=1)
        accumulator = accumulator * alpha[:, None] + tl.dot(
            probabilities.to(kv.dtype), kv
        )
        max_score = new_max

    # The sink is a virtual key whose value is zero. It contributes only to
    # the softmax denominator, but can also become the new normalization max.
    sink = tl.load(attn_sink_ptr + head_offsets, mask=head_mask, other=neg_large).to(
        tl.float32
    )
    final_max = tl.maximum(max_score, sink)
    alpha = tl.exp(max_score - final_max)
    final_sum = softmax_sum * alpha + tl.exp(sink - final_max)
    output = (accumulator * alpha[:, None]) / tl.maximum(final_sum[:, None], 1.0e-30)

    tl.store(
        out_ptr
        + token_idx * out_stride_t
        + head_offsets[:, None] * out_stride_h
        + dim_offsets[None, :] * out_stride_d,
        output,
        mask=head_mask[:, None],
    )


def sparse_prefill_bf16_sm89(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Run DSV4 sparse prefill over a dequantized BF16 workspace.

    Args:
        q: ``[num_queries, num_heads, 512]`` BF16.
        kv: ``[num_kv, 1, 512]`` or ``[num_kv, 512]`` BF16.
        indices: ``[num_queries, 1, topk]`` or ``[num_queries, topk]`` int32.
        topk_length: ``[num_queries]`` int32 valid-prefix lengths.
        attn_sink: ``[num_heads]`` float32 virtual-key scores.
        softmax_scale: QK scale.

    Returns:
        ``[num_queries, num_heads, 512]`` BF16 attention output.
    """
    if not q.is_cuda:
        raise RuntimeError("sparse_prefill_bf16_sm89 requires CUDA tensors")
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise ValueError(f"q and kv must be bfloat16, got {q.dtype=} and {kv.dtype=}")
    if q.ndim != 3 or q.shape[-1] != _DSV4_HEAD_DIM:
        raise ValueError(f"q must have shape [T, H, 512], got {tuple(q.shape)}")

    if kv.ndim == 3:
        if kv.shape[1] != 1:
            raise ValueError(f"kv must have one KV head, got {tuple(kv.shape)}")
        kv = kv.squeeze(1)
    if kv.ndim != 2 or kv.shape[-1] != _DSV4_HEAD_DIM:
        raise ValueError(
            f"kv must have shape [S, 1, 512] or [S, 512], got {tuple(kv.shape)}"
        )

    if indices.ndim == 3:
        if indices.shape[1] != 1:
            raise ValueError(
                f"indices must have one KV head, got {tuple(indices.shape)}"
            )
        indices = indices.squeeze(1)
    if indices.ndim != 2 or indices.shape[0] != q.shape[0]:
        raise ValueError(
            f"indices must have shape [T, 1, K] or [T, K], got {tuple(indices.shape)}"
        )
    if indices.dtype != torch.int32:
        raise ValueError(f"indices must be int32, got {indices.dtype}")
    if topk_length.shape != (q.shape[0],) or topk_length.dtype != torch.int32:
        raise ValueError(
            "topk_length must be int32 with shape "
            f"({q.shape[0]},), got {tuple(topk_length.shape)} {topk_length.dtype}"
        )
    if attn_sink.shape != (q.shape[1],) or attn_sink.dtype != torch.float32:
        raise ValueError(
            f"attn_sink must be float32 with shape ({q.shape[1]},), got "
            f"{tuple(attn_sink.shape)} {attn_sink.dtype}"
        )
    if not all(
        tensor.device == q.device for tensor in (kv, indices, topk_length, attn_sink)
    ):
        raise ValueError(
            "q, kv, indices, topk_length, and attn_sink must share a device"
        )

    q = q.contiguous()
    kv = kv.contiguous()
    indices = indices.contiguous()
    topk_length = topk_length.contiguous()
    attn_sink = attn_sink.contiguous()
    out = torch.empty_like(q)

    block_h = _SM89_BLOCK_H
    block_d = _DSV4_HEAD_DIM
    block_k = _SM89_BLOCK_K
    grid = (q.shape[0], triton.cdiv(q.shape[1], block_h))
    _sparse_prefill_bf16_sm89_kernel[grid](
        q,
        kv,
        indices,
        topk_length,
        attn_sink,
        out,
        kv.shape[0],
        indices.shape[1],
        float(softmax_scale),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        indices.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        q.shape[1],
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        num_warps=_SM89_NUM_WARPS,
        num_stages=_SM89_NUM_STAGES,
    )
    return out
