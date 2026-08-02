# DeepSeek-V4 Flash MXFP4 for SM89

An experimental SGLang fork focused on running the DeepSeek-V4-Flash-0731
MXFP4 checkpoint on a single 8 x NVIDIA RTX 4090 (SM89) server with tensor
parallelism 8.

This repository contains framework source code only. Model weights, compiled
caches, benchmark prompts, credentials, and host-specific configuration are not
included.

> [!WARNING]
> This branch is a performance research snapshot, not a production release.
> The target-only path is operational and benchmarked. DSpark speculative
> decoding did not pass strict token-by-token parity, and repeated target-only
> requests can still diverge when request slots or compression state are reused.
> See [Known limitations](#known-limitations) before using generated output.

## Snapshot

- Framework snapshot: `d8f0ff966`
- Release tag: `v0.1.0-experimental`
- Upstream base: SGLang `131bd51b`
- GPU architecture: NVIDIA Ada, SM89
- Validated topology: 8 x RTX 4090 24 GB, TP=8
- Checkpoint format: MXFP4 routed experts with FP8 KV cache
- Recommended serving mode: target-only, full decode CUDA graph
- License: Apache-2.0, inherited from SGLang

## What changed

The branch adds an SM89-specific DeepSeek-V4 execution path instead of relying
on kernels intended for Hopper or Blackwell:

- Enables MXFP4 Marlin MoE execution on SM89 and selects it automatically.
- Routes sparse MLA decode and prefill through SM89-compatible Triton kernels.
- Adds a dedicated BF16 sparse-prefill kernel for the Ada launch geometry.
- Pins the sparse decode launch configuration to 32-token tiles, 8 warps, and
  2 stages to avoid runtime autotune allocation spikes.
- Keeps the native eight local query heads under TP=8 rather than padding to a
  64-head FlashMLA layout.
- Avoids redundant SM89 indexer KV-pool copies.
- Adds SM89 fallbacks for FP8 indexer and MHC prenorm operations.
- Loads fused shared-expert weights in the DSpark draft model.
- Initializes offline C4/C128 compression-state pools to explicit empty-state
  sentinels.

The complete patch series is available with:

```bash
git log --oneline 131bd51b..d8f0ff966
```

## Measured performance

These results were measured on one 8 x RTX 4090 24 GB machine with TP=8,
context length 8192, maximum running requests 8, full decode CUDA graphs, and
custom all-reduce disabled. They are point measurements from this hardware and
software snapshot, not general SGLang claims.

| Scenario | Metric | Result |
| --- | --- | ---: |
| C1 | Decode throughput | 54.75 tok/s |
| C1 | End-to-end throughput | 50.36 tok/s |
| C1 | Time to first token | 0.222 s |
| C8 | Scheduler output throughput | 382.8-383.8 tok/s |
| C8 | Aggregate decode-window throughput | 374.58 tok/s |
| C8 | Aggregate end-to-end throughput | 318.59 tok/s |
| C8 | Mean GPU utilization | about 92.7% |

The benchmark used 256 input tokens, 128 generated tokens, greedy sampling,
`ignore_eos=true`, and simultaneous C8 request release. Throughput definitions
must be kept separate when comparing these numbers with other engines.

## Environment

Use a CUDA-capable Linux environment compatible with the current SGLang main
branch. The validated deployment used Python 3.10, CUDA 13 user-space
libraries, a matching PyTorch build, Triton, TVM-FFI JIT support, and an
SGLang kernel wheel built for the same CUDA ABI.

Follow the upstream [SGLang installation documentation](https://docs.sglang.ai/)
to create the base environment, then run this checkout through `PYTHONPATH`:

```bash
git clone https://github.com/xltzsoft/deepseek-v4-sm89.git
cd deepseek-v4-sm89

export PYTHONPATH="$PWD/python"
export TORCH_CUDA_ARCH_LIST=8.9
export CUDA_MODULE_LOADING=LAZY
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

The DeepSeek-V4-Flash-0731 MXFP4 weights must be downloaded separately. Set
`MODEL_PATH` to the local ModelScope or Hugging Face snapshot directory.

## Launch

The validated target-only configuration was:

```bash
export MODEL_PATH=/path/to/DeepSeek-V4-Flash-0731

python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port 31000 \
  --tp-size 8 \
  --context-length 8192 \
  --max-running-requests 8 \
  --max-total-tokens 8192 \
  --max-prefill-tokens 8192 \
  --chunked-prefill-size -1 \
  --mem-fraction-static 0.92 \
  --swa-full-tokens-ratio 1.0 \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend dsv4 \
  --moe-runner-backend marlin \
  --cuda-graph-backend-decode full \
  --cuda-graph-backend-prefill disabled \
  --cuda-graph-max-bs-decode 8 \
  --disable-custom-all-reduce
```

Keep the service bound to loopback unless authentication and network controls
have been configured explicitly.

## Validation

The SM89 patch includes focused unit coverage for backend selection, native
head geometry, sparse prefill, DSpark shared-expert loading, launch selection,
and compression-state initialization:

```bash
pytest -q \
  test/registered/attention/unittests/dsv4/test_deepseek_v4_sm89.py \
  test/registered/kernels/ops/attention/test_sparse_prefill_bf16_sm89.py \
  test/registered/unit/kernels/test_flash_mla_sm89_config.py \
  test/registered/unit/mem_cache/test_deepseek_v4_compress_state_init.py \
  test/registered/unit/models/test_deepseek_v4_dspark_weight_loading.py \
  test/registered/unit/models/test_deepseek_v4_sm89_head_padding.py
```

Before trusting a deployment, perform at least two cold starts and replay the
same greedy prompts in forward and reverse order. Compare every output token,
not only decoded text or acceptance rate.

## Known limitations

1. **Compression-state lifecycle is not fully resolved.** Disabling the radix
   cache does not eliminate request-order drift. C128 state follows request-slot
   ownership, while C4 attention and indexer state follow physical SWA pages.
   Precise generation/page cleanup is still required.
2. **DSpark is not accepted for this snapshot.** It can load and start within
   24 GB per GPU after the included fixes, but greedy outputs did not match the
   target reference and the eager-only path was slower than target CUDA graphs.
3. **Weights are not redistributed.** Obtain the checkpoint under its own
   license and usage terms.
4. **The performance envelope is narrow.** Different drivers, CUDA/PyTorch
   builds, clocks, thermals, prompts, or batching policies can materially change
   throughput.

## Upstream and license

This work is derived from [sgl-project/sglang](https://github.com/sgl-project/sglang).
The original copyright notices and Apache-2.0 license are retained. Changes in
this branch are provided under the same license.
