# Qwen3.6 NVFP4 Quantization Toolkit

Reproducible NVFP4 quantization + deployment scripts for **Qwen3.6** family on Blackwell-class GPUs (Spark sm_121 / RTX 5090 sm_120).

**Models published**:
- 🤗 [nerkyor/Qwen3.6-35B-A3B-NVFP4-v8-RTN](https://huggingface.co/nerkyor/Qwen3.6-35B-A3B-NVFP4-v8-RTN) — MoE 35B (active 3B), 21 GB
- 🤗 [nerkyor/Qwen3.6-27B-NVFP4-v8-RTN](https://huggingface.co/nerkyor/Qwen3.6-27B-NVFP4-v8-RTN) — Dense 27B, 18 GB

Built by [Lynn AI](https://lynn-ai.app).

## Why this toolkit exists

The community has unverified claims about Qwen3.6 NVFP4 performance (e.g. "RedHatAI 35B-A3B-NVFP4 vLLM cu130-nightly 433 tok/s") with **no reproducible recipe**. This toolkit provides:

- ✅ **Verified working calibration recipe** (llmcompressor RTN, ~5-16 min on 1× RTX 4090)
- ✅ **Verified working deployment** (SGLang `dev-cu13`, with all required flags)
- ✅ **Empirical performance data** on Spark sm_121 (head-to-head NVFP4 vs FP8, single-stream + concurrent)
- ✅ **Comprehensive pitfall documentation** ([PITFALLS.md](docs/PITFALLS.md)) covering inference framework compatibility, architecture mismatch traps, Chinese thinking injection, hybrid attention flags, etc.

## Quick start

### Quantize from BF16 source

```bash
# Install dependencies
pip install -U llmcompressor transformers torch

# Run RTN quantization (5-16 min on 1× RTX 4090)
python calibration/v8-rtn-llmcompressor.py \
  --model-path /path/to/Qwen3.6-35B-A3B \
  --output-dir /path/to/output-NVFP4
```

### Deploy with SGLang

```bash
# Patch ckpt for SGLang multimodal architecture (2 min)
python calibration/repack-multimodal-patch.py \
  --ckpt-dir /path/to/output-NVFP4 \
  --vision-source /path/to/Qwen3.6-35B-A3B-FP8

# Inject Chinese thinking marker (optional)
bash deploy/chat-template-inject-cn.sh /path/to/output-NVFP4

# Launch SGLang
bash deploy/sglang-launch.sh /path/to/output-NVFP4
```

### Benchmark

```bash
# Performance: TTFT + TPS + concurrency
python bench/perf-bench.py --base http://127.0.0.1:18099/v1 --model lynn-nvfp4

# V8 30-question quality eval
python bench/v8-30q-bench.py --base http://127.0.0.1:18099/v1 --model lynn-nvfp4
```

## ⚠️ Critical pitfalls — read first

**TL;DR**: as of 2026-05-05, only **SGLang `dev-cu13` + transformers 5.8.0.dev0 nightly** verified to load Qwen3.6 NVFP4 ckpts on Blackwell. vLLM 0.20.x and TensorRT-LLM 1.2.0 all hit blockers. See [docs/PITFALLS.md](docs/PITFALLS.md) for the full table.

⚠️ **Critical: MTP only on MoE — dense reverses gains** (PITFALLS §14). 35B-A3B with MTP **accelerates** (MoE active-3B + high draft-accept rate). 27B dense with MTP **decelerates 73-80%** + N=16 TTFT explodes 9.7×. Pass `--mtp` to `deploy/sglang-launch.sh` only for MoE.

| Framework | Status | Why |
|---|---|---|
| **SGLang dev-cu13** | ✅ Works | Native MTP/NEXTN support (MoE only), hybrid attention handled |
| vLLM 0.17.1+nvinternal (NGC 26.03) | ❌ | bundled `transformers 4.57.3` doesn't recognize `qwen3_5_moe`; upgrading breaks vLLM internals |
| vLLM 0.20.1.dev0 | ❌ | hybrid attention `unify_kv_cache_spec_page_size NotImplementedError` |
| vLLM 0.20.1 (5090) | ❌ | doesn't support `Qwen3_5ForCausalLM` LinearAttention architecture |
| TensorRT-LLM 1.2.0 (NGC 26.03) | ❌ | `transformers 4.57.3` ↔ `qwen3_5_moe` mismatch + openai SDK incompat |

## Empirical performance (Spark GB10 sm_121, SGLang dev-cu13)

3-run median, all temperature=0, MTP NEXTN steps=3 where applicable.

### 35B-A3B-NVFP4-v8-RTN vs 35B-A3B-FP8 base

| Scenario | NVFP4 v8-RTN (no `首先`) | FP8 (with `首先`) | NVFP4 v8-RTN (with `首先`) |
|---|---|---|---|
| short single TPS | **103.2** | 97.3 | 39.3 |
| medium single TPS | 78.7 | **101.6** | 39.4 |
| long single TPS | 74.0 | **88.6** | 39.0 |
| N=4 agg | **309.7** | 278.1 | 137.2 |
| N=8 agg | **499.0** | 417.4 | 281.0 |
| N=16 agg | **756.0** | 615.8 | 322.9 |

### Lynn V4-Pro Distill on v8-RTN (2026-05-14, no MTP) — distinct from base above

The above numbers are for **base Qwen3.6-35B-A3B** with NEXTN MTP enabled (since the base ships with trained MTP head weights). **Lynn V4-Pro Distill was distilled without MTP** and is served on a different config:

| Scenario | Lynn V4-Pro NVFP4 v8-RTN (no MTP) | Lynn V4-Pro + MTP NEXTN | Δ |
|---|---|---|---|
| single TPS (avg of 5 prompts) | **58.7 tok/s** | 28.4 tok/s ⚠️ | **-51.6%** |
| TTFT (single stream) | **81 ms** | 175 ms | +94 ms |
| N=4 aggregate | **219.5** | 80.6 ⚠️ | -63.0% |
| N=8 aggregate | **387.2** | 156.1 ⚠️ | -59.7% |
| N=16 aggregate | **599.4** | 234.9 ⚠️ | -60.7% |
| Long ctx 32K input | **48.4 tok/s ✓** | 25.8 tok/s | -46.7% |

Root cause: Lynn V4-Pro Distill training did **not** produce MTP head weights. SGLang's NEXTN draft model gives predictions that get rejected by the verifier → wasted compute. **For Lynn V4-Pro production, do not enable MTP.**

This is specific to the Distill checkpoint — the base Qwen3.6 numbers above (NEXTN +23%) still apply if you quantize the base yourself with this toolkit.

### Q4_K_M reference perf (Lynn V4-Pro, llama.cpp sm_121 native build)

For comparison with the consumer-tier sibling (Q4_K_M GGUF, llama.cpp, **not** in this NVFP4 toolkit's scope but useful context):

| N | NVFP4 v8-RTN @ SGLang | Q4_K_M @ llama.cpp `--parallel 16` |
|---|---|---|
| single | 58.7 tok/s | **74.9 tok/s** (+27%) |
| N=4 agg | **219.5** | 88.9 ⚠️ regressed |
| N=16 agg | **599.4** | 252.0 |

`llama.cpp --parallel` is **slot multiplexing, not true continuous batching** — small concurrent batches regress (N=4 agg < N=2 agg). For multi-user serving, prefer NVFP4 + SGLang. For consumer single-user, Q4_K_M is faster single-stream.

### 27B (dense) vs 35B-A3B (MoE) — both NVFP4 v8-RTN, no injection

| Scenario | 27B dense | 35B-A3B MoE | MoE faster by |
|---|---|---|---|
| short single TPS | 47.4 | 103.2 | **2.2×** |
| medium single TPS | 36.3 | 78.7 | 2.2× |
| long single TPS | 40.9 | 74.0 | 1.8× |
| N=16 agg | 611.7 | 756.0 | 1.2× |

**Takeaway**: MoE active-3B beats dense-27B at inference because actual compute per forward pass is smaller. Use 35B-A3B unless you specifically need dense.

### TTFT (cold cache)

All configs: **0.11-0.21 s** single-stream, **0.49-0.80 s** at N=16 concurrent. Latency-friendly.

## Repo structure

```
qwen3.6-nvfp4-toolkit/
├── README.md                       # this file
├── docs/
│   └── PITFALLS.md                 # full pitfall table + workarounds
├── calibration/
│   ├── v8-rtn-llmcompressor.py     # main RTN quantization (5-16 min)
│   ├── v7-modelopt-export.py       # alternative modelopt path (output not vllm-compatible currently)
│   └── repack-multimodal-patch.py  # patch text-only ckpt → multimodal wrapper for SGLang
├── deploy/
│   ├── sglang-launch.sh            # SGLang launch (correct flags)
│   ├── chat-template-inject-cn.sh  # inject "首先" for Chinese thinking
│   └── inference-frameworks-matrix.md
├── bench/
│   ├── perf-bench.py               # TTFT + TPS + concurrency
│   ├── v8-30q-bench.py             # 30 reasoning + tool-call questions
│   └── wet-test.py                 # quick chat/think/tool sanity check
└── examples/
    └── client-curl-examples.sh
```

## Contributing

Issues and PRs welcome. Please reproduce numbers on your hardware and report deltas in issues.

## Citation

If you use this toolkit or the published checkpoints, please cite:

```
@misc{lynn-qwen36-nvfp4-toolkit-2026,
  author = {Lynn AI},
  title  = {Qwen3.6 NVFP4 Quantization Toolkit},
  year   = {2026},
  url    = {https://github.com/MerkyorLynn/qwen3.6-nvfp4-toolkit}
}
```

## License

Apache-2.0. Quantized checkpoints inherit the base model's [Tongyi license](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/LICENSE).
