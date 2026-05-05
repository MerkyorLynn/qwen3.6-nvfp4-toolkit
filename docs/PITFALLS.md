# Pitfalls — Qwen3.6 NVFP4 Quantization & Deployment

Compiled from 2026-05-04/05 实测. All numbers and exit codes verified on actual hardware (Spark GB10 sm_121, RTX 5090 sm_120, RTX 4090 sm_89).

---

## 1. Inference framework compatibility matrix

| Framework | Image / Version | Status | Root cause |
|---|---|---|---|
| **SGLang dev-cu13** | `lmsysorg/sglang:dev-cu13` (sglang `0.0.0.dev1+gcbc2bee54`) | ✅ **WORKS** | bundled `transformers 5.8.0.dev0`, registered `Qwen3_5MoeForConditionalGeneration` + `Qwen3_5ForConditionalGeneration`, native MTP/NEXTN, hybrid attention handled |
| vLLM 0.17.1+nvinternal (NGC 26.03) | `nvcr.io/nvidia/vllm:26.03.post1-py3` | ❌ | `transformers 4.57.3` doesn't recognize `qwen3_5_moe` model_type (KeyError); upgrading to 5.x breaks vLLM's `AutoModelForVision2Seq` import |
| vLLM 0.20.1.dev0 (AEON-7 v3) | `ghcr.io/aeon-7/vllm-aeon-ultimate-dflash:qwen36-v3` | ❌ | `unify_kv_cache_spec_page_size NotImplementedError` for hybrid attention (linear_attn 4-interval + full_attn) + MoE |
| vLLM 0.20.1 (5090, latest) | `pip install vllm` | ❌ | `Qwen3_5ForCausalLM has no implementation` LinearAttention not supported |
| TensorRT-LLM 1.2.0 (NGC 26.03) | `nvcr.io/nvidia/tritonserver:26.03-trtllm-python-py3` | ❌ | Same `qwen3_5_moe` mismatch + `PartReasoningText` openai SDK incompat |

**Bottom line**: as of 2026-05-05, only SGLang dev-cu13 works for these checkpoints. Wait for vLLM 0.21+ or upstream support.

---

## 2. Architecture mismatch trap (text-only vs multimodal)

The Qwen3.6 base models are **multimodal wrappers**:
- 35B-A3B → `Qwen3_5MoeForConditionalGeneration` (MoE)
- 27B → `Qwen3_5ForConditionalGeneration` (dense)

But quantization tools (modelopt, llmcompressor) default to text-only init:
- output ckpt has `architectures: ["Qwen3_5MoeForCausalLM"]` or `["Qwen3_5ForCausalLM"]`

**SGLang strict checks architectures** and rejects:
```
ValueError: Qwen3_5MoeForCausalLM has no SGlang implementation
```

### Fix (in this repo)

`calibration/repack-multimodal-patch.py` does:
1. Replaces `config.json` with the multimodal source template, preserving the `quantization_config` field.
2. Splices the FP8 base's vision weights (333 tensors @ bf16) into the quantized text weights (`model.visual.*` keys).
3. Regenerates `model.safetensors.index.json` so SGLang can find every key.

The vision encoder is **not quantized**; it's bf16, taken verbatim from the FP8 base. Vision inference paths therefore work exactly as the base model.

---

## 3. Chinese thinking requires `chat_template` injection (`首先`)

Out-of-the-box, the Qwen3.6 reasoning models emit **English thinking** even on Chinese prompts:
```
Here's a thinking process:
1. Analyze User Input: ...
```

To force Chinese thinking, inject `首先` into `chat_template.jinja` near the `<think>` marker:

```jinja
{# original (line ~152) #}
{{- '<think>\n' }}
{# patched #}
{{- '<think>\n首先' }}
```

`deploy/chat-template-inject-cn.sh` does this safely (with backup).

### Performance trade-off (Spark sm_121, measured 2026-05-05)

| Metric | English thinking (default) | Chinese thinking (`首先`) | Δ |
|---|---|---|---|
| short single TPS (chars/s) | 103.2 | 39.3 | -62% |
| N=16 concurrent agg | 756.0 | 322.9 | -57% |

Chinese characters carry more information per token, so chars/s drops. **Total tokens/sec is similar** — pick injection only if you need Chinese reasoning content quality.

The companion FP8 base ships with `首先` already injected, so direct comparisons should match injection state.

---

## 4. Hybrid attention requires SGLang-specific flags

Qwen3.6 models have `linear_attention` + `full_attention` interleaved (4-interval). SGLang launch needs:

```
--mamba-scheduler-strategy extra_buffer
--page-size 64
```

Without these, mamba/attention page sizes don't unify and engine init fails.

vLLM doesn't have an equivalent — its `unify_kv_cache_spec_page_size` raises `NotImplementedError`. This is the underlying reason vLLM 0.20.1.dev0 cannot serve these models even after fixing transformers.

---

## 5. mem-fraction guidance (Spark / multi-GPU)

Empirical limits from 2026-04 to 2026-05 production:

| Scenario | Limit | Above this → |
|---|---|---|
| Spark single LLM + voice services (Qwen3-ASR + emotion2vec + CosyVoice) | **0.70** | OOM/kernel panic within 30 min |
| Spark LLM-only | **0.85** | safe, but no buffer for spike |
| 5090 dual-GPU (TP=2), no voice | **0.85** | tested |
| Any Spark multi-LLM | **forbidden** | OOM panic certain |

Override via `LYNN_SPARK_MEM_FRACTION` env (range 0.50-0.70 for Spark+voice).

---

## 6. Quantization choice: RTN vs GPTQ

We tried GPTQ first via `llmcompressor.modifiers.quantization.GPTQModifier` with `NUM_SAMPLES = 256`, `64`, `16`. Per-layer time barely changes (16 → 11 min/layer because Hessian computation is the bottleneck, not forward calibration).

For 35B-A3B (41 layers), GPTQ would take **~8 hours**.

**RTN (`QuantizationModifier`) takes ~16 minutes for 35B-A3B**, ~5 min for 27B. Quality difference on V8 30Q reasoning bench: THINK 100% pass on both, TOOL pass rate similar. **Recommend RTN unless you have very long calibration budget.**

---

## 7. Calibration data trap

`QuantizationModifier` (RTN) **does not use calibration data** — it's pure absmax weight quantization. But llmcompressor's `oneshot` pipeline still requires a non-empty dataset:

```python
# This errors:
oneshot(model=..., recipe=recipe, output_dir=...)
# TypeError: 'NoneType' object is not iterable

# Pass 4 dummy samples instead:
dummy = ["首先,介绍北京。", "请帮我推理。", ...]
ds = Dataset.from_list([{"text": t} for t in dummy])
oneshot(model=..., dataset=ds, num_calibration_samples=len(ds), ...)
```

See `calibration/v8-rtn-llmcompressor.py`.

---

## 8. Ignored layers (kept BF16)

Both 35B-A3B and 27B quantization recipes skip:
- `lm_head` (precision-sensitive)
- `*embed_tokens*`
- `*router*` (MoE expert selection — quantizing breaks gating)
- `*mlp.gate*`
- `*shared_expert_gate*`
- `*visual*` / `*vision*` (kept bf16, vision encoder not quantized)

Output sizes:
- 35B-A3B: 21 GB (text NVFP4) + 1.4 GB vision bf16 (patched in)
- 27B: 18 GB (text NVFP4) + 1.4 GB vision bf16 (patched in)

---

## 9. Cold start times

| Setup | Cold start | Note |
|---|---|---|
| SGLang dev-cu13 + 35B-A3B-NVFP4 | 3-4 min | Includes draft cuda graph capture for MTP |
| SGLang dev-cu13 + 27B-NVFP4 | 2-3 min | Smaller, dense, no MoE setup |
| vLLM 0.20.1.dev0 + 35B-A3B-NVFP4 | 13-18 min | Per stevescargall report; not reproduced (engine init fails before) |

---

## 10. PRC-network gotchas

- **HuggingFace HF mirror** — `https://hf-mirror.com` works for `Qwen/Qwen3.6-*` BF16 download. Use `aria2c -x 16 -s 16 -c` not `huggingface-cli` (xethub timeout silent deadlock).
- **PyPI / vLLM wheel** — Tsinghua mirror `pypi.tuna.tsinghua.edu.cn` may silently stall on `vllm-0.20.1` 244 MB wheel. aliyun `mirrors.aliyun.com/pypi/web/packages/...` is slow but stable (~250 KB/s).
- **NGC docker images** — pull from `nvcr.io` is slow in PRC. `nvcr.io/nvidia/tritonserver:26.03-trtllm-python-py3` (24 GB) takes ~30-60 min over residential.

---

## 11. Hardware support summary

| GPU | SM | Native FP4? | Verified? | Notes |
|---|---|---|---|---|
| Spark GB10 | sm_121 | ✅ | ✅ Production | SGLang dev-cu13 |
| RTX 5090 (dual) | sm_120 | ✅ | 🔄 vLLM install pending (PRC pip throughput) | TP=2 expected |
| RTX 4090 | sm_89 | ❌ | ❌ Not recommended | MARLIN fallback ~2-3× slower than native FP4 path |

---

## 12. Tool calling

Both 35B-A3B-NVFP4 and 27B-NVFP4 work with `--tool-call-parser qwen3_coder` in SGLang. V8 30Q empirical:
- TOOL pass rate: 50% (3/6) — passes on explicit-verb prompts ("帮我搜索 X", "帮我算 X"); fails on implicit-intent ("X 怎么样?", "今天 X")
- This pattern is consistent with the FP8 base. **Quantization does not visibly degrade tool-call ability**.

Workaround: prompt-engineer to use explicit verbs.

---

## 13. Reasoning model `max_tokens` budget

V8 30Q CODE/FIN/LONG bench got 0% pass — but this is **not a model capability issue**. The `reasoning_content` field consumed the entire 600-token budget, leaving zero for the actual answer.

For reasoning models, set `max_tokens >= 2000`. The `reasoning_parser=qwen3` correctly separates `reasoning_content` from `content`, but you must give it room.

---

## 14. Failed attempts (saved you from trying)

We tried these and they DON'T work as of 2026-05-05:

1. **modelopt 0.43 NVFP4 export** — packed format incompatible with vLLM/SGLang weight loaders (`load_merged_column_weight assertion shape mismatch`). Same key naming, same fused QKV layout, but `weight_packed [..,..]/2 uint8` vs expected `[..,..] fp8_e4m3fn`. Loader fails to dispatch unpacking.
2. **TRT-LLM 1.2.0 LLM Python API** — same transformers compat issue + openai SDK `PartReasoningText` import. Patching one breaks the other (vLLM internal `AutoModelForVision2Seq` removed in transformers 5.x).
3. **NGC vllm:26.03.post1-py3 + transformers nightly** — works for `Resolved architecture: Qwen3_5MoeForConditionalGeneration` and even gets to `Loading safetensors checkpoint shards` but fails at `parameter.py:175 load_merged_column_weight` shape assert.
4. **AEON-7 v3 + 35B-A3B-NVFP4 + `--mamba-block-size 32768 --block-size 32 --enable-prefix-caching`** — every combo fails on `unify_kv_cache_spec_page_size`.
5. **Direct architectures swap (text-only ckpt + multimodal config.json)** — SGLang loads but trips on missing `model.visual.*` weights. The repack-multimodal-patch.py step is mandatory.

Save 6+ hours by reading this section before retrying any of the above.
