#!/usr/bin/env python3
"""v8-RTN — llmcompressor QuantizationModifier (RTN/absmax) NVFP4

Verified working recipe for Qwen3.6 family (35B-A3B MoE / 27B dense).
Output: compressed-tensors `nvfp4-pack-quantized` format, single safetensors file.
Time: ~16 min for 35B-A3B (41 layers / ~3000 modules each), ~5 min for 27B (64 layers / ~496 modules each).

Usage:
  python v8-rtn-llmcompressor.py \\
    --model-path /path/to/Qwen3.6-35B-A3B \\
    --output-dir /path/to/output-NVFP4
"""
import argparse
import os

# Avoid torch.compile (Triton needs gcc + libcuda.so.1 reachable)
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="path to BF16 source (HF format)")
    ap.add_argument("--output-dir", required=True, help="output dir for NVFP4 ckpt")
    ap.add_argument("--max-seq", type=int, default=256)
    ap.add_argument("--moe", action="store_true",
                    help="add MoE-specific ignore patterns (router/gate/shared_expert_gate). "
                         "Auto-detected if config has num_experts.")
    args = ap.parse_args()

    from datasets import Dataset
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor import oneshot
    from transformers import AutoTokenizer, AutoConfig

    # llmcompressor 0.10.1a still forwards the deprecated `use_auth_token`
    # kwarg into AutoModelForCausalLM.from_pretrained().  transformers 5.x
    # passes that through to Qwen3_5MoeForCausalLM.__init__(), which rejects it.
    # Keep the compatibility shim local to this script instead of patching the
    # site-package globally.
    try:
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

        _orig_qwen35_moe_init = Qwen3_5MoeForCausalLM.__init__

        def _qwen35_moe_init_compat(self, config, *model_args, **kwargs):
            kwargs.pop("use_auth_token", None)
            return _orig_qwen35_moe_init(self, config, *model_args, **kwargs)

        Qwen3_5MoeForCausalLM.__init__ = _qwen35_moe_init_compat
        print("  applied transformers-5.x use_auth_token compatibility shim")
    except Exception as exc:
        print(f"  WARN: compatibility shim not applied: {type(exc).__name__}: {exc}")

    print("=" * 60)
    print("v8-RTN — QuantizationModifier (no Hessian) NVFP4")
    print(f"  model: {args.model_path}")
    print(f"  output: {args.output_dir}")
    print("=" * 60)

    cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    is_moe = args.moe or any(getattr(cfg, "num_experts", None),
                             getattr(getattr(cfg, "text_config", None), "num_experts", None))
    print(f"  detected: {'MoE' if is_moe else 'dense'}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # RTN doesn't use calibration data, but llmcompressor's pipeline requires
    # a non-empty dataloader. 4 dummy samples is enough.
    dummy = [
        "首先,介绍北京。",
        "请帮我推理一道数学题。",
        "用 Python 写快速排序。",
        "查询天气工具调用。",
    ]
    ds = Dataset.from_list([{"text": t} for t in dummy])
    def tok(x):
        return tokenizer(x["text"], max_length=args.max_seq, truncation=True,
                         padding=False, return_tensors=None)
    ds = ds.map(tok, remove_columns=ds.column_names)

    # Common skip patterns (kept BF16): lm_head, embeddings, vision encoder.
    # MoE adds router / mlp.gate / shared_expert_gate.
    ignore = [
        "lm_head",
        "re:.*embed_tokens.*",
        "re:.*visual.*",
        "re:.*vision.*",
    ]
    if is_moe:
        ignore += [
            "re:.*router.*",
            "re:.*mlp.gate.*",
            "re:.*shared_expert_gate.*",
        ]

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="NVFP4",  # 4-bit weights + 4-bit activations, group_size=16
        ignore=ignore,
    )

    # Auto-pick decoder layer name for sequential pipeline
    seq_targets = ["Qwen3_5MoeDecoderLayer"] if is_moe else ["Qwen3_5DecoderLayer"]

    oneshot(
        model=args.model_path,
        processor=tokenizer,
        dataset=ds,
        recipe=recipe,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq,
        num_calibration_samples=len(ds),
        save_compressed=True,
        trust_remote_code_model=True,
        pipeline="sequential",
        sequential_targets=seq_targets,
    )

    print(f"\n✅ NVFP4 ckpt → {args.output_dir}")
    print("Next steps:")
    print(f"  1. python repack-multimodal-patch.py --ckpt-dir {args.output_dir} "
          f"--vision-source <path-to-FP8-base>")
    print("  2. bash deploy/sglang-launch.sh <ckpt-dir>")


if __name__ == "__main__":
    main()
