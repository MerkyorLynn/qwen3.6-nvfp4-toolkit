#!/usr/bin/env python3
"""Repack a text-only NVFP4 ckpt → multimodal wrapper format.

Why: llmcompressor outputs `architectures: ["Qwen3_5MoeForCausalLM"]` (text-only).
SGLang only registers `Qwen3_5MoeForConditionalGeneration` (multimodal). Loading
the text-only architecture fails with:
  ValueError: Qwen3_5MoeForCausalLM has no SGlang implementation

This script:
  1. Replaces config.json with a multimodal source template (preserves quantization_config)
  2. Splices vision weights from the FP8 base (model.visual.* @ bf16, ~1.4 GB, 333 tensors)
  3. Regenerates model.safetensors.index.json so SGLang finds every key

Usage:
  python repack-multimodal-patch.py \\
    --ckpt-dir /path/to/Qwen3.6-35B-A3B-NVFP4-v8-rtn \\
    --vision-source /path/to/Qwen3.6-35B-A3B-FP8

Vision weights are kept BF16 (not quantized) so vision inference works exactly
as the base model.

Time: ~5 min (read 21 GB + splice + write 21 GB single shard).
RAM: ~30 GB peak (loads everything in RAM before save_file).
"""
import argparse
import json
import os
import shutil

from safetensors import safe_open
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True, help="dir of text-only NVFP4 ckpt to patch")
    ap.add_argument("--vision-source", required=True, help="dir of multimodal base ckpt (e.g. FP8 base) for vision weights")
    args = ap.parse_args()

    ckpt = args.ckpt_dir
    src = args.vision_source

    # -- Step 1: config.json: src multimodal template + ckpt's quantization_config --
    print("=== Step 1: patch config.json ===")
    shutil.copy(f"{ckpt}/config.json", f"{ckpt}/config.json.causal.bak")
    src_cfg = json.load(open(f"{src}/config.json"))
    ckpt_cfg = json.load(open(f"{ckpt}/config.json"))
    new_cfg = dict(src_cfg)
    new_cfg["quantization_config"] = ckpt_cfg["quantization_config"]
    json.dump(new_cfg, open(f"{ckpt}/config.json", "w"), indent=2, ensure_ascii=False)
    print(f"  architectures: {new_cfg['architectures']}")
    print(f"  model_type: {new_cfg['model_type']}")

    # -- Step 2: copy multimodal aux configs --
    print("\n=== Step 2: copy preprocessor configs ===")
    for fn in ["preprocessor_config.json", "video_preprocessor_config.json"]:
        sp = f"{src}/{fn}"
        if os.path.exists(sp):
            shutil.copy(sp, f"{ckpt}/{fn}")
            print(f"  copied: {fn}")

    # -- Step 3: locate vision weights in src ckpt --
    print("\n=== Step 3: scan src ckpt for vision weights ===")
    if os.path.exists(f"{src}/model.safetensors.index.json"):
        idx = json.load(open(f"{src}/model.safetensors.index.json"))
        vk = [k for k in idx["weight_map"]
              if k.startswith("model.visual.") or k.startswith("visual.")]
        files = sorted(set(idx["weight_map"][k] for k in vk))
    else:
        # single-file source
        with safe_open(f"{src}/model.safetensors", framework="pt") as f:
            vk = [k for k in f.keys() if k.startswith("model.visual.") or k.startswith("visual.")]
        files = ["model.safetensors"]
    print(f"  visual keys: {len(vk)}, files: {len(files)}")

    # -- Step 4: load ckpt + splice vision --
    print("\n=== Step 4: load ckpt + splice vision ===")
    print("  reading ckpt...")
    all_tensors = {}
    with safe_open(f"{ckpt}/model.safetensors", framework="pt") as f:
        for k in f.keys():
            all_tensors[k] = f.get_tensor(k)
    print(f"  ckpt keys: {len(all_tensors)}")

    for fn in files:
        with safe_open(f"{src}/{fn}", framework="pt") as f:
            for k in f.keys():
                if k.startswith("model.visual.") or k.startswith("visual."):
                    target_k = "model." + k if k.startswith("visual.") else k
                    if target_k not in all_tensors:
                        all_tensors[target_k] = f.get_tensor(k)
    print(f"  total after splice: {len(all_tensors)}")

    # -- Step 5: write new model.safetensors atomically --
    print("\n=== Step 5: write new model.safetensors ===")
    out_tmp = f"{ckpt}/model.safetensors.tmp"
    save_file(all_tensors, out_tmp)
    sz_mb = os.path.getsize(out_tmp) // (1024 * 1024)
    print(f"  wrote {sz_mb} MB")

    shutil.move(f"{ckpt}/model.safetensors", f"{ckpt}/model.safetensors.causal.bak")
    shutil.move(out_tmp, f"{ckpt}/model.safetensors")

    # -- Step 6: regenerate index.json --
    total_size = sum(t.numel() * t.element_size() for t in all_tensors.values())
    weight_map = {k: "model.safetensors" for k in all_tensors}
    idx_out = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    json.dump(idx_out, open(f"{ckpt}/model.safetensors.index.json", "w"), indent=2)
    print(f"  index.json: {len(weight_map)} entries")

    print("\n✅ DONE — ckpt patched to multimodal wrapper format")
    print(f"   Original text-only ckpt backed up to:")
    print(f"     {ckpt}/config.json.causal.bak")
    print(f"     {ckpt}/model.safetensors.causal.bak")


if __name__ == "__main__":
    main()
