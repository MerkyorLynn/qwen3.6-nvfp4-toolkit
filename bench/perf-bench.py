#!/usr/bin/env python3
"""Perf bench — TTFT + TPS + concurrency, OpenAI-compatible API.

Usage:
  python perf-bench.py --base http://127.0.0.1:18099/v1 --model qwen36-nvfp4

Bench: 3 runs median per scenario, all temperature=0.
Scenarios: short/medium/long single-stream, N=4/8/16 concurrent.

Output: stdout table + optional JSON dump.
"""
import argparse
import json
import statistics
import threading
import time
import urllib.request


def stream_one(base, model, prompt, max_tokens=128, timeout=120):
    """Single streaming request. Returns (TTFT, total_secs, chars_per_sec) or (None, None, None)."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    chars = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for ln in resp:
                ln = ln.decode().strip()
                if not ln.startswith("data: "):
                    continue
                if ln == "data: [DONE]":
                    break
                try:
                    d = json.loads(ln[6:])
                    delta = d.get("choices", [{}])[0].get("delta", {})
                    tok = delta.get("content", "") or delta.get("reasoning_content", "")
                    if tok:
                        if ttft is None:
                            ttft = time.time() - t0
                        chars += len(tok)
                except Exception:
                    pass
    except Exception as e:
        print(f"    request failed: {e}")
        return None, None, None
    total = time.time() - t0
    return ttft, total, (chars / total if total > 0 else 0)


def single(base, model, prompt, label, max_tokens=128, runs=3):
    print(f"\n[{label}] (single, max_tokens={max_tokens})")
    samples = []
    for i in range(runs):
        ttft, total, tps = stream_one(base, model, prompt, max_tokens)
        if ttft is None:
            print(f"  run {i+1}: FAIL")
            continue
        samples.append((ttft, total, tps))
        print(f"  run {i+1}: TTFT {ttft:.2f}s total {total:.2f}s TPS {tps:.1f} chars/s")
    if not samples:
        return None
    m_ttft = statistics.median(s[0] for s in samples)
    m_tot = statistics.median(s[1] for s in samples)
    m_tps = statistics.median(s[2] for s in samples)
    print(f"  📊 median TTFT {m_ttft:.2f}s total {m_tot:.2f}s TPS {m_tps:.1f}")
    return {"ttft_median": m_ttft, "total_median": m_tot, "tps_median": m_tps}


def concurrent(base, model, prompt, n, label, max_tokens=128):
    print(f"\n[{label}] N={n} concurrent (max_tokens={max_tokens})")
    results = [None] * n

    def worker(i):
        results[i] = stream_one(base, model, prompt, max_tokens)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    ok = [r for r in results if r and r[0] is not None]
    if not ok:
        print(f"  ❌ all {n} failed")
        return None
    avg_ttft = sum(r[0] for r in ok) / len(ok)
    agg_tps = sum(r[2] for r in ok)
    print(f"  wall {wall:.2f}s avg TTFT {avg_ttft:.2f}s aggregate TPS {agg_tps:.1f} chars/s ({len(ok)}/{n} ok)")
    return {"wall": wall, "avg_ttft": avg_ttft, "agg_tps": agg_tps, "ok": len(ok), "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18000/v1")
    ap.add_argument("--model", required=True, help="served-model-name")
    ap.add_argument("--out", help="optional JSON output path")
    args = ap.parse_args()

    print(f"=== perf bench ===")
    print(f"  endpoint: {args.base}")
    print(f"  model: {args.model}")
    print(f"  time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    short = "用一句话介绍北京。"
    medium = "解释 transformer 注意力机制,300 字以内。"
    long_ctx = "总结以下内容(用中文,200 字):" + ("北京是中华人民共和国首都,直辖市。" * 30)

    results = {}
    results["short_single"] = single(args.base, args.model, short, "short single", max_tokens=128)
    results["medium_single"] = single(args.base, args.model, medium, "medium single", max_tokens=512)
    results["long_single"] = single(args.base, args.model, long_ctx, "long single", max_tokens=256)
    results["concurrent_4"] = concurrent(args.base, args.model, short, 4, "short N=4")
    results["concurrent_8"] = concurrent(args.base, args.model, medium, 8, "medium N=8")
    results["concurrent_16"] = concurrent(args.base, args.model, short, 16, "short N=16")

    print("\n=== Done ===")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"endpoint": args.base, "model": args.model,
                       "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "results": results}, f, indent=2)
        print(f"  saved: {args.out}")


if __name__ == "__main__":
    main()
