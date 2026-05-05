#!/usr/bin/env python3
"""Quick sanity check — chat / thinking / tool-call.

Usage:
  python wet-test.py --base http://127.0.0.1:18099/v1 --model qwen36-nvfp4

Pass criteria:
  chat       -> non-empty content
  thinking   -> reasoning_content >= 80 chars + answer present
  tool-call  -> tool_calls[0].function.name matches expected
"""
import argparse
import json
import time
import urllib.request


def req(base, payload, timeout=60):
    r = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            body = resp.read().decode()
        return json.loads(body), time.time() - t0, None
    except Exception as e:
        return None, time.time() - t0, str(e)[:120]


def t1_chat(base, model):
    print("\n[Test 1] simple chat (Chinese)")
    r, dt, err = req(base, {"model": model, "messages": [{"role": "user", "content": "用一句话介绍北京"}],
                            "max_tokens": 200, "temperature": 0.3})
    if err:
        print(f"  ❌ {err}")
        return False
    msg = r["choices"][0]["message"]
    txt = msg.get("content", "") or msg.get("reasoning_content", "") or ""
    print(f"  ⏱ {dt:.2f}s  📝 {len(txt)} chars")
    print(f"  → {txt[:120]}")
    return len(txt) > 0


def t2_thinking(base, model):
    print("\n[Test 2] thinking (reasoning model)")
    r, dt, err = req(base, {"model": model, "messages": [{"role": "user",
        "content": "小明 3 个苹果,小红比他多 5 个,两人一共多少?写出推理过程。"}],
        "max_tokens": 400, "temperature": 0})
    if err:
        print(f"  ❌ {err}")
        return False
    msg = r["choices"][0]["message"]
    txt = msg.get("content", "") or ""
    rc = msg.get("reasoning_content", "") or ""
    print(f"  ⏱ {dt:.2f}s  📝 content {len(txt)} chars / reasoning {len(rc)} chars")
    if rc:
        print(f"  → reasoning [:120]: {rc[:120]}")
    return ("11" in txt or "11" in rc) and len(rc) >= 80


def t3_tool_call(base, model):
    print("\n[Test 3] tool-call")
    tools = [{"type": "function", "function": {"name": "get_weather",
        "description": "查城市天气", "parameters": {
            "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    r, dt, err = req(base, {"model": model, "messages": [{"role": "user", "content": "帮我查北京今天天气"}],
                            "tools": tools, "tool_choice": "auto", "max_tokens": 200, "temperature": 0})
    if err:
        print(f"  ❌ {err}")
        return False
    msg = r["choices"][0]["message"]
    tc = msg.get("tool_calls", [])
    print(f"  ⏱ {dt:.2f}s  🔧 {len(tc)} tool_calls")
    if tc:
        fn = tc[0].get("function", {})
        print(f"  → {fn.get('name')}({fn.get('arguments')})")
    return len(tc) > 0 and tc[0].get("function", {}).get("name") == "get_weather"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18000/v1")
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    results = {
        "chat": t1_chat(args.base, args.model),
        "thinking": t2_thinking(args.base, args.model),
        "tool_call": t3_tool_call(args.base, args.model),
    }

    print("\n=== Summary ===")
    for k, v in results.items():
        print(f"  {'✅' if v else '❌'} {k}")
    p = sum(1 for v in results.values() if v)
    print(f"\n{p}/{len(results)} passed")
    raise SystemExit(0 if p == len(results) else 1)


if __name__ == "__main__":
    main()
