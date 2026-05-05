#!/usr/bin/env bash
# SGLang launch — verified flags for Qwen3.6 NVFP4 ckpts on Blackwell-class GPUs
# Usage:
#   bash sglang-launch.sh <ckpt-dir> [--port 18099] [--name my-name] [--mtp]
set -euo pipefail

CKPT="${1:-/models/Qwen3.6-35B-A3B-NVFP4-v8-rtn}"
shift || true

PORT=18000
NAME="sglang-qwen36-nvfp4"
SERVED_NAME="qwen36-nvfp4"
MEM_FRAC=0.65
ENABLE_MTP=0
ENABLE_VOICE_COEX=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --served-name) SERVED_NAME="$2"; shift 2 ;;
    --mem) MEM_FRAC="$2"; shift 2 ;;
    --mtp) ENABLE_MTP=1; shift ;;
    --voice-coex) ENABLE_VOICE_COEX=1; MEM_FRAC=0.60; shift ;;
    *) echo "unknown: $1"; exit 1 ;;
  esac
done

# Voice coexistence: cap mem-fraction at 0.60 (memory ironclaw — see PITFALLS.md §5)
if [[ "$ENABLE_VOICE_COEX" == "1" && $(echo "$MEM_FRAC > 0.60" | bc) == "1" ]]; then
  echo "⚠️  --voice-coex requires --mem ≤ 0.60. Forcing 0.60."
  MEM_FRAC=0.60
fi

ARGS=(
  python3 -m sglang.launch_server
  --model-path "$CKPT"
  --port "$PORT" --host 0.0.0.0
  --quantization compressed-tensors
  --reasoning-parser qwen3
  --tool-call-parser qwen3_coder
  # Hybrid attention (linear_attn + full_attn 4-interval) requires these:
  --mamba-scheduler-strategy extra_buffer
  --page-size 64
  --mem-fraction-static "$MEM_FRAC"
  --kv-cache-dtype auto
  --trust-remote-code
  --served-model-name "$SERVED_NAME"
)

# Optional: NEXTN MTP (Spec V2). Only useful for MoE models with draft head.
if [[ "$ENABLE_MTP" == "1" ]]; then
  ARGS+=(
    --speculative-algorithm NEXTN
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
  )
  EXTRA_ENV=(-e SGLANG_ENABLE_SPEC_V2=1)
else
  EXTRA_ENV=()
fi

echo "=== Stop existing $NAME (if any) ==="
docker rm -f "$NAME" 2>/dev/null || true

echo "=== Start SGLang ==="
docker run -d --name "$NAME" \
  --restart unless-stopped \
  --gpus all --ipc=host \
  -v "$(dirname "$CKPT"):/models" \
  -p "${PORT}:${PORT}" \
  "${EXTRA_ENV[@]}" \
  lmsysorg/sglang:dev-cu13 \
  "${ARGS[@]}"

echo "=== Wait ready (max 8 min cold start) ==="
ready=0
for i in {1..96}; do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; then
    echo "✅ ready (${i}*5s = $((i*5))s)"
    ready=1; break
  fi
  if (( i % 12 == 0 )); then echo "  ... loading ($((i*5))s)"; fi
  if ! docker ps --format "{{.Names}}" | grep -q "^${NAME}$"; then
    echo "❌ container exited"; break
  fi
  sleep 5
done

if [[ "$ready" == "0" ]]; then
  echo "❌ NOT ready, last 60 lines:"
  docker logs --tail 60 "$NAME" 2>&1 | tail -50
  exit 1
fi

echo "=== /v1/models ==="
curl -s "http://127.0.0.1:${PORT}/v1/models" | python3 -m json.tool | head -15

echo "=== Quick chat verify ==="
curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${SERVED_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"用一句话介绍北京\"}],\"max_tokens\":80,\"temperature\":0}" \
  | python3 -m json.tool | head -25
