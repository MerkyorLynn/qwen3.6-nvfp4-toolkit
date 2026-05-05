#!/usr/bin/env bash
# Inject "首先" into chat_template.jinja for Chinese thinking
# Usage: bash chat-template-inject-cn.sh <ckpt-dir>
#
# See PITFALLS.md §3 for the performance trade-off.
set -euo pipefail

CKPT="${1:-}"
if [[ -z "$CKPT" || ! -f "$CKPT/chat_template.jinja" ]]; then
  echo "Usage: $0 <ckpt-dir>"
  echo "  ckpt-dir must contain chat_template.jinja"
  exit 1
fi

TPL="$CKPT/chat_template.jinja"
BAK="$CKPT/chat_template.jinja.bak.$(date +%Y%m%d-%H%M%S)"

# Already injected?
if grep -q "<think>\\\\n首先" "$TPL"; then
  echo "✅ already injected, no change"
  exit 0
fi

# Backup
cp "$TPL" "$BAK"
echo "  backup: $BAK"

# Inject — the standard Qwen3 template has a `{{- '<think>\n' }}` line just
# before the assistant marker. Replace with `{{- '<think>\n首先' }}`.
sed -i.tmp "s|{{- '<think>\\\\n' }}|{{- '<think>\\\\n首先' }}|" "$TPL"
rm -f "$TPL.tmp"

if grep -q "<think>\\\\n首先" "$TPL"; then
  echo "✅ injected '首先'"
  echo "  patched line:"
  grep "<think>" "$TPL" | head -2
else
  echo "❌ injection failed (template format unexpected)"
  echo "  Restoring from backup..."
  cp "$BAK" "$TPL"
  exit 1
fi

echo
echo "Restart your inference server (e.g. 'docker restart <name>') for the template to take effect."
