#!/usr/bin/env bash
# Pull the small result files (jsonl logs, kappa tables, rollout rows) out of a molab sandbox into this repo,
# so a sandbox termination does not lose them. Run from the Claude side, where execute-code.sh works.
#   bash pull_results.sh <notebook-url> <token> [dest-dir]
set -euo pipefail
URL=$1; TOKEN=$2; DEST=${3:-$(cd "$(dirname "$0")" && pwd)/results}
SKILL=${SKILL:-/home/user/data-forcing-distillation/.claude/skills/marimo-pair/scripts/execute-code.sh}
mkdir -p "$DEST"
MARIMO_TOKEN=$TOKEN bash "$SKILL" --url "$URL" -c "import subprocess,base64; subprocess.run(['bash','-c','cd /marimo/work && tar czf /tmp/results.tgz \$(find data -name \"*.jsonl\" 2>/dev/null) \$(ls *.log 2>/dev/null) 2>/dev/null']); print('TGZ:'+base64.b64encode(open('/tmp/results.tgz','rb').read()).decode())" 2>/dev/null \
  | grep -o "TGZ:.*" | cut -c5- | base64 -d > "$DEST/results.tgz"
tar xzf "$DEST/results.tgz" -C "$DEST" && rm "$DEST/results.tgz"
date -u +%Y-%m-%dT%H:%M:%SZ > "$DEST/PULLED_AT"
find "$DEST" -name "*.jsonl" | sed "s|$DEST/||" | sort
