#!/usr/bin/env bash
# End-to-end smoke test against a running codec-supervisor.
# Verifies admin endpoints + the codec msgpack stream.
#
# Usage:
#   scripts/smoke.sh                            # local, port 8080
#   scripts/smoke.sh http://192.168.1.88:8080   # remote
set -euo pipefail

BASE="${1:-http://localhost:8080}"
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

echo "[1/5] supervisor /health"
curl -sf "$BASE/health" | tee "$TMPDIR/health.json"; echo

echo "[2/5] /admin/status"
curl -sf "$BASE/admin/status" | tee "$TMPDIR/status.json"; echo

CURRENT_MODEL=$(python3 -c "import json,sys; print(json.load(open('$TMPDIR/status.json'))['current_model'] or '')")
if [ -z "$CURRENT_MODEL" ]; then
    echo "    no backend loaded — skipping inference tests"
    exit 0
fi
echo "    backend serving: $CURRENT_MODEL"

echo "[3/5] /admin/models (registry)"
curl -sf "$BASE/admin/models" | python3 -m json.tool

echo "[4/5] OpenAI-compat JSON SSE proxy sanity"
curl -sN "$BASE/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"x","prompt":"Explain entropy in one sentence:","max_tokens":40,"stream":true}' \
    | head -3
echo

echo "[5/5] Codec msgpack stream"
curl -sN "$BASE/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"x","prompt":"Explain entropy in one sentence:","max_tokens":40,"stream":true,"stream_format":"msgpack"}' \
    -o "$TMPDIR/codec.bin"

ls -la "$TMPDIR/codec.bin"
python3 - "$TMPDIR/codec.bin" <<'PY'
import io, sys, msgpack
data = open(sys.argv[1], "rb").read()
print(f"  bytes: {len(data)}")
total_ids, n = 0, 0
for frame in msgpack.Unpacker(io.BytesIO(data), raw=False):
    n += 1
    total_ids += len(frame.get("ids", []))
    if n <= 3 or frame.get("done"):
        print(f"  frame[{n}]: ids={len(frame.get('ids', []))} done={frame.get('done')} fr={frame.get('finish_reason')}")
print(f"  total frames: {n}, total ids: {total_ids}")
if total_ids:
    print(f"  bytes/token: {len(data)/total_ids:.2f}")
else:
    print("  ERROR: no token ids decoded — codec patch may not be active")
    sys.exit(1)
PY

echo
echo "[done] codec-supervisor smoke test passed"
