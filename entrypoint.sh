#!/usr/bin/env bash
# codec-supervisor entrypoint. Forwards CODEC_* env vars (set in Dockerfile or
# at `docker run` time) into the supervisor CLI; everything else is delegated
# to argparse defaults inside the package.
set -euo pipefail

# Print whichever backend HEAD file is present in the image so the user can
# see the exact upstream commit baked in. Each Dockerfile.* writes one of:
#   /opt/codec/CODEC_SGLANG_HEAD
#   /opt/codec/CODEC_VLLM_HEAD
#   /opt/codec/CODEC_LLAMACPP_HEAD
for f in /opt/codec/CODEC_*_HEAD; do
    [ -f "$f" ] || continue
    backend=$(basename "$f" | sed -e 's/^CODEC_//' -e 's/_HEAD$//' | tr '[:upper:]' '[:lower:]')
    echo "[codec] ${backend} patched from $(cat "$f")"
done

exec codec-supervisor "$@"
