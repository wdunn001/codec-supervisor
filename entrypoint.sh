#!/usr/bin/env bash
# codec-supervisor entrypoint. Forwards CODEC_* env vars (set in Dockerfile or
# at `docker run` time) into the supervisor CLI; everything else is delegated
# to argparse defaults inside the package.
set -euo pipefail

if [ -f /opt/codec/CODEC_SGLANG_HEAD ]; then
    echo "[codec] sglang patched from $(cat /opt/codec/CODEC_SGLANG_HEAD)"
fi

exec codec-supervisor "$@"
