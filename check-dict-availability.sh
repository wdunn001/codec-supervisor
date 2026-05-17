#!/bin/sh
# Release-checklist §1.7 sub-gate 2 — runtime dict availability probe.
#
# Exit policy:
#   * Token-stream dicts (CODEC_*_DICT_*_PATH, single file) — REQUIRED.
#     Missing/empty → exit 1 (release-blocker per §1.7).
#   * Latent dicts (CODEC_*_DICT_*_DIR, directory of *.dict)         — OPTIONAL.
#     Missing → WARN only; the engine is documented to fall back to
#     dict-less zstd or gzip (see Dockerfile.comfyui line ~110).
#     §1.7 scope is "wdunn001/codec-{sglang,vllm,llamacpp}:vX.Y" —
#     latent engines (comfyui/diffusers) are explicitly out of scope
#     for the release-blocker portion.
#
# Used by: docker run --rm --entrypoint /opt/codec/check-dict-availability.sh <image>

set -u
status=0
checked=0

for var in $(env | grep -oE "^CODEC_[A-Z0-9_]*DICT[A-Z0-9_]*PATH" || true); do
    eval "path=\$$var"
    checked=$((checked + 1))
    if [ -z "$path" ]; then
        printf "FAIL %s: empty value\n" "$var" >&2; status=1
    elif [ ! -f "$path" ]; then
        printf "FAIL %s: file not found: %s\n" "$var" "$path" >&2; status=1
    elif [ ! -s "$path" ]; then
        printf "FAIL %s: file is zero bytes: %s\n" "$var" "$path" >&2; status=1
    else
        bytes=$(wc -c < "$path")
        printf "ok   %s: %s (%s bytes)\n" "$var" "$path" "$bytes"
    fi
done

for var in $(env | grep -oE "^CODEC_[A-Z0-9_]*DICT[A-Z0-9_]*DIR" || true); do
    eval "dir=\$$var"
    checked=$((checked + 1))
    if [ -z "$dir" ]; then
        printf "warn %s: empty value (latent dicts optional — fork falls back to gzip)\n" "$var"
    elif [ ! -d "$dir" ]; then
        printf "warn %s: dir not found: %s (optional)\n" "$var" "$dir"
    else
        count=0
        for f in "$dir"/*.dict; do
            [ -s "$f" ] && count=$((count + 1))
        done
        if [ "$count" -eq 0 ]; then
            printf "warn %s: %s has no non-empty *.dict files (optional)\n" "$var" "$dir"
        else
            printf "ok   %s: %s (%s non-empty *.dict files)\n" "$var" "$dir" "$count"
        fi
    fi
done

if [ "$checked" -eq 0 ]; then
    echo "FAIL no CODEC_*_DICT_* env vars set" >&2
    status=1
fi

exit "$status"
