# syntax=docker/dockerfile:1.7
#
# codec/sglang — easily-deployable Codec inference instance.
#
# Layers:
#   1. lmsysorg/sglang:latest — base CUDA + sglang + kernels
#   2. wdunn001/sglang feat/codec-server-side-agent — the two open Codec PRs
#      (#24483 binary transport, #24557 ToolWatcher) overlaid via editable install
#   3. codec-supervisor — admin/control-plane FastAPI on :8080
#
# Result: a single container with `docker run` semantics. Hit :8080 and you
# get an OpenAI-compatible server plus /admin endpoints to upload / pull /
# swap models.

ARG SGLANG_BASE=lmsysorg/sglang:latest
FROM ${SGLANG_BASE}

ARG CODEC_SGLANG_REPO=https://github.com/wdunn001/sglang.git
ARG CODEC_SGLANG_REF=feat/codec-server-side-agent
ARG CODEC_SGLANG_COMMIT=

LABEL org.opencontainers.image.title="codec-sglang" \
      org.opencontainers.image.description="SGLang with Codec PRs (#24483, #24557) + codec-supervisor control plane" \
      org.opencontainers.image.source="https://github.com/wdunn001/codec-supervisor" \
      org.opencontainers.image.url="https://codecai.net" \
      org.opencontainers.image.vendor="Quasarke" \
      org.opencontainers.image.licenses="Apache-2.0 AND BUSL-1.1"

# ---------- 1. overlay codec patches on stock sglang ----------
WORKDIR /opt/codec
RUN --mount=type=cache,target=/root/.cache/pip,id=codec-pip \
    git clone --depth 50 --branch "${CODEC_SGLANG_REF}" "${CODEC_SGLANG_REPO}" sglang \
 && cd sglang \
 && if [ -n "${CODEC_SGLANG_COMMIT}" ]; then \
        git fetch --depth 50 origin "${CODEC_SGLANG_COMMIT}" \
        && git checkout "${CODEC_SGLANG_COMMIT}"; \
    fi \
 && git rev-parse HEAD > /opt/codec/CODEC_SGLANG_SHA \
 && git log -1 --pretty='%h %s' > /opt/codec/CODEC_SGLANG_HEAD \
 && cd python \
 && pip install --no-deps -e . \
 && pip install msgpack brotli zstandard \
 && pip install --upgrade "sglang-kernel>=0.4.2.post1"

# ---------- 1b. fetch reference Codec zstd dicts so dict-zstd works out of the box ----------
# Per spec/PROTOCOL.md "Pre-trained ZSTD dictionaries", a server MUST load
# a dict before the negotiator can pick zstd. Baking the canonical dicts
# from Codec/dictionaries/ into the image means a fresh `docker run` of
# wdunn001/codec-sglang:latest unlocks the zstd column of the bench
# matrix without operator action. Operators who want their own dicts
# can mount over /opt/codec/dicts at runtime or override the env vars
# below to point at a different path.
RUN mkdir -p /opt/codec/dicts \
 && curl -fsSL -o /opt/codec/dicts/qwen2.5-synth-msgpack-v1.dict \
      https://raw.githubusercontent.com/wdunn001/Codec/main/dictionaries/qwen2.5-synth-msgpack-v1.dict \
 && curl -fsSL -o /opt/codec/dicts/qwen2.5-synth-protobuf-v1.dict \
      https://raw.githubusercontent.com/wdunn001/Codec/main/dictionaries/qwen2.5-synth-protobuf-v1.dict

# ---------- 2. install codec-supervisor ----------
COPY pyproject.toml /opt/codec/supervisor/pyproject.toml
COPY codec_supervisor /opt/codec/supervisor/codec_supervisor
COPY README.md /opt/codec/supervisor/README.md
RUN --mount=type=cache,target=/root/.cache/pip,id=codec-pip \
    pip install /opt/codec/supervisor

# ---------- 3. runtime config ----------
ENV CODEC_HOST=0.0.0.0 \
    CODEC_PORT=8080 \
    CODEC_BACKEND=sglang \
    CODEC_BACKEND_HOST=127.0.0.1 \
    CODEC_BACKEND_PORT=30000 \
    CODEC_MODELS_DIR=/models \
    CODEC_INITIAL_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
    CODEC_BACKEND_ARGS="--mem-fraction-static 0.85 --attention-backend triton" \
    CODEC_LOG_LEVEL=INFO \
    CODEC_ZSTD_DICT_MSGPACK_PATH=/opt/codec/dicts/qwen2.5-synth-msgpack-v1.dict \
    CODEC_ZSTD_DICT_PROTOBUF_PATH=/opt/codec/dicts/qwen2.5-synth-protobuf-v1.dict

VOLUME ["/models"]
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=3 \
  CMD python3 -c "import os,urllib.request,sys; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"CODEC_PORT\",\"8080\")}/health', timeout=4); sys.exit(0)" || exit 1

COPY entrypoint.sh /opt/codec/entrypoint.sh
RUN chmod +x /opt/codec/entrypoint.sh

ENTRYPOINT ["/opt/codec/entrypoint.sh"]
