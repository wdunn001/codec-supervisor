# Handoff — codec-supervisor v0.3 latent modality (in flight)

**Branch:** `feat/v0.3-latent-modality` on `wdunn001/codec-supervisor`
**Companion branch:** `feat/v0.3-latent-modality` on `wdunn001/Codec` (spec + clients + reference Python encoder)
**Status (2026-05-08):** Dockerfiles + compose + README landed; the underlying engine forks do not yet exist on GitHub.

This file is a temporary handoff for the next worker. Delete before merging to `main`.

---

## What's done on this branch

| File | Change |
|---|---|
| `Dockerfile.comfyui` *(new)* | Builds `codec/comfyui:dev` from `wdunn001/ComfyUI feat/codec-latent-transport`. Same `ARG CODEC_*_REPO` / `_REF` / `_COMMIT` pattern as `Dockerfile.{vllm,sglang,llamacpp}`. CUDA 12.8 runtime base, torch 2.5.1 + diffusers 0.31.0 pinned to match the bench/golden perceptual-reference image. Ships codec-supervisor on :8080, ComfyUI on :8188 inside the container. Pre-fetches reference latent zstd dicts. |
| `Dockerfile.diffusers` *(new)* | Builds `codec/diffusers:dev` from `wdunn001/diffusers feat/codec-latent-transport`. Same pinning posture. Ships codec-supervisor on :8080, the fork's `examples/codec_server/` FastAPI wrapper on :8200. **This image doubles as the bench/golden perceptual-conformance reference** — torch + diffusers versions pinned here ARE the perceptual contract; bumping them re-pins the bench. |
| `compose.yml` | Added `codec-comfyui` and `codec-diffusers` services. Both gated behind compose profiles (`latents`, `comfyui`, `diffusers`) so a default `docker compose up` doesn't pull them — image layers are heavy (~10 GB). External ports default to 8090 (comfyui) and 8091 (diffusers); shared models + hf-cache volumes. |
| `README.md` | New "Latent modality (image / video, v0.3+)" subsection under "Codec patches". Documents the **fork-only posture** for ComfyUI and diffusers vs the upstream-PR-track posture for vLLM/sglang/llama.cpp. |

---

## What needs to happen before either image will build

Both Dockerfiles `git clone --branch feat/codec-latent-transport` from forks that **do not yet exist**. Until those branches are created:

```
fatal: Remote branch feat/codec-latent-transport not found in upstream origin
```

### ComfyUI fork — `wdunn001/ComfyUI` `feat/codec-latent-transport`

Fork [`comfyanonymous/ComfyUI`](https://github.com/comfyanonymous/ComfyUI) to `wdunn001/ComfyUI`, branch `feat/codec-latent-transport`. Add:

1. Vendored copy of `packages/python/src/codecai/server/latent_frame.py` from the Codec repo (sibling branch `feat/v0.3-latent-modality`).
2. New endpoint module(s) — likely under `app/codec_endpoints.py` or as a built-in custom node — exposing `/v1/images/generations` and `/v1/videos/generations` accepting `stream_format` + `modality`. Hook into ComfyUI's VAE encode path: tap the latent tensor between sampler and VAE decode → call `LatentStreamEncoder.frame(...)`.
3. `codec_compression.py` overlay paralleling the text-side sglang/vllm pattern: msgpack + protobuf streaming, optional zstd with the per-(latent_space, format, pipeline) dict slot.
4. Set `Codec-Latent-Map` + `Codec-Zstd-Dict` response headers per Codec spec v0.3.

### diffusers fork — `wdunn001/diffusers` `feat/codec-latent-transport`

Fork [`huggingface/diffusers`](https://github.com/huggingface/diffusers) to `wdunn001/diffusers`, branch `feat/codec-latent-transport`. Add:

1. New `examples/codec_server/` package — FastAPI app loading a diffusers `StableDiffusionPipeline` (or video equivalent), exposing the same endpoints as ComfyUI's fork.
2. Vendored `latent_frame.py` (same as above).
3. Entry point `python -m codec_server` — what the Dockerfile's entrypoint invokes.

The Dockerfile's `pip install -e /opt/codec/diffusers` makes the fork importable; codec_server uses VAE encode hooks from inside the diffusers `AutoencoderKL` to capture the latent tensor without re-running the encoder.

---

## codec-supervisor side (this repo) — what's still TODO

| Task | Where |
|---|---|
| Add `comfyui` and `diffusers` `Backend` implementations to `codec_supervisor/backend.py` | Needed for the supervisor's hot-swap + `/health` proxy to work against the new backends. Mirror the existing `sglang` / `vllm` / `llamacpp` backends. |
| Update `codec_supervisor/proxy.py` for latent paths | Latent endpoints (`/v1/images/generations`, `/v1/videos/generations`) need to route through to the backend the same way text endpoints do today. |
| Latent-space-map upload admin endpoint | Mirror the existing model-upload endpoint. The supervisor should accept a latent-space-map JSON, validate it via the schema in the Codec repo, and pin it for the loaded backend. |
| Smoke tests | Extend `tests/` with a `test_latent_smoke.py` that exercises `/v1/images/generations` against a tiny SD checkpoint, asserts the response Content-Type and frame structure. |

---

## Don't touch on this branch

- `Dockerfile.metamcp` — pending changes are from another branch (Node 24 / V8 13 regex bump).
- `scripts/deploy-codec-metamcp-lab.sh` — also belongs to that other branch.

These were intentionally excluded from this branch's commits.
