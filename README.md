# codec-supervisor

A backend-agnostic supervisor / control plane for inference servers. Wraps an OpenAI-compatible engine (sglang today; TGI/vLLM stubs in place) with:

- **Single port** — clients hit one HTTP endpoint and the supervisor proxies `/v1/*` to the live backend.
- **Model registry** — list, pull from Hugging Face, tarball-upload, delete models on a `/models` volume.
- **Hot swap** — `POST /admin/load` restarts the backend with a different model. No container restart.
- **Easily deployable** — one Docker image bundles the engine + the [Codec PRs](#codec-patches) + the supervisor. `docker run` and you have a working Codec inference instance.

> **v0.5 shipped** (2026-05-18). Wire-additive over v0.4. New
> opt-in surfaces: discoverable Zstandard dictionaries at
> `.well-known/codec/dicts/<sha256>.zstd` (hash-pinned; release-
> checklist §1.7 gates dict-bake + the
> `/opt/codec/check-dict-availability.sh` probe in every engine
> image); bolt-on tool dispatcher contract; content-aware
> compression picker rewrite. v0.4 safety-policy surface from the
> prior release stays: sanitized policy descriptors at
> `.well-known/codec/policies/<id>.json`, operator-side authoring
> via `/admin/policies/`, logits-space enforcement + classifier
> registry. v0.5 cohort retired TGI; supported engines now sglang
> + vLLM + llama.cpp + ComfyUI + diffusers.
>
> Every v0.4+ capability is **opt-on, two-stage** (per the
> [spec](https://github.com/wdunn001/Codec/blob/main/spec/versions/v0.4.md#capabilities-are-opt-on-at-the-server-two-stage)):
> default OFF. Per-capability env vars enable; `*_REQUIRED=1` flips
> on enforcement. A supervisor with no v0.4+ capabilities enabled
> serves byte-equivalent v0.3 wire — no v0.4 negotiation headers,
> no 426 for version reasons. Operators turn this on deliberately
> for public / multi-tenant / mandatory-policy deployments.

## Image catalog (current v0.5.0)

This repo's [`release.yml` workflow](.github/workflows/release.yml) builds and pushes the following images on every `v*` git tag. Tags emitted per image: `:vX.Y.Z` (semver, immutable) · `:latest` (moves with each release) · `:sha-<git7>` (immutable git-tree pin for hotfixes).

| Image                                                                          | Current tag | Engine fork                                                                                       | Modality              |
|--------------------------------------------------------------------------------|:-----------:|---------------------------------------------------------------------------------------------------|-----------------------|
| [`wdunn001/codec-sglang`](https://hub.docker.com/r/wdunn001/codec-sglang)      | **v0.5.0**  | sglang + Codec patches (token-native binary transport + server-side ToolWatcher + bolt-on dispatcher) | text-tokens           |
| [`wdunn001/codec-vllm`](https://hub.docker.com/r/wdunn001/codec-vllm)          | **v0.5.0**  | vLLM + Codec patches (token-native binary transport on `/v1/completions` and `/v1/chat/completions`)  | text-tokens           |
| [`wdunn001/codec-llamacpp`](https://hub.docker.com/r/wdunn001/codec-llamacpp)  | **v0.5.0**  | llama.cpp + Codec patches on `llama-server` (covers Ollama too)                                       | text-tokens           |
| [`wdunn001/codec-metamcp`](https://hub.docker.com/r/wdunn001/codec-metamcp)    | **v0.3.2**  | [`wdunn001/metamcp`](https://github.com/wdunn001/metamcp) `feat/codec-binary-transport`               | MCP gateway (with leaf-mode bypass) |
| [`wdunn001/codec-time-leaf`](https://hub.docker.com/r/wdunn001/codec-time-leaf) | **v0.3.2**  | Reference Codec-aware MCP server ([`@codecai/codec-time-leaf`](https://www.npmjs.com/package/@codecai/codec-time-leaf)) | MCP tool (v0.3) |
| [`wdunn001/codec-comfyui`](https://hub.docker.com/r/wdunn001/codec-comfyui)    | **v0.5.0**  | [`wdunn001/ComfyUI`](https://github.com/wdunn001/ComfyUI) `feat/codec-latent-transport`               | latents (v0.3+)       |
| [`wdunn001/codec-diffusers`](https://hub.docker.com/r/wdunn001/codec-diffusers) | **v0.5.0**  | [`wdunn001/diffusers`](https://github.com/wdunn001/diffusers) `feat/codec-latent-transport`           | latents (v0.3+) |

`wdunn001/codec-tgi` was retired in v0.5 — TGI is treated as a dead project; the cohort is now five engines (sglang + vLLM + llama.cpp + ComfyUI + diffusers).

The v0.5.0 release (2026-05-18) added `/opt/codec/dicts/qwen2.5-synth-{msgpack,protobuf}-v1.dict` bake-in to every text-engine image plus `/opt/codec/check-dict-availability.sh` — the release-checklist §1.7 sub-gate 2 runtime probe. Each image is also dep-verified for `import brotli, zstandard, msgpack` before push (release-checklist §1.9). metamcp + time-leaf stay on v0.3.2; no codec wire changes for them in v0.5.

To cut a release: `git tag v0.X.Y && git push origin v0.X.Y`. The workflow runs `docker buildx` against each Dockerfile in parallel and pushes once green. Secrets required: `DOCKERHUB_USERNAME` + `DOCKERHUB_TOKEN`.

## Why this exists

Codec is a [token-native binary transport](https://codecai.net) for AI APIs — token IDs as 4-byte frames instead of 50–100 bytes of UTF-8/JSON per token. The Codec wire format is going through upstream as two PRs against [sgl-project/sglang](https://github.com/sgl-project/sglang):

- [#24483](https://github.com/sgl-project/sglang/pull/24483) — token-native binary transport for completions streaming
- [#24557](https://github.com/sgl-project/sglang/pull/24557) — server-side ToolWatcher

While the PRs are in review, this repo provides a ready-to-run image that overlays them on the official sglang base. The supervisor is a separate, backend-agnostic concern that ships independently of those PRs and stays useful long after they merge.

## Quick start

### Docker

```bash
git clone https://github.com/wdunn001/codec-supervisor.git
cd codec-supervisor
docker compose up -d
```

The supervisor listens on `http://localhost:8080`. First boot downloads `Qwen/Qwen2.5-0.5B-Instruct` (override with `CODEC_INITIAL_MODEL`).

```bash
# OpenAI-compatible: hits the backend through the proxy
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "x", "prompt": "Hello", "max_tokens": 20}'

# Codec wire format (msgpack frames of token IDs)
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "x", "prompt": "Hello", "max_tokens": 20, "stream": true, "stream_format": "msgpack"}' \
  -o codec.bin
```

### Without Docker (dev)

```bash
pip install -e .
codec-supervisor --backend sglang --models-dir ./models --initial-model Qwen/Qwen2.5-0.5B-Instruct
```

## Running your own models

Three ways, progressively more "self-serve":

### 1. Override `CODEC_INITIAL_MODEL` at `docker run` time

Any HF repo id (or local path inside the container) — supervisor downloads + boots on first start, caches in the volume.

```bash
docker run --gpus all -p 8080:8080 \
  -e CODEC_INITIAL_MODEL=meta-llama/Llama-3.1-8B-Instruct \
  -e HF_TOKEN=hf_xxxxx \
  -v hf-cache:/root/.cache/huggingface \
  wdunn001/codec-sglang:latest
```

`HF_TOKEN` is only needed for gated models.

### 2. Mount a local model directory

For checkpoints / fine-tunes you don't want to upload to HF — just bind-mount and point `CODEC_INITIAL_MODEL` at it:

```bash
docker run --gpus all -p 8080:8080 \
  -e CODEC_INITIAL_MODEL=/models/my-finetune \
  -v /path/to/my-finetune:/models/my-finetune:ro \
  wdunn001/codec-sglang:latest
```

The directory is read-only inside the container — no risk of the supervisor mutating your weights.

### 3. Hot-swap via the admin API after boot

This is what the supervisor adds on top of stock sglang. The container stays up, the model swaps:

```bash
# Pull a model into the registry while the container is running
curl -X POST http://localhost:8080/admin/models/pull \
  -H "Content-Type: application/json" \
  -d '{"repo_id": "Qwen/Qwen2.5-7B-Instruct"}'

# Or upload a tarball of a local fine-tune
tar -cf my-finetune.tar -C ./checkpoints/my-finetune .
curl -X POST "http://localhost:8080/admin/models/upload?name=my-finetune" \
  -F "file=@my-finetune.tar"

# Hot-swap to it (kills running sglang, boots a new one with the new model)
curl -X POST http://localhost:8080/admin/load \
  -H "Content-Type: application/json" \
  -d '{"name": "my-finetune"}'

# Or pass an HF id directly without staging it first
curl -X POST http://localhost:8080/admin/load \
  -H "Content-Type: application/json" \
  -d '{"name": "Qwen/Qwen2.5-7B-Instruct", "allow_remote": true}'
```

Pass `CODEC_BACKEND_ARGS` to tune sglang per-model (`--tp 2 --quantization fp8 --mem-fraction-static 0.9`, etc.). For per-load tuning, include `extra_args` in the `/admin/load` body — overrides the supervisor's default for that one model.

> **Hot-swap caveat**: there's a few-second gap during the swap (terminate child → fork new → poll `/health`). For zero-downtime multi-model serving, run multiple containers behind a router. The supervisor is single-model-per-container by design.

## Admin API

| Method | Path | Body | Notes |
|---|---|---|---|
| `GET`  | `/health`                | — | Supervisor liveness, plus `backend_running` flag |
| `GET`  | `/admin/status`          | — | Backend, current model, started_at |
| `GET`  | `/admin/models`          | — | List models in the `/models` volume |
| `POST` | `/admin/models/pull`     | `{"repo_id": "Qwen/Qwen2.5-7B-Instruct"}` | `snapshot_download` into `/models` |
| `POST` | `/admin/models/upload`   | multipart `file=<tarball>` + `?name=<name>` | Extract a `.tar` of a model dir |
| `DELETE` | `/admin/models/{name}` | — | Refuses if the model is currently loaded |
| `POST` | `/admin/load`            | `{"name": "qwen2.5-7b"}` or `{"name": "Qwen/Qwen2.5-7B-Instruct", "allow_remote": true}` | Restart backend with this model |
| `POST` | `/admin/stop`            | — | Stop backend; supervisor stays up |
| `GET`  | `/admin/policies`        | — | List safety policies on disk (id, version, hash, summary) — shipped in v0.4 |
| `POST` | `/admin/policies`        | internal-policy JSON | Save a new policy revision (re-sanitize + re-hash) — **v0.4, in flight** |
| `GET`  | `/admin/policies/{id}`   | — | Internal policy with full banned-id list / classifier thresholds (operator-only) — **v0.4** |
| `GET`  | `/admin/policies/{id}/descriptor` | — | The sanitized publishable descriptor — same bytes served at `.well-known/codec/policies/<id>.json` — **v0.4** |
| `*`    | `/{anything}`            | (proxied) | Forwarded to the backend (incl. SSE/msgpack streams) |

### Examples

```bash
# 1. Pull a model into the registry
curl -X POST http://localhost:8080/admin/models/pull \
  -H "Content-Type: application/json" \
  -d '{"repo_id": "Qwen/Qwen2.5-7B-Instruct"}'

# 2. Upload a local fine-tune (tarball of the model dir)
tar -cf my-finetune.tar -C ./checkpoints/my-finetune .
curl -X POST "http://localhost:8080/admin/models/upload?name=my-finetune" \
  -F "file=@my-finetune.tar"

# 3. Hot-swap to it
curl -X POST http://localhost:8080/admin/load \
  -H "Content-Type: application/json" \
  -d '{"name": "my-finetune"}'

# 4. Verify
curl http://localhost:8080/admin/status
```

## Configuration

Everything has a `--flag` and a `CODEC_*` env var. See [`.env.example`](.env.example) for the full list. Most-used:

| Var | Default | Notes |
|---|---|---|
| `CODEC_PORT` | `8080` | Public port |
| `CODEC_BACKEND` | `sglang` | `sglang`, `tgi`, `vllm` (latter two are stubs) |
| `CODEC_BACKEND_PORT` | `30000` | Internal-only |
| `CODEC_MODELS_DIR` | `/models` | Volume-mounted in the container |
| `CODEC_INITIAL_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | HF id, local name, or absolute path |
| `CODEC_BACKEND_ARGS` | `--mem-fraction-static 0.85 --attention-backend triton` | Verbatim to sglang |
| `CODEC_STARTUP_TIMEOUT_S` | `1800` | How long to wait for `/health` after launch |
| `HF_TOKEN` | _(unset)_ | Required for gated models |

## Architecture

```
                ┌──────────────────────────────────────┐
                │  codec-supervisor (FastAPI, :8080)   │
                │                                      │
                │  /admin/*  ◀── you (control plane)   │
                │  /v1/*     ──▶ proxy to backend ──┐  │
                │                                   │  │
                │  ProcessManager ──fork/exec──┐    │  │
                └──────────────────────────────┼────┼──┘
                                               ▼    ▼
                              ┌──────────────────────────────┐
                              │  sglang.launch_server :30000 │
                              │  (codec patches applied)     │
                              └──────────────────────────────┘
                                               │
                              ┌────────────────┴───────────┐
                              │   /models (named volume)   │
                              │   ~/.cache/huggingface     │
                              └────────────────────────────┘
```

The supervisor owns at most one backend child. Hot-swap = `terminate(child)` → `Popen(new_cmd)` → poll `/health` until ready. There's a few-second gap during the swap; if you need zero-downtime multi-model serving, run two containers behind a router.

## Codec patches

The Docker image overlays [`wdunn001/sglang`](https://github.com/wdunn001/sglang) `feat/codec-server-side-agent` on top of `lmsysorg/sglang:latest` via `pip install --no-deps -e .`. That branch contains both PRs (#24557 is stacked on #24483). Pure-Python overlay — no CUDA rebuild, kernels stay intact.

When the PRs merge upstream, the Dockerfile drops the overlay and just `pip install codec-supervisor` against `lmsysorg/sglang:<release>`.

### Latent modality (image / video, v0.3+)

`Dockerfile.comfyui` and `Dockerfile.diffusers` are siblings of the text-engine Dockerfiles in this repo, sized for the latent-stream surface defined in [Codec spec v0.3 §Latent Modality](https://github.com/wdunn001/Codec/blob/main/spec/PROTOCOL.md). Both image-gen images share the same `.well-known/codec/latents/<id>.json` registry and the same zstd-dict pool, just with different inference engines underneath.

| Image | Source | Posture |
|---|---|---|
| `codec/comfyui:dev`   | [`wdunn001/ComfyUI`](https://github.com/wdunn001/ComfyUI) `feat/codec-latent-transport` | **Fork-only — no upstream PR planned.** ComfyUI's plugin/custom-node architecture would let us ship the codec endpoints as a custom node, but the latent-frame emitter and zstd-dict overlay touch enough of the request loop that maintaining a downstream fork is cleaner than threading hooks. |
| `codec/diffusers:dev` | [`wdunn001/diffusers`](https://github.com/wdunn001/diffusers) `feat/codec-latent-transport` | **Fork-only — no upstream PR planned.** diffusers is a library; our fork adds an `examples/codec_server/` FastAPI wrapper. This image doubles as the **bench/golden perceptual-conformance reference**: the pinned `torch` + `diffusers` versions here define the SSIM / PSNR / LPIPS contract every latent bench cell resolves against. |

Both images are gated behind compose profiles so a default `docker compose up` doesn't pull them (the image layers are heavy: CUDA + torch + diffusers ~ 10 GB).

```bash
docker compose --profile latents up -d           # both image-gen services
docker compose --profile comfyui up -d codec-comfyui
docker compose --profile diffusers up -d codec-diffusers
```

The text-engine forks (`vllm`, `sglang`, `llama.cpp`) remain upstream-PR-track; only the latent forks are explicitly fork-only.

### Safety enforcement (v0.4 — work in progress)

The supervisor implements the operator side of the Codec
[safety-policy negotiation](https://github.com/wdunn001/Codec/blob/main/spec/versions/v0.4.md#safety-policy-negotiation):
authors **internal** policy configs and publishes **sanitized**
descriptors; the descriptor is what clients fetch and pin. v0.4 is
in flight pending the release-checklist gates (validation, benches,
docs, READMEs, website) — see
[Codec/docs/RELEASE_CHECKLIST.md](https://github.com/wdunn001/Codec/blob/main/docs/RELEASE_CHECKLIST.md).

What ships:

- **Layered architecture** mirroring the spec — prefilter (client),
  logits processor (server, token-space), streaming classifier
  (server, embedding/text), and a per-category action policy
  (`stop` / `redact` / `regenerate` / `flag`).
- **Internal-vs-published split** (`codec_supervisor/safety.py`).
  Internal Pydantic model carries banned-token-ID lists, regex
  patterns, multi-token patterns, classifier thresholds; the
  `sanitize()` step strips all of those and emits only categories,
  action types, classifier family, and rules-summary counts. Operator
  configs never cross the wire.
- **Logits-space enforcement** (`codec_supervisor/safety_logits.py`)
  — banned-token-ID masking compatible with the vLLM
  `LogitsProcessor` interface; pure token-space.
- **Multi-token banned-pattern matcher**
  (`codec_supervisor/safety_token_matcher.py` +
  `safety_aho_corasick.py`) — Aho-Corasick automaton over int
  alphabets so multi-token banned strings (slurs, secret-shaped
  patterns) match during generation without per-step regex.
- **Delay-k streaming decisioning**
  (`codec_supervisor/safety_streaming.py`) following the Streaming
  Content Monitor (arxiv 2506.09996) pattern — tolerate k
  uncertain frames before forcing the classifier's hand.
- **Pluggable classifier registry**
  (`codec_supervisor/safety_classifier.py` +
  `codec_supervisor/safety_classifiers/`) with three v1
  implementations: **Llama Guard 3 1B** (14-category taxonomy),
  **ShieldGemma 2B** (4-category), **embedding-space**
  (engine-hidden-state, no detok). Generator-DI on every
  classifier — constructors accept an injectable callable so tests
  run without weights.
- **Adversarial defenses**
  (`codec_supervisor/safety_adversarial.py`) — TokenBreak,
  EchoGram, and glitch-token (undertrained-slot) detection helpers
  that complement banned-id-list enforcement.
- **Admin REST surface** (`codec_supervisor/admin_safety.py`) —
  mounted at `/admin/policies/*`; see the admin-API table above.
- **Admin React app** (`admin/`, Vite) for policy authoring,
  classifier configuration, and live test-bench. Mounted at
  `/admin/policies/`.
- **Optional training pipeline**
  (`codec_supervisor/training/bootstrap_corpus.py`) — distill
  text-space classifier judgments into the embedding-space
  classifier without exposing labels back to the wire.

What hosts and clients see in v0.4:

- Server adds `safety_policy_id` + `safety_policy_hash` to its
  `READY` frame and publishes `.well-known/codec/policies/<id>.json`
  (plus the content-addressed `sha256/<hex>.json` sibling) from the
  sanitized descriptor.
- Clients verify the hash, learn the *shape* of enforcement
  (categories + actions + classifier family), and never see the
  operator's internal banned-id lists or thresholds — that's the
  disclosure-boundary contract.
- Streaming completions emit `finish_reason: "policy_violation"`
  when an action of `stop` fires; `redact` / `regenerate` are
  invisible on the wire by design.

**Optional install extras** (Python):

```bash
pip install -e '.[safety]'              # base safety stack (logits, matcher, streaming)
pip install -e '.[safety-llamaguard]'   # + Llama Guard 3 1B classifier
pip install -e '.[safety-shieldgemma]'  # + ShieldGemma 2B classifier
pip install -e '.[safety-embedding]'    # + engine-hidden-state classifier
```

Treat all of the above as work-in-progress until v0.4 cuts. The
matching client-side package is
[`@codecai/web-safety`](https://github.com/wdunn001/Codec/tree/main/packages/web-safety)
(prefilter + browser classifier registry). The CLI surface for
authoring, sanitizing, and publishing descriptors lives in
[`@codecai/maps-cli`](https://www.npmjs.com/package/@codecai/maps-cli)
under the `policies-*` subcommands.

## Adding a new backend

Implement the [`Backend`](codec_supervisor/backend.py) protocol — three things: a `name`, a `health_path`, and a `command(model_path, host, port, extra_args)` that returns argv. Register it in `BACKENDS`. That's it. Stubs for TGI and vLLM are already there.

## License

[BUSL-1.1](LICENSE) for codec-supervisor itself. The bundled sglang remains under Apache-2.0; the Codec patches retain whatever license they ship under upstream.
