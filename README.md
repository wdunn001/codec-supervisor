# codec-supervisor

A backend-agnostic supervisor / control plane for inference servers. Wraps an OpenAI-compatible engine (sglang today; TGI/vLLM stubs in place) with:

- **Single port** — clients hit one HTTP endpoint and the supervisor proxies `/v1/*` to the live backend.
- **Model registry** — list, pull from Hugging Face, tarball-upload, delete models on a `/models` volume.
- **Hot swap** — `POST /admin/load` restarts the backend with a different model. No container restart.
- **Easily deployable** — one Docker image bundles sglang + the [Codec PRs](#codec-patches) + the supervisor. `docker run` and you have a working Codec inference instance.

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

## Adding a new backend

Implement the [`Backend`](codec_supervisor/backend.py) protocol — three things: a `name`, a `health_path`, and a `command(model_path, host, port, extra_args)` that returns argv. Register it in `BACKENDS`. That's it. Stubs for TGI and vLLM are already there.

## License

[BUSL-1.1](LICENSE) for codec-supervisor itself. The bundled sglang remains under Apache-2.0; the Codec patches retain whatever license they ship under upstream.
