"""Backend abstraction.

A Backend defines how to launch and health-check an inference server. The
supervisor invokes ``command(...)`` to build the argv for the child process,
then polls ``health_path`` on the chosen port until the server is ready.

Adding a new backend (TGI, vLLM, llama-server, etc.) is a single class.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Backend(Protocol):
    name: str

    def command(
        self,
        model_path: str,
        host: str,
        port: int,
        extra_args: list[str],
    ) -> list[str]:
        """Argv to launch this backend serving ``model_path`` on host:port."""

    @property
    def health_path(self) -> str:
        """HTTP path that returns 200 when the backend is ready to serve."""


class SglangBackend:
    name = "sglang"
    health_path = "/health"

    def command(
        self,
        model_path: str,
        host: str,
        port: int,
        extra_args: list[str],
    ) -> list[str]:
        return [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_path,
            "--host",
            host,
            "--port",
            str(port),
            *extra_args,
        ]


class TgiBackend:
    """Hugging Face text-generation-inference. Stub for future use."""

    name = "tgi"
    health_path = "/health"

    def command(
        self,
        model_path: str,
        host: str,
        port: int,
        extra_args: list[str],
    ) -> list[str]:
        return [
            "text-generation-launcher",
            "--model-id",
            model_path,
            "--hostname",
            host,
            "--port",
            str(port),
            *extra_args,
        ]


class VllmBackend:
    """vLLM OpenAI-compatible server."""

    name = "vllm"
    health_path = "/health"

    def command(
        self,
        model_path: str,
        host: str,
        port: int,
        extra_args: list[str],
    ) -> list[str]:
        return [
            "python3",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_path,
            "--host",
            host,
            "--port",
            str(port),
            *extra_args,
        ]


class LlamaCppBackend:
    """llama.cpp `llama-server` (the standalone HTTP server).

    Accepts two kinds of model spec:
      - Local path to a .gguf file → ``--model <path>``
      - Hugging Face ``<owner>/<repo>:<filename-glob>`` → ``-hf <spec>``
        (e.g. ``Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M``)
    """

    name = "llamacpp"
    health_path = "/health"

    def command(
        self,
        model_path: str,
        host: str,
        port: int,
        extra_args: list[str],
    ) -> list[str]:
        argv: list[str] = ["llama-server", "--host", host, "--port", str(port)]
        if model_path:
            # HF id heuristic: contains "/" and ":" and isn't an absolute path.
            looks_like_hf = (
                "/" in model_path
                and ":" in model_path
                and not model_path.startswith("/")
                and not model_path.startswith(".")
            )
            if looks_like_hf:
                argv += ["-hf", model_path]
            else:
                argv += ["--model", model_path]
        argv += list(extra_args)
        return argv


class DiffusersBackend:
    """HuggingFace diffusers reference codec_server (latent modality, v0.3+).

    Spawns ``python -m codec_server`` from the wdunn001/diffusers fork's
    ``examples/codec_server/`` package. The CLI reads its model + latent
    space + bind addr from a mix of ``CODEC_*`` env vars and the flags
    we pass below.

    Model spec is a HuggingFace diffusers repo id (e.g.
    ``stabilityai/stable-diffusion-2-1-base``). The latent space the
    server's bytes resolve against is named separately via the
    ``CODEC_INITIAL_LATENT_SPACE`` env var (Dockerfile.diffusers default
    is ``stabilityai/sd-vae-ft-mse``).
    """

    name = "diffusers"
    health_path = "/health"

    def command(
        self,
        model_path: str,
        host: str,
        port: int,
        extra_args: list[str],
    ) -> list[str]:
        # python3 (not python) — the slim CUDA base image only ships
        # python3 in PATH; the unversioned `python` symlink isn't
        # installed by python3-minimal in debian.
        argv: list[str] = [
            "python3",
            "-m",
            "codec_server",
            "--host",
            host,
            "--port",
            str(port),
        ]
        if model_path:
            argv += ["--model", model_path]
        argv += list(extra_args)
        return argv


class ComfyUIBackend:
    """ComfyUI fork at wdunn001/ComfyUI feat/codec-latent-transport.

    ComfyUI's main.py is the entry point; the codec patch adds
    ``/codec/*`` latent-stream endpoints alongside the standard ones.
    Model loading is workflow-driven, not CLI-driven, so the
    ``model_path`` argument is informational here — the active model is
    whichever the loaded workflow requests.
    """

    name = "comfyui"
    health_path = "/system_stats"

    def command(
        self,
        model_path: str,
        host: str,
        port: int,
        extra_args: list[str],
    ) -> list[str]:
        argv: list[str] = [
            "python3",  # slim CUDA base ships python3 only, no `python` symlink
            "/opt/codec/comfyui/main.py",
            "--listen",
            host,
            "--port",
            str(port),
        ]
        argv += list(extra_args)
        return argv


BACKENDS: dict[str, type] = {
    "sglang": SglangBackend,
    "tgi": TgiBackend,
    "vllm": VllmBackend,
    "llamacpp": LlamaCppBackend,
    "diffusers": DiffusersBackend,
    "comfyui": ComfyUIBackend,
}


def get_backend(name: str) -> Backend:
    cls = BACKENDS.get(name)
    if cls is None:
        raise ValueError(f"unknown backend {name!r}; choose from {list(BACKENDS)}")
    return cls()
