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
    """vLLM OpenAI-compatible server. Stub for future use."""

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


BACKENDS: dict[str, type] = {
    "sglang": SglangBackend,
    "tgi": TgiBackend,
    "vllm": VllmBackend,
}


def get_backend(name: str) -> Backend:
    cls = BACKENDS.get(name)
    if cls is None:
        raise ValueError(f"unknown backend {name!r}; choose from {list(BACKENDS)}")
    return cls()
