"""Subprocess lifecycle for the backend inference server.

The ProcessManager owns at most one child process at a time. Starting a new
model implicitly stops the previous child. All transitions are serialized
under an asyncio.Lock so concurrent /admin requests can't race.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
from datetime import datetime, timezone

import httpx

from .backend import Backend

logger = logging.getLogger(__name__)


class BackendCrashed(RuntimeError):
    pass


class BackendNotReady(RuntimeError):
    pass


class ProcessManager:
    def __init__(
        self,
        backend: Backend,
        host: str,
        port: int,
        startup_timeout_s: int = 1800,
        shutdown_grace_s: int = 30,
    ) -> None:
        self.backend = backend
        self.host = host
        self.port = port
        self.startup_timeout_s = startup_timeout_s
        self.shutdown_grace_s = shutdown_grace_s

        self._process: subprocess.Popen | None = None
        self._current_model: str | None = None
        self._started_at: datetime | None = None
        self._lock = asyncio.Lock()

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def current_model(self) -> str | None:
        return self._current_model if self.is_running else None

    @property
    def started_at(self) -> datetime | None:
        return self._started_at if self.is_running else None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    async def start(self, model_path: str, extra_args: list[str]) -> None:
        async with self._lock:
            if self._process is not None:
                await self._stop_unlocked()

            cmd = self.backend.command(model_path, self.host, self.port, extra_args)
            logger.info("starting backend: %s", " ".join(cmd))

            # Use a new process group so we can signal the whole tree on stop.
            popen_kwargs: dict = {}
            if os.name == "posix":
                popen_kwargs["start_new_session"] = True

            self._process = subprocess.Popen(cmd, **popen_kwargs)
            self._current_model = model_path
            self._started_at = datetime.now(timezone.utc)

            try:
                await self._wait_for_healthy()
            except Exception:
                await self._stop_unlocked()
                raise

    async def stop(self) -> None:
        async with self._lock:
            await self._stop_unlocked()

    async def _stop_unlocked(self) -> None:
        if self._process is None:
            return

        proc = self._process
        if proc.poll() is None:
            logger.info("stopping backend (pid %s)", proc.pid)
            try:
                if os.name == "posix":
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                else:
                    proc.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(proc.wait), timeout=self.shutdown_grace_s
                )
            except asyncio.TimeoutError:
                logger.warning("backend ignored SIGTERM, sending SIGKILL")
                try:
                    if os.name == "posix":
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    else:
                        proc.kill()
                except ProcessLookupError:
                    pass
                await asyncio.to_thread(proc.wait)

        self._process = None
        self._current_model = None
        self._started_at = None

    async def _wait_for_healthy(self) -> None:
        url = f"{self.base_url}{self.backend.health_path}"
        deadline = asyncio.get_event_loop().time() + self.startup_timeout_s
        async with httpx.AsyncClient(timeout=2.0) as client:
            while asyncio.get_event_loop().time() < deadline:
                if self._process is None or self._process.poll() is not None:
                    code = self._process.returncode if self._process else "n/a"
                    raise BackendCrashed(f"backend exited during startup (code={code})")
                try:
                    r = await client.get(url)
                    if r.status_code == 200:
                        logger.info("backend healthy at %s", url)
                        return
                except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError):
                    pass
                await asyncio.sleep(1.0)
        raise BackendNotReady(
            f"backend did not become healthy within {self.startup_timeout_s}s"
        )
