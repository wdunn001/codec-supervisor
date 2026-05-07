"""Streaming reverse proxy from supervisor port to backend port.

The supervisor sits in front of the backend so clients hit a single port. All
non-admin requests get forwarded as-is, with response bytes streamed through
unchanged — preserving SSE, msgpack frames, and any other Codec wire format.
"""

from __future__ import annotations

import logging

import httpx
from fastapi import Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

# Hop-by-hop headers — must not be forwarded per RFC 7230.
HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


async def proxy_request(request: Request, backend_url: str) -> StreamingResponse:
    target = f"{backend_url}{request.url.path}"
    if request.url.query:
        target += f"?{request.url.query}"

    fwd_headers = {
        k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP
    }

    body = await request.body()

    client = httpx.AsyncClient(timeout=None)
    upstream_req = client.build_request(
        request.method,
        target,
        headers=fwd_headers,
        content=body,
    )

    try:
        upstream = await client.send(upstream_req, stream=True)
    except httpx.ConnectError as e:
        await client.aclose()
        logger.warning("proxy connect error to %s: %s", target, e)
        from fastapi import HTTPException

        raise HTTPException(status_code=502, detail=f"backend unreachable: {e}") from e

    response_headers = {
        k: v for k, v in upstream.headers.items() if k.lower() not in HOP_BY_HOP
    }
    # Expose response timing to cross-origin pages (e.g. the demo-web bench
    # at localhost:5173) — required for `PerformanceResourceTiming.encodedBodySize`
    # to report real wire bytes instead of 0. Harmless when same-origin.
    response_headers.setdefault("timing-allow-origin", "*")

    async def gen():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        gen(),
        status_code=upstream.status_code,
        headers=response_headers,
        media_type=upstream.headers.get("content-type"),
    )
