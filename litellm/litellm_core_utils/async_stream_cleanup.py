"""Release request-owned streams without closing shared HTTP clients."""

import asyncio
import inspect
from typing import Any

import anyio

from litellm._logging import verbose_logger


async def close_stream(stream: Any) -> None:
    """Propagate closure through iterator wrappers, even during ASGI cancellation."""
    with anyio.CancelScope(shield=True):
        try:
            close = getattr(stream, "aclose", None) or getattr(stream, "close", None)
            if close is not None:
                result = close()
                if inspect.isawaitable(result):
                    await result
        except (Exception, asyncio.CancelledError) as exc:
            # Cleanup must not replace the original cancellation/provider error.
            verbose_logger.debug("Error closing response stream: %s", exc)
