"""Cancellation regressions for proxy SSE and Anthropic stream ownership."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import anyio
import httpx
import pytest
from starlette.requests import ClientDisconnect

from litellm.litellm_core_utils.async_stream_cleanup import close_stream
from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)
from litellm.llms.anthropic.experimental_pass_through.messages.streaming_iterator import (
    BaseAnthropicMessagesStreamingIterator,
)
from litellm.llms.custom_httpx.aiohttp_transport import AiohttpResponseStream
from litellm.proxy.common_request_processing import (
    ProxyBaseLLMRequestProcessing,
    create_response,
)


class TrackingStream:
    def __init__(self):
        self.closed = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        return "data: {}\n\n"

    async def aclose(self):
        await anyio.sleep(0)
        self.closed += 1


async def owned_generator(stream):
    try:
        async for chunk in stream:
            yield chunk
    finally:
        await close_stream(stream)


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks", [1, 2])
async def test_closing_combined_generator_closes_prefetched_stream(chunks):
    backend = TrackingStream()
    inner = owned_generator(backend)
    response = await create_response(inner, "text/event-stream", {})
    for _ in range(chunks):
        await response.body_iterator.__anext__()
    with anyio.CancelScope() as scope:
        scope.cancel()
        await response.body_iterator.aclose()
    assert backend.closed == 1
    assert inner.ag_frame is None


@pytest.mark.asyncio
@pytest.mark.parametrize("disconnect_at", ["headers", "body"])
@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
async def test_asgi_disconnect_during_send_closes_stream(disconnect_at, spec_version):
    backend = TrackingStream()
    inner = owned_generator(backend)
    response = await create_response(inner, "text/event-stream", {})
    disconnect = anyio.Event()

    async def send(message):
        target = (
            "http.response.start"
            if disconnect_at == "headers"
            else "http.response.body"
        )
        if message["type"] == target:
            if spec_version == "2.4":
                raise OSError("client disconnected")
            disconnect.set()
            await anyio.sleep_forever()

    async def receive():
        await disconnect.wait()
        return {"type": "http.disconnect"}

    scope = {"type": "http", "asgi": {"spec_version": spec_version}}
    with anyio.fail_after(5):
        if spec_version == "2.4":
            with pytest.raises(ClientDisconnect):
                await response(scope, receive, send)
        else:
            await response(scope, receive, send)
    assert backend.closed == 1
    assert inner.ag_frame is None


@pytest.mark.asyncio
async def test_cancel_before_first_chunk_closes_iterator_and_reraises():
    class WaitingStream(TrackingStream):
        async def __anext__(self):
            raise asyncio.CancelledError

    backend = WaitingStream()
    with pytest.raises(asyncio.CancelledError):
        await create_response(backend, "text/event-stream", {})
    assert backend.closed == 1


@pytest.mark.asyncio
async def test_first_chunk_provider_error_closes_iterator_preserving_response():
    class FailingStream(TrackingStream):
        async def __anext__(self):
            raise ValueError("provider failed")

    backend = FailingStream()
    response = await create_response(backend, "text/event-stream", {})
    assert response.status_code == 500
    assert backend.closed == 1


@pytest.mark.asyncio
async def test_normal_completion_keeps_chunks_and_closes_stream():
    closed = []

    async def stream():
        try:
            yield "data: {}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            closed.append(True)

    response = await create_response(stream(), "text/event-stream", {"x-test": "ok"})
    assert [chunk async for chunk in response.body_iterator] == [
        "data: {}\n\n",
        "data: [DONE]\n\n",
    ]
    assert response.headers["x-test"] == "ok"
    assert closed == [True]


@pytest.mark.asyncio
async def test_anthropic_adapter_closes_backend_after_initial_message():
    # message_start is synthetic: the backend may not have yielded any tokens yet.
    backend = TrackingStream()
    adapter = AnthropicStreamWrapper(backend, "test-model")
    stream = adapter.async_anthropic_sse_wrapper()
    assert b"message_start" in await stream.__anext__()
    with anyio.CancelScope() as scope:
        scope.cancel()
        await stream.aclose()
    await adapter.aclose()
    assert backend.closed == 1


@pytest.mark.asyncio
async def test_anthropic_base_wrapper_closes_input_stream():
    backend = TrackingStream()
    wrapper = BaseAnthropicMessagesStreamingIterator(MagicMock(), {})
    wrapper._handle_streaming_logging = AsyncMock()
    stream = wrapper.async_sse_wrapper(backend)
    await stream.__anext__()
    await stream.aclose()
    assert backend.closed == 1
    wrapper._handle_streaming_logging.assert_not_awaited()


@pytest.mark.asyncio
async def test_proxy_anthropic_sse_wrapper_closes_inner_generator():
    backend = TrackingStream()
    logging = MagicMock()

    async def iterator_hook(**kwargs):
        async for chunk in kwargs["response"]:
            yield chunk

    async def chunk_hook(**kwargs):
        return kwargs["response"]

    logging.async_post_call_streaming_iterator_hook = iterator_hook
    logging.async_post_call_streaming_hook = chunk_hook
    stream = ProxyBaseLLMRequestProcessing.async_sse_data_generator(
        response=backend,
        user_api_key_dict=MagicMock(),
        request_data={},
        proxy_logging_obj=logging,
    )
    await stream.__anext__()
    await stream.aclose()
    assert backend.closed == 1


@pytest.mark.asyncio
async def test_close_supports_sdk_close_and_preserves_cancellation():
    stream = MagicMock(spec=["close"])
    stream.close = AsyncMock()
    with anyio.CancelScope() as scope:
        scope.cancel()
        await close_stream(stream)
    stream.close.assert_awaited_once()
    stream.close.side_effect = RuntimeError("cleanup failed")
    await close_stream(stream)


@pytest.mark.asyncio
async def test_disconnect_reaches_backend_tcp_connection_without_another_token():
    """Exercise real aiohttp/httpx transport closure against a stalled local SSE server."""
    disconnected = asyncio.Event()

    async def handle(reader, writer):
        try:
            await reader.readuntil(b"\r\n\r\n")
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                b"Transfer-Encoding: chunked\r\n\r\n"
                b"a\r\ndata: {}\n\n\r\n"
            )
            await writer.drain()
            await reader.read()
        finally:
            writer.close()
            await writer.wait_closed()
            disconnected.set()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    async with server, aiohttp.ClientSession() as session:
        upstream = await session.get(f"http://127.0.0.1:{port}/")
        http_response = httpx.Response(200, stream=AiohttpResponseStream(upstream))

        async def stream():
            try:
                async for chunk in http_response.aiter_text():
                    yield chunk
            finally:
                await close_stream(http_response)

        inner = stream()
        response = await create_response(inner, "text/event-stream", {})

        async def send(message):
            if message["type"] == "http.response.body":
                raise OSError("client disconnected during send")

        async def receive():
            await anyio.sleep_forever()

        with anyio.fail_after(5):
            with pytest.raises(ClientDisconnect):
                await response(
                    {"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send
                )
            await disconnected.wait()
        assert upstream.closed
        assert http_response.is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
async def test_disconnect_while_waiting_for_next_token_closes_backend(spec_version):
    waiting = anyio.Event()

    class StalledStream(TrackingStream):
        def __init__(self):
            super().__init__()
            self.first = True

        async def __anext__(self):
            if self.first:
                self.first = False
                return "data: {}\n\n"
            waiting.set()
            await anyio.sleep_forever()

    backend = StalledStream()
    inner = owned_generator(backend)
    response = await create_response(inner, "text/event-stream", {})

    async def receive():
        await waiting.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        pass

    with anyio.fail_after(5):
        await response(
            {"type": "http", "asgi": {"spec_version": spec_version}}, receive, send
        )
    assert backend.closed == 1


@pytest.mark.asyncio
async def test_full_anthropic_proxy_chain_closes_after_synthetic_first_event():
    backend = TrackingStream()
    adapter = AnthropicStreamWrapper(backend, "test-model")
    anthropic_stream = adapter.async_anthropic_sse_wrapper()
    logging = MagicMock()

    async def iterator_hook(**kwargs):
        async for chunk in kwargs["response"]:
            yield chunk

    async def chunk_hook(**kwargs):
        return kwargs["response"]

    logging.async_post_call_streaming_iterator_hook = iterator_hook
    logging.async_post_call_streaming_hook = chunk_hook
    proxy_stream = ProxyBaseLLMRequestProcessing.async_sse_data_generator(
        response=anthropic_stream,
        user_api_key_dict=MagicMock(),
        request_data={},
        proxy_logging_obj=logging,
    )
    response = await create_response(proxy_stream, "text/event-stream", {})

    async def send(message):
        if message["type"] == "http.response.body":
            raise OSError("client disconnected")

    async def receive():
        await anyio.sleep_forever()

    with anyio.fail_after(5), pytest.raises(ClientDisconnect):
        await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
    assert backend.closed == 1
    assert anthropic_stream.ag_frame is None
    assert proxy_stream.ag_frame is None
