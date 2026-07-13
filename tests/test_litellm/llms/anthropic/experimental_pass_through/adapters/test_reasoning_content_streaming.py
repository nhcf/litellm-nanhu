"""
Tests for AnthropicStreamWrapper handling ``reasoning_content`` streaming deltas.

Providers like DeepSeek return reasoning/thinking text via the
``delta.reasoning_content`` field (rather than ``delta.thinking_blocks``).
These tests verify that the wrapper maps such deltas to proper Anthropic
``thinking`` content blocks (type, index, block-start/stop, signature field).
"""

import os
import sys
from typing import List

import pytest

sys.path.insert(0, os.path.abspath("../../../../../.."))

from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices, Usage


class MockCompletionStream:
    def __init__(self, responses: List[ModelResponseStream]):
        self.responses = responses
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.responses):
            raise StopIteration
        response = self.responses[self.index]
        self.index += 1
        return response

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.responses):
            raise StopAsyncIteration
        response = self.responses[self.index]
        self.index += 1
        return response


def _make_reasoning_chunk(text: str) -> ModelResponseStream:
    return ModelResponseStream(
        choices=[
            StreamingChoices(
                delta=Delta(reasoning_content=text),
                index=0,
                finish_reason=None,
            )
        ],
    )


def _make_text_chunk(text: str) -> ModelResponseStream:
    return ModelResponseStream(
        choices=[
            StreamingChoices(delta=Delta(content=text), index=0, finish_reason=None)
        ],
    )


def _make_stop_chunk() -> ModelResponseStream:
    return ModelResponseStream(
        choices=[
            StreamingChoices(
                delta=Delta(content="", stop_reason="stop"),
                index=0,
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=88, completion_tokens=172, total_tokens=260),
    )


def _chunks_for(stream: AnthropicStreamWrapper) -> list:
    return list(stream)


def _types(chunks: list) -> list:
    return [c.get("type") for c in chunks]


def _block_type(chunks: list, index: int) -> str:
    """Get the content_block type from the content_block_start at the given index."""
    assert index < len(chunks)
    return chunks[index]["content_block"]["type"]


def test_reasoning_content_should_be_translated_to_thinking_block_type():
    """reasoning_content deltas must produce a thinking content block, not text."""
    responses = [
        _make_reasoning_chunk("hello"),
        _make_stop_chunk(),
    ]
    wrapper = AnthropicStreamWrapper(
        completion_stream=MockCompletionStream(responses),
        model="deepseek-v4-flash",
    )
    chunks = _chunks_for(wrapper)

    # The first content_block_start must be a "thinking" block
    starts = [c for c in chunks if c["type"] == "content_block_start"]
    assert len(starts) == 1
    assert starts[0]["content_block"]["type"] == "thinking"


def test_reasoning_content_should_include_signature_field_in_thinking_block_start():
    """The thinking content_block_start must include the signature field."""
    responses = [
        _make_reasoning_chunk("hello"),
        _make_stop_chunk(),
    ]
    wrapper = AnthropicStreamWrapper(
        completion_stream=MockCompletionStream(responses),
        model="deepseek-v4-flash",
    )
    chunks = _chunks_for(wrapper)

    starts = [c for c in chunks if c["type"] == "content_block_start"]
    assert starts[0]["content_block"].get("signature") == ""


def test_reasoning_content_should_emit_content_block_stop_for_thinking():
    """A thinking block must be closed with a content_block_stop."""
    responses = [
        _make_reasoning_chunk("hello"),
        _make_stop_chunk(),
    ]
    wrapper = AnthropicStreamWrapper(
        completion_stream=MockCompletionStream(responses),
        model="deepseek-v4-flash",
    )
    chunks = _chunks_for(wrapper)

    types = _types(chunks)
    assert types.count("content_block_stop") == 1


def test_reasoning_then_text_should_use_incrementing_indices():
    """
    When transitioning from reasoning_content to text, the text block must
    appear at a new index (not the same index as thinking).
    """
    responses = [
        _make_reasoning_chunk("thinking..."),
        _make_text_chunk("answer"),
        _make_stop_chunk(),
    ]
    wrapper = AnthropicStreamWrapper(
        completion_stream=MockCompletionStream(responses),
        model="deepseek-v4-flash",
    )
    chunks = _chunks_for(wrapper)

    starts = [c for c in chunks if c["type"] == "content_block_start"]
    assert len(starts) == 2
    assert starts[0]["index"] == 0
    assert starts[1]["index"] == 1


def test_reasoning_then_text_should_emit_block_transition():
    """
    thinking -> text transition must emit:
    content_block_stop (for thinking) -> content_block_start (for text).
    This verifies issues #3 (missing content_block_stop) and #5 (missing
    block separator).
    """
    responses = [
        _make_reasoning_chunk("thinking..."),
        _make_text_chunk("answer"),
        _make_stop_chunk(),
    ]
    wrapper = AnthropicStreamWrapper(
        completion_stream=MockCompletionStream(responses),
        model="deepseek-v4-flash",
    )
    chunks = _chunks_for(wrapper)

    types = _types(chunks)
    assert types == [
        "message_start",
        "content_block_start",  # thinking (index 0)
        "content_block_delta",  # thinking_delta
        "content_block_stop",  # end of thinking
        "content_block_start",  # text (index 1)
        "content_block_delta",  # text_delta
        "content_block_stop",  # end of text
        "message_delta",
        "message_stop",
    ]


def test_reasoning_then_text_should_not_drop_text_delta_on_transition():
    """
    The text delta from the transition trigger chunk must not be silently
    dropped — it must be emitted after the new content_block_start.
    """
    responses = [
        _make_reasoning_chunk("thinking..."),
        _make_text_chunk("first text"),
        _make_text_chunk(" more text"),
        _make_stop_chunk(),
    ]
    wrapper = AnthropicStreamWrapper(
        completion_stream=MockCompletionStream(responses),
        model="deepseek-v4-flash",
    )
    chunks = _chunks_for(wrapper)

    text_deltas = [
        c["delta"]["text"]
        for c in chunks
        if c["type"] == "content_block_delta"
        and c.get("delta", {}).get("type") == "text_delta"
    ]
    assert text_deltas == ["first text", " more text"]


def test_full_deepseek_streaming_should_match_expected_anthropic_sse():
    """
    End-to-end simulation of a DeepSeek stream (reasoning_content followed
    by text content) and verify the full Anthropic SSE sequence.
    """
    responses = [
        _make_reasoning_chunk("我们需要"),
        _make_reasoning_chunk("理解问题"),
        _make_text_chunk("抱歉"),
        _make_text_chunk("，无法查询"),
        _make_stop_chunk(),
    ]
    wrapper = AnthropicStreamWrapper(
        completion_stream=MockCompletionStream(responses),
        model="deepseek-v4-flash",
    )
    chunks = _chunks_for(wrapper)

    types = _types(chunks)
    assert types == [
        "message_start",
        "content_block_start",  # thinking (index 0)
        "content_block_delta",  # thinking_delta: "我们需要"
        "content_block_delta",  # thinking_delta: "理解问题"
        "content_block_stop",  # end of thinking
        "content_block_start",  # text (index 1)
        "content_block_delta",  # text_delta: "抱歉"
        "content_block_delta",  # text_delta: "，无法查询"
        "content_block_stop",  # end of text
        "message_delta",
        "message_stop",
    ]

    # Verify block types and indices
    starts = [c for c in chunks if c["type"] == "content_block_start"]
    assert starts[0]["content_block"]["type"] == "thinking"
    assert starts[0]["index"] == 0
    assert starts[1]["content_block"]["type"] == "text"
    assert starts[1]["index"] == 1

    # Verify thinking content was preserved
    thinking_deltas = [
        c["delta"]["thinking"]
        for c in chunks
        if c["type"] == "content_block_delta"
        and c.get("delta", {}).get("type") == "thinking_delta"
    ]
    assert thinking_deltas == ["我们需要", "理解问题"]


@pytest.mark.asyncio
async def test_async_reasoning_content_streaming_should_match_sync():
    """The async path should produce the same output as the sync path."""
    responses = [
        _make_reasoning_chunk("thinking..."),
        _make_text_chunk("answer"),
        _make_stop_chunk(),
    ]

    # sync
    sync_wrapper = AnthropicStreamWrapper(
        completion_stream=MockCompletionStream(responses),
        model="deepseek-v4-flash",
    )
    sync_chunks = _chunks_for(sync_wrapper)

    # async (need a fresh mock stream since the index pointer is shared)
    responses2 = [
        _make_reasoning_chunk("thinking..."),
        _make_text_chunk("answer"),
        _make_stop_chunk(),
    ]
    async_wrapper = AnthropicStreamWrapper(
        completion_stream=MockCompletionStream(responses2),
        model="deepseek-v4-flash",
    )
    async_chunks = []
    async for chunk in async_wrapper:
        async_chunks.append(chunk)

    # Types must match exactly
    assert _types(sync_chunks) == _types(async_chunks)

    # Compare everything except the dynamically-generated message id
    assert len(sync_chunks) == len(async_chunks)
    for sc, ac in zip(sync_chunks, async_chunks):
        assert sc["type"] == ac["type"]
        if sc["type"] == "message_start":
            # Both must have the same structure, only the id differs
            assert sc["message"]["type"] == ac["message"]["type"]
            assert sc["message"]["role"] == ac["message"]["role"]
            assert sc["message"]["model"] == ac["message"]["model"]
            assert sc["message"]["usage"] == ac["message"]["usage"]
        else:
            # Non-message_start chunks must be identical
            assert sc == ac
