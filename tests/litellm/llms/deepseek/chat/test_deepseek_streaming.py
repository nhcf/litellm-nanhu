"""
Unit tests for DeepSeek streaming iterator.

Tests correct handling of Claude-format thinking blocks:
- content_block_type identification
- signature field preservation
- block index increment
- reasoning token counting
"""

import pytest
from litellm.llms.deepseek.chat.streaming_iterator import DeepSeekStreamingIterator
from litellm.types.llms.openai import ChatCompletionThinkingBlock


class TestDeepSeekStreamingIterator:
    """Test DeepSeek streaming iterator for thinking block handling."""

    def setup_method(self):
        """Create iterator instance for each test."""
        self.iterator = DeepSeekStreamingIterator(
            streaming_response=None,
            sync_stream=True,
        )

    def test_message_start_event(self):
        """Test message_start event handling."""
        chunk = {
            "type": "message_start",
            "message": {
                "id": "msg_test123",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 0,
                },
            },
        }

        response = self.iterator.chunk_parser(chunk)

        assert response.id == "msg_test123"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 0

    def test_content_block_start_thinking(self):
        """Test thinking block correctly identified in content_block_start."""
        chunk = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "thinking",
                "thinking": "",
                "signature": "sig_test_123",
            },
        }

        response = self.iterator.chunk_parser(chunk)

        # Check current block type tracking
        assert self.iterator.current_content_block_type == "thinking"
        assert self.iterator.current_signature == "sig_test_123"

        # Check response structure
        assert len(response.choices) == 1
        assert response.choices[0].index == 0

        # Check thinking blocks
        delta = response.choices[0].delta
        assert hasattr(delta, "thinking_blocks")
        assert len(delta.thinking_blocks) == 1
        assert delta.thinking_blocks[0]["type"] == "thinking"
        assert delta.thinking_blocks[0]["signature"] == "sig_test_123"

        # Check reasoning_content field for OpenAI compatibility
        assert hasattr(delta, "reasoning_content")

    def test_content_block_start_text(self):
        """Test text block correctly identified in content_block_start."""
        chunk = {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "text",
                "text": "",
            },
        }

        response = self.iterator.chunk_parser(chunk)

        # Check current block type tracking
        assert self.iterator.current_content_block_type == "text"

        # Check response structure
        assert len(response.choices) == 1
        assert response.choices[0].index == 1

        # Check delta content
        delta = response.choices[0].delta
        assert delta.content == ""

    def test_content_block_delta_thinking(self):
        """Test thinking_delta correctly handled."""
        chunk = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "thinking_delta",
                "thinking": "我们需要理解用户的问题",
            },
        }

        response = self.iterator.chunk_parser(chunk)

        # Check reasoning content accumulation
        assert len(self.iterator.reasoning_content_chunks) == 1
        assert self.iterator.reasoning_content_chunks[0] == "我们需要理解用户的问题"

        # Check response structure
        delta = response.choices[0].delta

        # Check thinking blocks
        assert hasattr(delta, "thinking_blocks")
        assert len(delta.thinking_blocks) == 1
        assert delta.thinking_blocks[0]["thinking"] == "我们需要理解用户的问题"

        # Check reasoning_content
        assert delta.reasoning_content == "我们需要理解用户的问题"

    def test_content_block_delta_text(self):
        """Test text_delta correctly handled."""
        chunk = {
            "type": "content_block_delta",
            "index": 1,
            "delta": {
                "type": "text_delta",
                "text": "抱歉，我无法实时查询天气",
            },
        }

        response = self.iterator.chunk_parser(chunk)

        # Check response structure
        delta = response.choices[0].delta
        assert delta.content == "抱歉，我无法实时查询天气"

    def test_content_block_stop_index_increment(self):
        """Test index correctly incremented after content_block_stop."""
        # Start with thinking block
        chunk1 = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": "sig"},
        }
        self.iterator.chunk_parser(chunk1)

        assert self.iterator.current_block_index == 0

        # End thinking block
        chunk2 = {"type": "content_block_stop", "index": 0}
        self.iterator.chunk_parser(chunk2)

        # Check index incremented
        assert self.iterator.current_block_index == 1
        assert self.iterator.completed_blocks == [0]
        assert self.iterator.current_content_block_type is None

    def test_full_streaming_sequence(self):
        """Test complete streaming sequence: thinking → text."""
        chunks = [
            # message_start
            {
                "type": "message_start",
                "message": {
                    "id": "msg_test",
                    "usage": {"input_tokens": 5, "output_tokens": 0},
                },
            },
            # thinking block start
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "thinking",
                    "thinking": "",
                    "signature": "sig123",
                },
            },
            # thinking delta 1
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "我们需要"},
            },
            # thinking delta 2
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "理解问题"},
            },
            # thinking block stop
            {"type": "content_block_stop", "index": 0},
            # text block start
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": ""},
            },
            # text delta
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": "抱歉"},
            },
            # text block stop
            {"type": "content_block_stop", "index": 1},
            # message_delta with usage
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"input_tokens": 88, "output_tokens": 172},
            },
            # message_stop
            {"type": "message_stop"},
        ]

        responses = []
        for chunk in chunks:
            response = self.iterator.chunk_parser(chunk)
            responses.append(response)

        # Check final state
        assert len(self.iterator.completed_blocks) == 2
        assert self.iterator.current_block_index == 2

        # Check reasoning content accumulation
        reasoning_content = "".join(self.iterator.reasoning_content_chunks)
        assert reasoning_content == "我们需要理解问题"

        # Check final usage
        final_response = responses[-2]  # message_delta response
        assert final_response.usage.prompt_tokens == 88
        assert final_response.usage.completion_tokens == 172

        # Check reasoning tokens in completion_tokens_details
        if hasattr(final_response.usage, "completion_tokens_details"):
            assert final_response.usage.completion_tokens_details["reasoning_tokens"] > 0

    def test_signature_field_preserved(self):
        """Test signature field is correctly preserved."""
        # Thinking block start with signature
        chunk1 = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "thinking",
                "thinking": "",
                "signature": "initial_signature",
            },
        }
        self.iterator.chunk_parser(chunk1)

        assert self.iterator.current_signature == "initial_signature"

        # Thinking delta without signature (should use initial)
        chunk2 = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "思考内容"},
        }
        response = self.iterator.chunk_parser(chunk2)

        # Signature should still be initial_signature
        delta = response.choices[0].delta
        assert delta.thinking_blocks[0]["signature"] == "initial_signature"

        # Signature delta event
        chunk3 = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "final_signature"},
        }
        self.iterator.chunk_parser(chunk3)

        assert self.iterator.current_signature == "final_signature"

    def test_index_consistency(self):
        """Test block indices are consistent throughout streaming."""
        # Thinking block (index=0)
        chunk1 = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        }
        response1 = self.iterator.chunk_parser(chunk1)
        assert response1.choices[0].index == 0

        # Thinking delta (index=0)
        chunk2 = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "思考"},
        }
        response2 = self.iterator.chunk_parser(chunk2)
        assert response2.choices[0].index == 0

        # Thinking stop (index=0)
        chunk3 = {"type": "content_block_stop", "index": 0}
        response3 = self.iterator.chunk_parser(chunk3)
        assert response3.choices[0].index == 0

        # Text block (index=1)
        chunk4 = {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "text", "text": ""},
        }
        response4 = self.iterator.chunk_parser(chunk4)
        assert response4.choices[0].index == 1

        # Text delta (index=1)
        chunk5 = {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "回答"},
        }
        response5 = self.iterator.chunk_parser(chunk5)
        assert response5.choices[0].index == 1

    def test_redacted_thinking_block(self):
        """Test redacted_thinking block handling."""
        chunk = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "redacted_thinking",
                "data": "redacted_data_here",
            },
        }

        response = self.iterator.chunk_parser(chunk)

        # Check thinking blocks
        delta = response.choices[0].delta
        assert hasattr(delta, "thinking_blocks")
        assert len(delta.thinking_blocks) == 1
        assert delta.thinking_blocks[0]["type"] == "redacted_thinking"
        assert delta.thinking_blocks[0]["data"] == "redacted_data_here"

    def test_ping_event_ignored(self):
        """Test ping events are ignored."""
        chunk = {"type": "ping"}

        response = self.iterator.chunk_parser(chunk)

        # Should return empty response
        assert len(response.choices) == 0

    def test_unknown_event_type(self):
        """Test unknown event types return empty response."""
        chunk = {"type": "unknown_event"}

        response = self.iterator.chunk_parser(chunk)

        # Should return empty response
        assert len(response.choices) == 0


class TestDeepSeekThinkingBlockFormat:
    """Test DeepSeek thinking block format matches Claude format."""

    def test_thinking_block_structure(self):
        """Test thinking block has correct Claude structure."""
        iterator = DeepSeekStreamingIterator(None, True)

        chunk = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "thinking",
                "thinking": "思考内容",
                "signature": "sig_test",
            },
        }

        response = iterator.chunk_parser(chunk)

        # Check structure matches Claude format
        delta = response.choices[0].delta

        # Required fields
        assert hasattr(delta, "thinking_blocks")
        thinking_block = delta.thinking_blocks[0]

        assert thinking_block["type"] == "thinking"
        assert "thinking" in thinking_block
        assert "signature" in thinking_block

        # Optional: reasoning_content for OpenAI compatibility
        assert hasattr(delta, "reasoning_content")

    def test_thinking_delta_vs_text_delta(self):
        """Test thinking_delta and text_delta are correctly distinguished."""
        iterator = DeepSeekStreamingIterator(None, True)

        # Thinking delta
        thinking_chunk = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "思考"},
        }

        thinking_response = iterator.chunk_parser(thinking_chunk)
        thinking_delta = thinking_response.choices[0].delta

        # Should have thinking_blocks and reasoning_content
        assert hasattr(thinking_delta, "thinking_blocks")
        assert hasattr(thinking_delta, "reasoning_content")

        # Text delta
        text_chunk = {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "回答"},
        }

        text_response = iterator.chunk_parser(text_chunk)
        text_delta = text_response.choices[0].delta

        # Should have content, not thinking_blocks
        assert hasattr(text_delta, "content")
        assert not hasattr(text_delta, "thinking_blocks") or len(text_delta.thinking_blocks) == 0


class TestDeepSeekTokenUsage:
    """Test token usage calculation for DeepSeek streaming."""

    def test_reasoning_tokens_calculated(self):
        """Test reasoning tokens are calculated from thinking content."""
        iterator = DeepSeekStreamingIterator(None, True)

        # Accumulate reasoning content
        iterator.reasoning_content_chunks = ["思考内容一", "思考内容二"]

        # Message delta with usage
        chunk = {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"input_tokens": 88, "output_tokens": 172},
        }

        response = iterator.chunk_parser(chunk)

        # Check completion_tokens_details
        if hasattr(response.usage, "completion_tokens_details"):
            details = response.usage.completion_tokens_details
            assert "reasoning_tokens" in details
            assert details["reasoning_tokens"] > 0
            assert "text_tokens" in details
            # text_tokens = total - reasoning
            assert details["text_tokens"] == 172 - details["reasoning_tokens"]

    def test_usage_without_reasoning(self):
        """Test usage calculation without reasoning content."""
        iterator = DeepSeekStreamingIterator(None, True)

        # No reasoning content accumulated
        iterator.reasoning_content_chunks = []

        chunk = {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }

        response = iterator.chunk_parser(chunk)

        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 20

    def test_cache_tokens_preserved(self):
        """Test cache-related tokens are preserved."""
        iterator = DeepSeekStreamingIterator(None, True)

        chunk = {
            "type": "message_delta",
            "delta": {},
            "usage": {
                "input_tokens": 88,
                "output_tokens": 172,
                "cache_creation_input_tokens": 5,
                "cache_read_input_tokens": 10,
            },
        }

        response = iterator.chunk_parser(chunk)

        assert response.usage.cache_creation_input_tokens == 5
        assert response.usage.cache_read_input_tokens == 10