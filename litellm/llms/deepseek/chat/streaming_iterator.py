"""
DeepSeek streaming iterator for handling Claude-format thinking blocks.

Handles the multi-block streaming sequence:
- thinking block (index=0)
- content_block_stop
- text block (index=1)
- content_block_stop

Correctly tracks:
- content_block_type
- signature field
- reasoning_content_chunks for token counting
- block index increment
"""

from typing import Any, Dict, List, Optional, Union, cast

from litellm.litellm_core_utils.core_helpers import map_finish_reason
from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.types.llms.openai import (
    ChatCompletionRedactedThinkingBlock,
    ChatCompletionThinkingBlock,
)
from litellm.types.utils import (
    Delta,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)


class DeepSeekStreamingIterator(BaseModelResponseIterator):
    """
    Handles DeepSeek's Claude-format streaming responses.

    DeepSeek returns responses in Anthropic Claude format:
    - content_block_start with type: "thinking" or "text"
    - content_block_delta with thinking_delta or text_delta
    - content_block_stop events
    - message_delta with final usage

    This iterator converts to OpenAI-compatible format while preserving:
    - thinking_blocks structure
    - signature field
    - correct block index increment
    - accurate token usage statistics
    """

    def __init__(
        self,
        streaming_response: Union[Any, Dict],
        sync_stream: bool,
    ):
        super().__init__(streaming_response, sync_stream)

        # Track current content block type
        self.current_content_block_type: Optional[str] = None

        # Accumulate reasoning/thinking content for token counting
        self.reasoning_content_chunks: List[str] = []

        # Track block indices (DeepSeek uses 0 for thinking, 1 for text)
        self.current_block_index: int = 0
        self.completed_blocks: List[int] = []

        # Current signature for thinking block
        self.current_signature: Optional[str] = None

        # Response ID
        self.response_id: Optional[str] = None

    def chunk_parser(self, chunk: dict) -> ModelResponseStream:
        """
        Parse a DeepSeek streaming chunk and convert to OpenAI format.

        Args:
            chunk: Raw chunk from DeepSeek API
                - {"type": "content_block_start", ...}
                - {"type": "content_block_delta", ...}
                - {"type": "content_block_stop", ...}
                - {"type": "message_start", ...}
                - {"type": "message_delta", ...}
                - {"type": "message_stop", ...}

        Returns:
            ModelResponseStream: OpenAI-compatible streaming response
        """
        try:
            # Extract event type
            event_type = chunk.get("type")

            # Handle different event types
            if event_type == "message_start":
                return self._handle_message_start(chunk)
            elif event_type == "content_block_start":
                return self._handle_content_block_start(chunk)
            elif event_type == "content_block_delta":
                return self._handle_content_block_delta(chunk)
            elif event_type == "content_block_stop":
                return self._handle_content_block_stop(chunk)
            elif event_type == "message_delta":
                return self._handle_message_delta(chunk)
            elif event_type == "message_stop":
                return self._handle_message_stop(chunk)
            elif event_type == "ping":
                # Ping events are ignored in streaming
                return ModelResponseStream()
            else:
                # Unknown event type, return empty response
                return ModelResponseStream()

        except Exception as e:
            raise e

    def _handle_message_start(self, chunk: dict) -> ModelResponseStream:
        """
        Handle message_start event.

        Example:
        {"type": "message_start", "message": {"id": "...", "usage": {...}}}
        """
        message = chunk.get("message", {})
        self.response_id = message.get("id")

        # Initial usage (may have input_tokens, but output_tokens usually 0)
        usage_dict = message.get("usage", {})

        return ModelResponseStream(
            id=self.response_id,
            choices=[],
            usage=Usage(
                prompt_tokens=usage_dict.get("input_tokens", 0),
                completion_tokens=usage_dict.get("output_tokens", 0),
                total_tokens=(
                    usage_dict.get("input_tokens", 0)
                    + usage_dict.get("output_tokens", 0)
                ),
            ),
        )

    def _handle_content_block_start(
        self, chunk: dict
    ) -> ModelResponseStream:
        """
        Handle content_block_start event.

        DeepSeek format:
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": "..."}}

        Or:
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}
        """
        content_block = chunk.get("content_block", {})
        block_type = content_block.get("type")

        # Set current block type
        self.current_content_block_type = block_type

        # Use chunk's index, but ensure it's consistent with our tracking
        # DeepSeek sends the correct index in the chunk
        chunk_index = chunk.get("index", self.current_block_index)

        # Update our tracking if needed
        if chunk_index != self.current_block_index:
            self.current_block_index = chunk_index

        delta_dict: Dict[str, Any] = {}

        if block_type == "thinking":
            # Save signature
            self.current_signature = content_block.get("signature", "")

            # Reset reasoning chunks for new thinking block
            self.reasoning_content_chunks = []

            # Create thinking block structure
            thinking_content = content_block.get("thinking", "")
            if thinking_content:
                self.reasoning_content_chunks.append(thinking_content)

            delta_dict["thinking_blocks"] = [
                ChatCompletionThinkingBlock(
                    type="thinking",
                    thinking=thinking_content,
                    signature=self.current_signature or "",
                )
            ]

            # Also provide reasoning_content for OpenAI compatibility
            delta_dict["reasoning_content"] = thinking_content

        elif block_type == "text":
            text_content = content_block.get("text", "")
            delta_dict["content"] = text_content

        elif block_type == "redacted_thinking":
            # Handle redacted thinking block
            data = content_block.get("data", "")
            delta_dict["thinking_blocks"] = [
                ChatCompletionRedactedThinkingBlock(
                    type="redacted_thinking",
                    data=data,
                )
            ]

        return self._create_model_response(chunk_index, delta_dict)

    def _handle_content_block_delta(
        self, chunk: dict
    ) -> ModelResponseStream:
        """
        Handle content_block_delta event.

        DeepSeek format:
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "我们需要"}}

        Or:
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "抱歉"}}
        """
        delta = chunk.get("delta", {})
        delta_type = delta.get("type")

        # Use chunk's index
        chunk_index = chunk.get("index", self.current_block_index)

        delta_dict: Dict[str, Any] = {}

        if delta_type == "thinking_delta":
            thinking_content = delta.get("thinking", "")

            # Accumulate thinking content for token counting
            if thinking_content:
                self.reasoning_content_chunks.append(thinking_content)

            # Get signature (may be in delta or from block start)
            signature = delta.get("signature", self.current_signature or "")

            # Create thinking block
            delta_dict["thinking_blocks"] = [
                ChatCompletionThinkingBlock(
                    type="thinking",
                    thinking=thinking_content,
                    signature=signature,
                )
            ]

            # Also provide reasoning_content for OpenAI compatibility
            delta_dict["reasoning_content"] = thinking_content

        elif delta_type == "text_delta":
            text_content = delta.get("text", "")
            delta_dict["content"] = text_content

        elif delta_type == "signature_delta":
            # Signature delta event
            signature = delta.get("signature", "")
            if signature:
                self.current_signature = signature

        elif delta_type == "input_json_delta":
            # Tool use delta (if DeepSeek supports tools)
            delta_dict["tool_calls"] = [
                {
                    "index": chunk_index,
                    "function": {
                        "arguments": delta.get("partial_json", ""),
                    },
                }
            ]

        return self._create_model_response(chunk_index, delta_dict)

    def _handle_content_block_stop(
        self, chunk: dict
    ) -> ModelResponseStream:
        """
        Handle content_block_stop event.

        DeepSeek format:
        {"type": "content_block_stop", "index": 0}

        This marks the end of a content block (thinking or text).
        """
        chunk_index = chunk.get("index", self.current_block_index)

        # Record completed block
        self.completed_blocks.append(chunk_index)

        # Increment index for next block
        self.current_block_index = chunk_index + 1

        # Reset current block type
        self.current_content_block_type = None

        # Return empty response for stop event
        # The index should match the block that just ended
        return ModelResponseStream(
            id=self.response_id,
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=chunk_index,
                    delta=Delta(content=""),
                )
            ],
        )

    def _handle_message_delta(self, chunk: dict) -> ModelResponseStream:
        """
        Handle message_delta event with final usage.

        DeepSeek format:
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"input_tokens": 88, "output_tokens": 172}}

        Calculate reasoning tokens from accumulated thinking content.
        """
        delta = chunk.get("delta", {})
        usage_dict = chunk.get("usage", {})

        # Get stop reason
        stop_reason = delta.get("stop_reason")
        finish_reason = map_finish_reason(stop_reason) if stop_reason else None

        # Calculate reasoning tokens from accumulated thinking content
        reasoning_content = "".join(self.reasoning_content_chunks)
        reasoning_tokens = 0
        text_tokens = 0

        if reasoning_content:
            try:
                from litellm.utils import token_counter

                completion_tokens = usage_dict.get("output_tokens", 0)
                estimated_reasoning_tokens = token_counter(
                    text=reasoning_content, count_response_tokens=True
                )
                reasoning_tokens = min(estimated_reasoning_tokens, completion_tokens)
                text_tokens = completion_tokens - reasoning_tokens
            except Exception:
                # If token_counter fails, just use output_tokens
                pass

        # Build usage with details
        usage = Usage(
            prompt_tokens=usage_dict.get("input_tokens", 0),
            completion_tokens=usage_dict.get("output_tokens", 0),
            total_tokens=(
                usage_dict.get("input_tokens", 0)
                + usage_dict.get("output_tokens", 0)
            ),
        )

        # Add completion tokens details if we have reasoning tokens
        if reasoning_tokens > 0:
            usage.completion_tokens_details = {
                "reasoning_tokens": reasoning_tokens,
                "text_tokens": text_tokens,
            }

        # Add cache details if present
        if "cache_creation_input_tokens" in usage_dict:
            usage.cache_creation_input_tokens = usage_dict.get(
                "cache_creation_input_tokens", 0
            )
        if "cache_read_input_tokens" in usage_dict:
            usage.cache_read_input_tokens = usage_dict.get("cache_read_input_tokens", 0)

        return ModelResponseStream(
            id=self.response_id,
            choices=[
                StreamingChoices(
                    finish_reason=finish_reason,
                    index=0,  # OpenAI format always uses index=0 for single choice
                    delta=Delta(),
                )
            ],
            usage=usage,
        )

    def _handle_message_stop(self, chunk: dict) -> ModelResponseStream:
        """
        Handle message_stop event.

        This is the final event in the streaming sequence.
        """
        return ModelResponseStream(
            id=self.response_id,
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(),
                )
            ],
        )

    def _create_model_response(
        self, index: int, delta_dict: Dict[str, Any]
    ) -> ModelResponseStream:
        """
        Create a ModelResponseStream from the parsed chunk.

        Args:
            index: Block index (0 for thinking, 1 for text)
            delta_dict: Dictionary of delta content

        Returns:
            ModelResponseStream with proper structure
        """
        # Ensure content field exists
        if "content" not in delta_dict and "thinking_blocks" not in delta_dict:
            delta_dict["content"] = ""

        # Create delta object
        delta = Delta(**delta_dict)

        return ModelResponseStream(
            id=self.response_id,
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=index,
                    delta=delta,
                )
            ],
        )