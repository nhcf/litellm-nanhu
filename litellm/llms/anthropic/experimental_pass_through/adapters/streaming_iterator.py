# What is this?
## Translates OpenAI call to Anthropic `/v1/messages` format
import json
import traceback
from collections import deque
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Iterator, Literal, Optional

from litellm import verbose_logger
from litellm._uuid import uuid
from litellm.types.llms.anthropic import UsageDelta
from litellm.types.utils import AdapterCompletionStreamWrapper

if TYPE_CHECKING:
    from litellm.types.utils import ModelResponseStream


class AnthropicStreamWrapper(AdapterCompletionStreamWrapper):
    """
    - first chunk return 'message_start'
    - content block must be started and stopped
    - finish_reason must map exactly to anthropic reason, else anthropic client won't be able to parse it.
    """

    from litellm.types.llms.anthropic import (
        ContentBlockContentBlockDict,
        ContentBlockStart,
        ContentBlockStartText,
        TextBlock,
    )

    sent_first_chunk: bool = False
    sent_content_block_start: bool = False
    sent_content_block_finish: bool = False
    current_content_block_type: Literal["text", "tool_use", "thinking"] = "text"
    sent_last_message: bool = False
    holding_chunk: Optional[Any] = None
    holding_stop_reason_chunk: Optional[Any] = None
    queued_usage_chunk: bool = False
    current_content_block_index: int = 0
    current_content_block_start: ContentBlockContentBlockDict = TextBlock(
        type="text",
        text="",
    )
    chunk_queue: deque = deque()  # Queue for buffering multiple chunks

    def __init__(
        self,
        completion_stream: Any,
        model: str,
        tool_name_mapping: Optional[Dict[str, str]] = None,
    ):
        super().__init__(completion_stream)
        self.model = model
        # Mapping of truncated tool names to original names (for OpenAI's 64-char limit)
        self.tool_name_mapping = tool_name_mapping or {}

    def _create_initial_usage_delta(self) -> UsageDelta:
        """
        Create the initial UsageDelta for the message_start event.

        Initializes cache token fields (cache_creation_input_tokens, cache_read_input_tokens)
        to 0 to indicate to clients (like Claude Code) that prompt caching is supported.

        The actual cache token values will be provided in the message_delta event at the
        end of the stream, since Bedrock Converse API only returns usage data in the final
        response chunk.

        Returns:
            UsageDelta with all token counts initialized to 0.
        """
        return UsageDelta(
            input_tokens=0,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            prompt_tokens_details={"cached_tokens": 0},
        )

    def __next__(self):
        from .transformation import LiteLLMAnthropicMessagesAdapter

        try:
            # Always return queued chunks first
            if self.chunk_queue:
                return self.chunk_queue.popleft()

            # Queue initial chunks if not sent yet
            if self.sent_first_chunk is False:
                self.sent_first_chunk = True
                self.chunk_queue.append(
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg_{}".format(uuid.uuid4()),
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": self.model,
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": self._create_initial_usage_delta(),
                        },
                    }
                )
                return self.chunk_queue.popleft()

            for chunk in self.completion_stream:
                if chunk == "None" or chunk is None:
                    raise Exception

                # Skip chunks that carry no meaningful content (e.g. DeepSeek's
                # leading content="" chunk) so they don't start a spurious
                # empty text block at index 0. This runs before the first
                # content_block_start is emitted.
                if self._is_chunk_empty(chunk):
                    continue

                should_start_new_block = self._should_start_new_content_block(chunk)

                # Emit the first content_block_start based on the first chunk's
                # actual type, instead of eagerly assuming "text". Providers that
                # lead with reasoning (e.g. DeepSeek via ``reasoning_content``) must
                # get a proper "thinking" block at index 0.
                if self.sent_content_block_start is False:
                    self.sent_content_block_start = True
                    # ``_should_start_new_content_block`` already populated
                    # ``current_content_block_start`` / ``current_content_block_type``.
                    self.chunk_queue.append(
                        {
                            "type": "content_block_start",
                            "index": self.current_content_block_index,
                            "content_block": self.current_content_block_start,
                        }
                    )
                    # This is the very first block — there is no prior block to
                    # stop, so do not treat it as a block transition.
                    should_start_new_block = False
                elif should_start_new_block:
                    self._increment_content_block_index()

                processed_chunk = LiteLLMAnthropicMessagesAdapter().translate_streaming_openai_response_to_anthropic(
                    response=chunk,
                    current_content_block_index=self.current_content_block_index,
                )

                if should_start_new_block and not self.sent_content_block_finish:
                    # Queue the sequence: content_block_stop -> content_block_start.
                    # The trigger chunk's delta is re-queued via
                    # ``_maybe_queue_trigger_delta`` so its content is not silently
                    # dropped when switching block types (e.g. thinking -> text,
                    # where the first text chunk carries actual text content).

                    # 1. Stop current content block
                    self.chunk_queue.append(
                        {
                            "type": "content_block_stop",
                            "index": max(self.current_content_block_index - 1, 0),
                        }
                    )

                    # 2. Start new content block
                    self.chunk_queue.append(
                        {
                            "type": "content_block_start",
                            "index": self.current_content_block_index,
                            "content_block": self.current_content_block_start,
                        }
                    )

                    # 3. Re-queue the trigger chunk's delta when it carries content.
                    self._maybe_queue_trigger_delta(processed_chunk)

                    self.sent_content_block_finish = False
                    return self.chunk_queue.popleft()

                if (
                    processed_chunk["type"] == "message_delta"
                    and self.sent_content_block_finish is False
                ):
                    # Queue both the content_block_stop and the message_delta
                    self.chunk_queue.append(
                        {
                            "type": "content_block_stop",
                            "index": self.current_content_block_index,
                        }
                    )
                    self.sent_content_block_finish = True
                    self.chunk_queue.append(processed_chunk)
                    return self.chunk_queue.popleft()
                elif self.holding_chunk is not None:
                    self.chunk_queue.append(self.holding_chunk)
                    self.chunk_queue.append(processed_chunk)
                    self.holding_chunk = None
                    return self.chunk_queue.popleft()
                else:
                    self.chunk_queue.append(processed_chunk)
                    return self.chunk_queue.popleft()

            # Handle any remaining held chunks after stream ends
            if self.holding_chunk is not None:
                self.chunk_queue.append(self.holding_chunk)
                self.holding_chunk = None

            if not self.sent_last_message:
                self.sent_last_message = True
                self.chunk_queue.append({"type": "message_stop"})

            if self.chunk_queue:
                return self.chunk_queue.popleft()

            raise StopIteration
        except StopIteration:
            if self.chunk_queue:
                return self.chunk_queue.popleft()
            if self.sent_last_message is False:
                self.sent_last_message = True
                return {"type": "message_stop"}
            raise StopIteration
        except Exception as e:
            verbose_logger.error(
                "Anthropic Adapter - {}\n{}".format(e, traceback.format_exc())
            )
            raise StopAsyncIteration

    async def __anext__(self):  # noqa: PLR0915
        from .transformation import LiteLLMAnthropicMessagesAdapter

        try:
            # Always return queued chunks first
            if self.chunk_queue:
                return self.chunk_queue.popleft()

            # Queue initial chunks if not sent yet
            if self.sent_first_chunk is False:
                self.sent_first_chunk = True
                self.chunk_queue.append(
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg_{}".format(uuid.uuid4()),
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": self.model,
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": self._create_initial_usage_delta(),
                        },
                    }
                )
                return self.chunk_queue.popleft()

            async for chunk in self.completion_stream:
                if chunk == "None" or chunk is None:
                    raise Exception

                # Skip chunks that carry no meaningful content (e.g. DeepSeek's
                # leading content="" chunk) so they don't start a spurious
                # empty text block at index 0. This runs before the first
                # content_block_start is emitted.
                if self._is_chunk_empty(chunk):
                    continue

                # Check if we need to start a new content block
                should_start_new_block = self._should_start_new_content_block(chunk)

                # Emit the first content_block_start based on the first chunk's
                # actual type, instead of eagerly assuming "text". Providers that
                # lead with reasoning (e.g. DeepSeek via ``reasoning_content``) must
                # get a proper "thinking" block at index 0.
                if self.sent_content_block_start is False:
                    self.sent_content_block_start = True
                    self.chunk_queue.append(
                        {
                            "type": "content_block_start",
                            "index": self.current_content_block_index,
                            "content_block": self.current_content_block_start,
                        }
                    )
                    # This is the very first block — there is no prior block to
                    # stop, so do not treat it as a block transition.
                    should_start_new_block = False
                elif should_start_new_block:
                    self._increment_content_block_index()

                processed_chunk = LiteLLMAnthropicMessagesAdapter().translate_streaming_openai_response_to_anthropic(
                    response=chunk,
                    current_content_block_index=self.current_content_block_index,
                )

                # Check if this is a usage chunk and we have a held stop_reason chunk
                if (
                    self.holding_stop_reason_chunk is not None
                    and getattr(chunk, "usage", None) is not None
                ):
                    # Merge usage into the held stop_reason chunk
                    merged_chunk = self.holding_stop_reason_chunk.copy()
                    if "delta" not in merged_chunk:
                        merged_chunk["delta"] = {}

                    # Add usage to the held chunk
                    uncached_input_tokens = chunk.usage.prompt_tokens or 0
                    cached_tokens = 0
                    if (
                        hasattr(chunk.usage, "prompt_tokens_details")
                        and chunk.usage.prompt_tokens_details
                    ):
                        cached_tokens = (
                            getattr(
                                chunk.usage.prompt_tokens_details, "cached_tokens", 0
                            )
                            or 0
                        )
                        uncached_input_tokens -= cached_tokens

                    usage_dict: UsageDelta = {
                        "input_tokens": uncached_input_tokens,
                        "output_tokens": chunk.usage.completion_tokens or 0,
                        "cache_creation_input_tokens": (
                            chunk.usage._cache_creation_input_tokens
                            if hasattr(chunk.usage, "_cache_creation_input_tokens")
                            else 0
                        ),
                        "cache_read_input_tokens": (
                            chunk.usage._cache_read_input_tokens
                            if hasattr(chunk.usage, "_cache_read_input_tokens")
                            else cached_tokens
                        ),
                        "prompt_tokens_details": {"cached_tokens": cached_tokens},
                    }
                    merged_chunk["usage"] = usage_dict

                    # Queue the merged chunk and reset
                    self.chunk_queue.append(merged_chunk)
                    self.queued_usage_chunk = True
                    self.holding_stop_reason_chunk = None
                    return self.chunk_queue.popleft()

                # Check if this processed chunk has a stop_reason - hold it for next chunk

                if not self.queued_usage_chunk:
                    if should_start_new_block and not self.sent_content_block_finish:
                        # Queue the sequence: content_block_stop -> content_block_start.
                        # The trigger chunk's delta is re-queued via
                        # ``_maybe_queue_trigger_delta`` so its content is not silently
                        # dropped when switching block types (e.g. thinking -> text,
                        # where the first text chunk carries actual text content).

                        # 1. Stop current content block
                        self.chunk_queue.append(
                            {
                                "type": "content_block_stop",
                                "index": max(self.current_content_block_index - 1, 0),
                            }
                        )
                        self.chunk_queue.append(
                            {
                                "type": "content_block_start",
                                "index": self.current_content_block_index,
                                "content_block": self.current_content_block_start,
                            }
                        )

                        # 2. Re-queue the trigger chunk's delta when it carries content.
                        self._maybe_queue_trigger_delta(processed_chunk)

                        # Reset state for new block
                        self.sent_content_block_finish = False
                        return self.chunk_queue.popleft()

                    if (
                        processed_chunk["type"] == "message_delta"
                        and self.sent_content_block_finish is False
                    ):
                        # Queue both the content_block_stop and the holding chunk
                        self.chunk_queue.append(
                            {
                                "type": "content_block_stop",
                                "index": self.current_content_block_index,
                            }
                        )
                        self.sent_content_block_finish = True
                        if (
                            processed_chunk.get("delta", {}).get("stop_reason")
                            is not None
                        ):
                            self.holding_stop_reason_chunk = processed_chunk
                        else:
                            self.chunk_queue.append(processed_chunk)
                        return self.chunk_queue.popleft()
                    elif self.holding_chunk is not None:
                        # Queue both chunks
                        self.chunk_queue.append(self.holding_chunk)
                        self.chunk_queue.append(processed_chunk)
                        self.holding_chunk = None
                        return self.chunk_queue.popleft()
                    else:
                        # Queue the current chunk
                        self.chunk_queue.append(processed_chunk)
                        return self.chunk_queue.popleft()

            # Handle any remaining held chunks after stream ends
            if not self.queued_usage_chunk:
                if self.holding_stop_reason_chunk is not None:
                    self.chunk_queue.append(self.holding_stop_reason_chunk)
                    self.holding_stop_reason_chunk = None

                if self.holding_chunk is not None:
                    self.chunk_queue.append(self.holding_chunk)
                    self.holding_chunk = None

            if not self.sent_last_message:
                self.sent_last_message = True
                self.chunk_queue.append({"type": "message_stop"})

            # Return queued items if any
            if self.chunk_queue:
                return self.chunk_queue.popleft()

            raise StopIteration

        except StopIteration:
            # Handle any remaining queued chunks before stopping
            if self.chunk_queue:
                return self.chunk_queue.popleft()
            # Handle any held stop_reason chunk
            if self.holding_stop_reason_chunk is not None:
                return self.holding_stop_reason_chunk
            if not self.sent_last_message:
                self.sent_last_message = True
                return {"type": "message_stop"}
            raise StopAsyncIteration

    def anthropic_sse_wrapper(self) -> Iterator[bytes]:
        """
        Convert AnthropicStreamWrapper dict chunks to Server-Sent Events format.
        Similar to the Bedrock bedrock_sse_wrapper implementation.

        This wrapper ensures dict chunks are SSE formatted with both event and data lines.
        """
        for chunk in self:
            if isinstance(chunk, dict):
                event_type: str = str(chunk.get("type", "message"))
                payload = f"event: {event_type}\ndata: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                yield payload.encode()
            else:
                # For non-dict chunks, forward the original value unchanged
                yield chunk

    async def async_anthropic_sse_wrapper(self) -> AsyncIterator[bytes]:
        """
        Async version of anthropic_sse_wrapper.
        Convert AnthropicStreamWrapper dict chunks to Server-Sent Events format.
        """
        async for chunk in self:
            if isinstance(chunk, dict):
                event_type: str = str(chunk.get("type", "message"))
                payload = f"event: {event_type}\ndata: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                yield payload.encode()
            else:
                # For non-dict chunks, forward the original value unchanged
                yield chunk

    def _increment_content_block_index(self):
        self.current_content_block_index += 1

    def _maybe_queue_trigger_delta(self, processed_chunk: Any) -> None:
        """
        When transitioning between content block types (e.g. thinking -> text),
        the streaming chunk that triggers the transition may itself carry content
        (e.g. the first text delta, or the first tool-call argument payload).

        Previously only ``input_json_delta`` chunks with non-empty ``partial_json``
        were re-queued after the new ``content_block_start``, causing
        ``thinking_delta`` and ``text_delta`` content from the trigger chunk to be
        silently dropped. This helper re-queues any ``content_block_delta`` carried
        by the trigger chunk that has actual content so no content is lost across
        block boundaries, while still skipping empty deltas (e.g. a tool-call name
        chunk that arrives with ``arguments=""``).
        """
        if (
            not isinstance(processed_chunk, dict)
            or processed_chunk.get("type") != "content_block_delta"
            or not isinstance(processed_chunk.get("delta"), dict)
        ):
            return

        delta = processed_chunk["delta"]
        delta_type = delta.get("type")

        if delta_type == "input_json_delta":
            # Only re-queue tool argument deltas that carry actual content.
            if delta.get("partial_json"):
                self.chunk_queue.append(processed_chunk)
        elif delta_type == "thinking_delta":
            if delta.get("thinking"):
                self.chunk_queue.append(processed_chunk)
        elif delta_type == "text_delta":
            if delta.get("text"):
                self.chunk_queue.append(processed_chunk)
        elif delta_type == "signature_delta":
            if delta.get("signature"):
                self.chunk_queue.append(processed_chunk)

    def _is_chunk_empty(self, chunk: "ModelResponseStream") -> bool:
        """
        Check if a streaming chunk carries no meaningful content that would
        start or extend a content block.

        Some providers (e.g. DeepSeek) emit a leading chunk with an empty
        ``content=""`` string before sending reasoning. Such a chunk should
        neither start a content block nor trigger a block transition — otherwise
        a spurious empty "text" block appears at index 0 and pushes the real
        content to higher indices.

        A chunk is considered "empty" only when it has no tool_calls, no
        thinking_blocks, no reasoning_content, and no (or empty) content.
        Chunks that carry a finish_reason or usage are NOT considered empty,
        because they signal stream completion and must be processed.
        """
        # Usage-only chunks (e.g. from stream_options={"include_usage": True})
        # carry token counts even when delta is None. They must not be skipped
        # so the merge logic has a chance to inject usage into message_delta.
        if getattr(chunk, "usage", None) is not None:
            return False
        if not chunk.choices:
            # No choices — could still carry usage/finish, don't skip.
            return False
        choice = chunk.choices[0]
        # finish_reason chunks must be processed (they emit message_delta).
        if choice.finish_reason is not None:
            return False
        delta = choice.delta
        if delta is None:
            return True
        # Non-empty tool calls / thinking blocks keep the chunk significant.
        if getattr(delta, "tool_calls", None):
            return False
        if getattr(delta, "thinking_blocks", None):
            return False
        # reasoning_content counts, even if it is an empty string, because its
        # presence signals that the provider is in thinking mode.
        if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
            return False
        # content only disqualifies emptiness if it is non-empty.
        content = getattr(delta, "content", None)
        if content is not None and len(content) > 0:
            return False
        # Also treat presence of stop_reason on the delta as non-empty so that
        # stop/done chunks are never skipped.
        if getattr(delta, "stop_reason", None) is not None:
            return False
        return True

    def _should_start_new_content_block(self, chunk: "ModelResponseStream") -> bool:
        """
        Determine if we should start a new content block based on the processed chunk.
        Override this method with your specific logic for detecting new content blocks.

        Examples of when you might want to start a new content block:
        - Switching from text to tool calls
        - Different content types in the response
        - Specific markers in the content
        """
        from .transformation import LiteLLMAnthropicMessagesAdapter

        # Usage-only chunks carry token counts (from stream_options={"include_usage": True})
        # They do not represent content block transitions. Handle early to avoid
        # treating them as new blocks.
        if getattr(chunk, "usage", None) is not None and (
            not chunk.choices or chunk.choices[0].finish_reason is None
        ):
            return False

        # If chunk indicates a tool call
        if chunk.choices[0].finish_reason is not None:
            return False

        # Skip empty leading chunks (e.g. DeepSeek's content="" opening chunk).
        if self._is_chunk_empty(chunk):
            return False

        (
            block_type,
            content_block_start,
        ) = LiteLLMAnthropicMessagesAdapter()._translate_streaming_openai_chunk_to_anthropic_content_block(
            choices=chunk.choices  # type: ignore
        )

        # Restore original tool name if it was truncated for OpenAI's 64-char limit
        if block_type == "tool_use":
            # Type narrowing: content_block_start is ToolUseBlock when block_type is "tool_use"
            from typing import cast

            from litellm.types.llms.anthropic import ToolUseBlock

            tool_block = cast(ToolUseBlock, content_block_start)

            if tool_block.get("name"):
                truncated_name = tool_block["name"]
                original_name = self.tool_name_mapping.get(
                    truncated_name, truncated_name
                )
                tool_block["name"] = original_name

        if block_type != self.current_content_block_type:
            self.current_content_block_type = block_type
            self.current_content_block_start = content_block_start
            return True

        # For parallel tool calls, we'll necessarily have a new content block
        # if we get a function name since it signals a new tool call
        if block_type == "tool_use":
            from typing import cast

            from litellm.types.llms.anthropic import ToolUseBlock

            tool_block = cast(ToolUseBlock, content_block_start)
            if tool_block.get("name"):
                self.current_content_block_type = block_type
                self.current_content_block_start = content_block_start
                return True

        return False
