import asyncio
from copy import deepcopy

import pytest

from litellm.llms.deepseek.chat.transformation import DeepSeekChatConfig


IMAGE_URLS = [
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB",
    "https://example.com/logo.png",
]


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("image_url", IMAGE_URLS)
@pytest.mark.parametrize("with_text", [False, True])
def test_preserves_images_in_message_history(is_async, image_url, with_text):
    content = [{"type": "text", "text": "Describe this image"}] if with_text else []
    content.extend(
        [
            {"type": "image_url", "image_url": {"url": image_url, "detail": "low"}},
            {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
        ]
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "Be concise"}]},
        {"role": "user", "content": content},
        {"role": "assistant", "content": "A logo", "reasoning_content": "Inspect it"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What "},
                {"type": "text", "text": "color?"},
            ],
        },
    ]
    expected_content = deepcopy(content)
    result = DeepSeekChatConfig()._transform_messages(
        messages, model="DeepSeek-V4.1-Flash", is_async=is_async
    )
    if is_async:
        result = asyncio.run(result)

    assert result[0]["content"] == "Be concise"
    assert result[1]["content"] == expected_content
    assert result[2] == {"role": "assistant", "content": "Inspect it\n\nA logo"}
    assert result[3]["content"] == "What color?"


@pytest.mark.parametrize("is_async", [False, True])
def test_reasoning_preserves_assistant_content_blocks_and_tool_calls(is_async):
    blocks = [
        {"type": "text", "text": "A logo"},
        {"type": "image_url", "image_url": {"url": IMAGE_URLS[0]}},
    ]
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "inspect", "arguments": "{}"},
        }
    ]
    messages = [
        {
            "role": "assistant",
            "content": deepcopy(blocks),
            "reasoning_content": "Inspect it",
            "thinking_blocks": [{"type": "thinking", "thinking": "Inspect it"}],
            "tool_calls": deepcopy(tool_calls),
        }
    ]
    result = DeepSeekChatConfig()._transform_messages(
        messages, model="DeepSeek-V4.1-Flash", is_async=is_async
    )
    if is_async:
        result = asyncio.run(result)

    assert result == [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Inspect it\n\n"}] + blocks,
            "tool_calls": tool_calls,
        }
    ]


@pytest.mark.parametrize("is_async", [False, True])
def test_preserves_empty_and_string_content(is_async):
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": None, "reasoning_content": "Thinking"},
        {"role": "user", "content": []},
    ]
    result = DeepSeekChatConfig()._transform_messages(
        messages, model="deepseek-chat", is_async=is_async
    )
    if is_async:
        result = asyncio.run(result)

    assert result == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Thinking"},
        {"role": "user", "content": []},
    ]
