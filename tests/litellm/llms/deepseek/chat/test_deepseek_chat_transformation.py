"""
Unit tests for DeepSeek chat transformation.

Tests the thinking and reasoning_effort parameter handling for DeepSeek models.
"""

from copy import deepcopy
from typing import Any, Dict, List

import pytest
from litellm.llms.deepseek.chat.transformation import DeepSeekChatConfig


class TestMergeReasoningContentToContent:
    """Tests for _merge_reasoning_content_to_content static method.

    This method ensures SGLang and similar backends that ignore
    ``reasoning_content`` in input messages can still render historical
    thinking during tool-call replay rounds.
    """

    def test_merges_reasoning_content_into_content_string(self):
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "tool call text",
                "reasoning_content": "private thinking here",
                "thinking_blocks": [{"type": "thinking", "thinking": "private thinking here"}],
            },
        ]
        original_user_msg = deepcopy(messages[0])

        DeepSeekChatConfig._merge_reasoning_content_to_content(messages)

        assert messages[0] == original_user_msg
        assert messages[1]["content"] == "private thinking here\n\ntool call text"
        assert "reasoning_content" not in messages[1]
        assert "thinking_blocks" not in messages[1]

    def test_merges_reasoning_content_when_no_existing_content(self):
        messages: List[Dict[str, Any]] = [
            {
                "role": "assistant",
                "reasoning_content": "only thinking",
                "thinking_blocks": [{"type": "thinking", "thinking": "only thinking"}],
            },
        ]

        DeepSeekChatConfig._merge_reasoning_content_to_content(messages)

        assert messages[0]["content"] == "only thinking"
        assert "reasoning_content" not in messages[0]

    def test_skips_non_assistant_messages(self):
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": "question", "reasoning_content": "should be kept"},
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "assistant thinking",
            },
        ]

        DeepSeekChatConfig._merge_reasoning_content_to_content(messages)

        # Non-assistant messages are untouched
        assert messages[0] == {"role": "user", "content": "question", "reasoning_content": "should be kept"}
        assert messages[1]["content"] == "assistant thinking\n\nanswer"
        assert "reasoning_content" not in messages[1]

    def test_skips_messages_without_reasoning_content(self):
        messages: List[Dict[str, Any]] = [
            {"role": "assistant", "content": "plain answer"},
        ]

        DeepSeekChatConfig._merge_reasoning_content_to_content(messages)

        assert messages[0] == {"role": "assistant", "content": "plain answer"}

    def test_removes_thinking_blocks_even_without_reasoning_content(self):
        messages: List[Dict[str, Any]] = [
            {
                "role": "assistant",
                "content": "answer",
                "thinking_blocks": [{"type": "thinking", "thinking": "ignored"}],
            },
        ]

        DeepSeekChatConfig._merge_reasoning_content_to_content(messages)

        assert "thinking_blocks" not in messages[0]
        assert messages[0]["content"] == "answer"

    def test_empty_string_reasoning_content_not_merged(self):
        messages: List[Dict[str, Any]] = [
            {"role": "assistant", "content": "answer", "reasoning_content": ""},
        ]

        DeepSeekChatConfig._merge_reasoning_content_to_content(messages)

        assert messages[0]["content"] == "answer"
        assert "reasoning_content" not in messages[0]

    def test_tool_call_replay_scenario(self):
        """Simulate the exact scenario from BUG_REPORT.md: a tool-call round
        where the assistant message has thinking + tool_use."""
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": "Think privately, then call the note tool with x=hello."},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "REF-TEST-1234. Now I will call the note tool."},
                    {"type": "tool_use", "id": "toolu_probe", "name": "note", "input": {"x": "hello"}},
                ],
                "reasoning_content": "REF-TEST-1234. Now I will call the note tool.",
                "thinking_blocks": [
                    {"type": "thinking", "thinking": "REF-TEST-1234. Now I will call the note tool."},
                ],
            },
        ]

        DeepSeekChatConfig._merge_reasoning_content_to_content(messages)

        assert "reasoning_content" not in messages[1]
        assert "thinking_blocks" not in messages[1]
        assert "REF-TEST-1234" in messages[1]["content"]

    def test_preserves_system_messages(self):
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "query"},
            {
                "role": "assistant",
                "content": "result",
                "reasoning_content": "thinking...",
            },
        ]

        DeepSeekChatConfig._merge_reasoning_content_to_content(messages)

        assert messages[0] == {"role": "system", "content": "You are a helpful assistant."}
        assert messages[1] == {"role": "user", "content": "query"}
        assert messages[2]["content"] == "thinking...\n\nresult"


class TestDeepSeekThinkingParams:
    """Test thinking and reasoning_effort parameter handling for DeepSeek."""

    def setup_method(self):
        self.config = DeepSeekChatConfig()
        self.model = "deepseek-reasoner"

    def test_get_supported_openai_params_includes_thinking(self):
        """Test that thinking and reasoning_effort are in supported params."""
        params = self.config.get_supported_openai_params(self.model)
        assert "thinking" in params
        assert "reasoning_effort" in params
        assert "chat_template_kwargs" in params

    def test_map_thinking_enabled(self):
        """Test that thinking={"type": "enabled"} generates correct chat_template_kwargs."""
        non_default_params = {"thinking": {"type": "enabled"}}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert "chat_template_kwargs" in result["extra_body"]
        assert result["extra_body"]["chat_template_kwargs"]["thinking"] is True
        assert result["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True

    def test_map_thinking_with_budget_tokens_strips_budget(self):
        """Test that budget_tokens is stripped from thinking param (DeepSeek doesn't support it)."""
        non_default_params = {"thinking": {"type": "enabled", "budget_tokens": 2048}}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        # Should strip budget_tokens and generate chat_template_kwargs
        assert "chat_template_kwargs" in result["extra_body"]
        assert result["extra_body"]["chat_template_kwargs"]["thinking"] is True
        assert "budget_tokens" not in result.get("extra_body", {})

    def test_map_reasoning_effort_medium(self):
        """Test that reasoning_effort='medium' generates correct chat_template_kwargs."""
        non_default_params = {"reasoning_effort": "medium"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        # Per DeepSeek docs: medium normalizes to high
        assert result["extra_body"]["chat_template_kwargs"]["reasoning_effort"] == "high"
        assert result["extra_body"]["chat_template_kwargs"]["thinking"] is True

    def test_map_reasoning_effort_low(self):
        """Test that reasoning_effort='low' normalizes to 'high' per DeepSeek docs."""
        non_default_params = {"reasoning_effort": "low"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["extra_body"]["chat_template_kwargs"]["reasoning_effort"] == "high"
        assert result["extra_body"]["chat_template_kwargs"]["thinking"] is True

    def test_map_reasoning_effort_high(self):
        """Test that reasoning_effort='high' generates correct chat_template_kwargs."""
        non_default_params = {"reasoning_effort": "high"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["extra_body"]["chat_template_kwargs"]["reasoning_effort"] == "high"
        assert result["extra_body"]["chat_template_kwargs"]["thinking"] is True

    def test_map_reasoning_effort_none_does_not_enable_thinking(self):
        """Test that reasoning_effort='none' does not enable thinking."""
        non_default_params = {"reasoning_effort": "none"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["extra_body"]["thinking"] == {"type": "disabled"}

    def test_thinking_preserves_existing_extra_body(self):
        """Test that thinking is merged into existing extra_body without overwriting."""
        non_default_params = {"thinking": {"type": "enabled"}}
        optional_params = {"extra_body": {"existing_key": "existing_value"}}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["extra_body"]["chat_template_kwargs"]["thinking"] is True
        assert result["extra_body"]["existing_key"] == "existing_value"

    def test_reasoning_effort_preserves_existing_extra_body(self):
        """Test that reasoning_effort merges into existing extra_body without overwriting."""
        non_default_params = {"reasoning_effort": "medium"}
        optional_params = {"extra_body": {"existing_key": "existing_value"}}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["extra_body"]["chat_template_kwargs"]["thinking"] is True
        assert result["extra_body"]["existing_key"] == "existing_value"

    # New tests for chat_template_kwargs format
    def test_chat_template_kwargs_with_reasoning_effort_high(self):
        """Test that reasoning_effort='high' generates correct chat_template_kwargs."""
        non_default_params = {"reasoning_effort": "high"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert "chat_template_kwargs" in result["extra_body"]
        assert (
            result["extra_body"]["chat_template_kwargs"]["reasoning_effort"] == "high"
        )
        assert result["extra_body"]["chat_template_kwargs"]["thinking"] is True
        assert result["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
        # Legacy thinking param should also be present
        assert result["extra_body"]["thinking"] == {"type": "enabled"}

    def test_chat_template_kwargs_with_reasoning_effort_max(self):
        """Test that reasoning_effort='max' generates correct chat_template_kwargs."""
        non_default_params = {"reasoning_effort": "max"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["extra_body"]["chat_template_kwargs"]["reasoning_effort"] == "max"
        assert result["extra_body"]["chat_template_kwargs"]["thinking"] is True
        assert result["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True

    def test_chat_template_kwargs_with_thinking_adaptive(self):
        """Test that thinking={"type": "adaptive"} generates correct chat_template_kwargs."""
        non_default_params = {"thinking": {"type": "adaptive"}}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        # Should default to high for adaptive without reasoning_effort
        # (/v1/messages agent requests default to max via the adapter path)
        assert (
            result["extra_body"]["chat_template_kwargs"]["reasoning_effort"] == "high"
        )
        assert result["extra_body"]["chat_template_kwargs"]["thinking"] is True
        assert result["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True

    def test_chat_template_kwargs_with_thinking_adaptive_and_effort(self):
        """Test that thinking={"type": "adaptive"} with reasoning_effort generates correct chat_template_kwargs."""
        non_default_params = {
            "thinking": {"type": "adaptive"},
            "reasoning_effort": "low",
        }
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        # low normalizes to high per DeepSeek docs
        assert result["extra_body"]["chat_template_kwargs"]["reasoning_effort"] == "high"
        assert result["extra_body"]["chat_template_kwargs"]["thinking"] is True
        assert result["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True

    def test_chat_template_kwargs_with_thinking_enabled(self):
        """Test that thinking={"type": "enabled"} generates correct chat_template_kwargs."""
        non_default_params = {"thinking": {"type": "enabled"}}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        # Should default to high for enabled without reasoning_effort
        # (/v1/messages agent requests default to max via the adapter path)
        assert (
            result["extra_body"]["chat_template_kwargs"]["reasoning_effort"] == "high"
        )
        assert result["extra_body"]["chat_template_kwargs"]["thinking"] is True
        assert result["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True

    def test_chat_template_kwargs_with_thinking_enabled_and_effort(self):
        """Test thinking={"type": "enabled"} with reasoning_effort='medium'."""
        non_default_params = {
            "thinking": {"type": "enabled"},
            "reasoning_effort": "medium",
        }
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        # medium normalizes to high per DeepSeek docs
        assert (
            result["extra_body"]["chat_template_kwargs"]["reasoning_effort"] == "high"
        )
        assert result["extra_body"]["chat_template_kwargs"]["thinking"] is True

    def test_reasoning_effort_dict_with_effort_key(self):
        """Test reasoning_effort as dict with 'effort' key (from OpenAI adapter)."""
        non_default_params = {"reasoning_effort": {"effort": "high"}}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert (
            result["extra_body"]["chat_template_kwargs"]["reasoning_effort"] == "high"
        )
        assert result["extra_body"]["chat_template_kwargs"]["thinking"] is True

    def test_invalid_reasoning_effort_value_no_thinking(self):
        """Test that invalid reasoning_effort value does not enable thinking."""
        non_default_params = {"reasoning_effort": "invalid_value"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert "chat_template_kwargs" not in result
        assert "thinking" not in result

    def test_default_no_thinking_param_enables_thinking_with_high_effort(self):
        """Test that when no thinking or reasoning_effort is specified,
        thinking is enabled by default with reasoning_effort='high'.
        /v1/messages agent requests default to 'max' via the adapter path."""
        non_default_params = {}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert "chat_template_kwargs" in result["extra_body"]
        assert result["extra_body"]["chat_template_kwargs"]["reasoning_effort"] == "high"
        assert result["extra_body"]["chat_template_kwargs"]["thinking"] is True
        assert result["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
        assert result["extra_body"]["thinking"] == {"type": "enabled"}

    def test_thinking_disabled_passes_through_to_backend(self):
        """Test that thinking={"type": "disabled"} passes through to extra_body
        with chat_template_kwargs explicitly disabling thinking."""
        non_default_params = {"thinking": {"type": "disabled"}}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["extra_body"]["thinking"] == {"type": "disabled"}
        assert result["extra_body"]["chat_template_kwargs"]["reasoning_effort"] == "none"
        assert result["extra_body"]["chat_template_kwargs"]["thinking"] is False
        assert result["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False

    def test_reasoning_effort_none_disables_thinking(self):
        """Test that reasoning_effort='none' explicitly disables thinking
        with chat_template_kwargs."""
        non_default_params = {"reasoning_effort": "none"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["extra_body"]["thinking"] == {"type": "disabled"}
        assert result["extra_body"]["chat_template_kwargs"]["reasoning_effort"] == "none"
        assert result["extra_body"]["chat_template_kwargs"]["thinking"] is False
        assert result["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False

    # Tests for chat_template_kwargs input parameter support

    def test_chat_template_kwargs_thinking_false_disables_thinking(self):
        """chat_template_kwargs with thinking=false should disable thinking."""
        non_default_params = {"chat_template_kwargs": {"thinking": False}}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["extra_body"]["chat_template_kwargs"] == {
            "reasoning_effort": "none",
            "thinking": False,
            "enable_thinking": False,
        }
        assert result["extra_body"]["thinking"] == {"type": "disabled"}

    def test_chat_template_kwargs_enable_thinking_false_disables_thinking(self):
        """chat_template_kwargs with enable_thinking=false should disable thinking."""
        non_default_params = {"chat_template_kwargs": {"enable_thinking": False}}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["extra_body"]["chat_template_kwargs"] == {
            "reasoning_effort": "none",
            "thinking": False,
            "enable_thinking": False,
        }
        assert result["extra_body"]["thinking"] == {"type": "disabled"}

    def test_chat_template_kwargs_with_thinking_false_overrides_effort(self):
        """chat_template_kwargs thinking=false takes precedence over reasoning_effort."""
        non_default_params = {
            "chat_template_kwargs": {"thinking": False},
            "reasoning_effort": "high",
        }
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["extra_body"]["chat_template_kwargs"]["thinking"] is False

    def test_chat_template_kwargs_user_values_merged(self):
        """User-provided chat_template_kwargs should be merged with generated defaults."""
        non_default_params = {"chat_template_kwargs": {"custom_field": "value"}}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["extra_body"]["chat_template_kwargs"]["custom_field"] == "value"
        assert result["extra_body"]["chat_template_kwargs"]["thinking"] is True
        assert result["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True

    def test_chat_template_kwargs_no_top_level_reasoning_effort(self):
        """Top-level reasoning_effort should NOT be set; only chat_template_kwargs."""
        non_default_params = {"reasoning_effort": "high"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        # Top-level reasoning_effort must NOT be present (SGLang backend
        # rejects max/xhigh values with HTTP 400).
        assert "reasoning_effort" not in result
        assert result["extra_body"]["chat_template_kwargs"]["reasoning_effort"] == "high"
