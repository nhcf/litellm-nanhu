"""
Unit tests for DeepSeek chat transformation.

Tests the thinking and reasoning_effort parameter handling for DeepSeek models.
"""

import pytest
from litellm.llms.deepseek.chat.transformation import DeepSeekChatConfig


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

        assert result["extra_body"]["chat_template_kwargs"]["reasoning_effort"] == "medium"
        assert result["extra_body"]["chat_template_kwargs"]["thinking"] is True

    def test_map_reasoning_effort_low(self):
        """Test that reasoning_effort='low' generates correct chat_template_kwargs."""
        non_default_params = {"reasoning_effort": "low"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["extra_body"]["chat_template_kwargs"]["reasoning_effort"] == "low"
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

        assert "thinking" not in result

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

        assert result["extra_body"]["chat_template_kwargs"]["reasoning_effort"] == "low"
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

        assert (
            result["extra_body"]["chat_template_kwargs"]["reasoning_effort"] == "medium"
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

        assert "chat_template_kwargs" not in result.get("extra_body", {})
        assert "thinking" not in result.get("extra_body", {})

    def test_thinking_disabled_no_chat_template_kwargs(self):
        """Test that thinking={"type": "disabled"} does not generate chat_template_kwargs."""
        non_default_params = {"thinking": {"type": "disabled"}}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert "chat_template_kwargs" not in result.get("extra_body", {})
        assert "thinking" not in result.get("extra_body", {})
