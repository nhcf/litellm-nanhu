"""
Translates from OpenAI's `/v1/chat/completions` to DeepSeek's `/v1/chat/completions`
"""

from typing import Any, Coroutine, List, Literal, Optional, Tuple, Union, overload

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    handle_messages_with_content_list_to_str_conversion,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class DeepSeekChatConfig(OpenAIGPTConfig):
    def get_supported_openai_params(self, model: str) -> list:
        """
        DeepSeek reasoner models support thinking parameter.
        """
        params = super().get_supported_openai_params(model)
        params.extend(["thinking", "reasoning_effort"])
        return params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI params to DeepSeek params.

        Handles `thinking` and `reasoning_effort` parameters for DeepSeek reasoner models.
        DeepSeek supports `chat_template_kwargs` format for thinking mode:
        - chat_template_kwargs: {"reasoning_effort": "high", "thinking": true, "enable_thinking": true}

        Reference: https://api-docs.deepseek.com/guides/thinking_mode
        """
        # Let parent handle standard params first
        optional_params = super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )

        # Pop thinking/reasoning_effort from optional_params first (parent may have added them)
        # Then re-add only if valid for DeepSeek
        thinking_value = optional_params.pop("thinking", None)
        reasoning_effort = optional_params.pop("reasoning_effort", None)

        # Determine if thinking mode should be enabled and get reasoning_effort value
        enable_thinking = False
        final_reasoning_effort = None

        # Valid reasoning_effort values for chat_template_kwargs
        # Per DeepSeek official docs: low/medium are mapped to high for compatibility,
        # xhigh is mapped to max.
        valid_effort_values = {"low", "medium", "high", "max", "xhigh"}

        _EFFORT_NORMALIZE = {"low": "high", "medium": "high", "xhigh": "max"}

        # Handle thinking parameter
        if thinking_value is not None and isinstance(thinking_value, dict):
            thinking_type = thinking_value.get("type")
            if thinking_type in ("enabled", "adaptive"):
                # For adaptive thinking, check if we have a valid reasoning_effort
                if thinking_type == "adaptive":
                    # reasoning_effort should come from output_config.effort or reasoning_effort param
                    if reasoning_effort and reasoning_effort in valid_effort_values:
                        enable_thinking = True
                        final_reasoning_effort = reasoning_effort
                    elif reasoning_effort is None:
                        # Default to max if no reasoning_effort provided with adaptive
                        enable_thinking = True
                        final_reasoning_effort = "max"
                else:  # type == "enabled"
                    enable_thinking = True
                    final_reasoning_effort = (
                        reasoning_effort
                        if reasoning_effort in valid_effort_values
                        else "max"
                    )
            else:
                # Explicitly disabled, set thinking at top level to disable
                optional_params["thinking"] = thinking_value
                optional_params["chat_template_kwargs"] = {
                    "reasoning_effort": "none",
                    "thinking": False,
                    "enable_thinking": False,
                }

        # Handle reasoning_effort alone (without thinking param)
        elif reasoning_effort is not None and reasoning_effort != "none":
            # Check for dict type first (from OpenAI adapter) to avoid TypeError
            if isinstance(reasoning_effort, dict):
                # reasoning_effort might be a dict with "effort" key (from OpenAI adapter)
                effort_value = reasoning_effort.get("effort")
                if effort_value in valid_effort_values:
                    enable_thinking = True
                    final_reasoning_effort = effort_value
            elif reasoning_effort in valid_effort_values:
                enable_thinking = True
                final_reasoning_effort = reasoning_effort

        # Handle reasoning_effort="none" → explicitly disable thinking
        elif reasoning_effort == "none":
            if "extra_body" not in optional_params:
                optional_params["extra_body"] = {}
            optional_params["thinking"] = {"type": "disabled"}
            optional_params["chat_template_kwargs"] = {
                "reasoning_effort": "none",
                "thinking": False,
                "enable_thinking": False,
            }

        # Default: enable thinking with reasoning_effort="max" when neither
        # thinking nor reasoning_effort is specified, matching DeepSeek API default.
        elif thinking_value is None and reasoning_effort is None:
            enable_thinking = True
            final_reasoning_effort = "max"

        # Normalize reasoning_effort per DeepSeek official docs:
        # low/medium → high, xhigh → max
        if final_reasoning_effort and final_reasoning_effort in _EFFORT_NORMALIZE:
            final_reasoning_effort = _EFFORT_NORMALIZE[final_reasoning_effort]

        # Generate chat_template_kwargs for thinking mode
        if enable_thinking:
            optional_params["chat_template_kwargs"] = {
                "reasoning_effort": final_reasoning_effort,
                "thinking": True,
                "enable_thinking": True,
            }
            if "extra_body" not in optional_params:
                optional_params["extra_body"] = {}
            optional_params["thinking"] = {"type": "enabled"}

        return optional_params

    @overload
    def _transform_messages(
        self, messages: List[AllMessageValues], model: str, is_async: Literal[True]
    ) -> Coroutine[Any, Any, List[AllMessageValues]]: ...

    @overload
    def _transform_messages(
        self,
        messages: List[AllMessageValues],
        model: str,
        is_async: Literal[False] = False,
    ) -> List[AllMessageValues]: ...

    def _transform_messages(
        self, messages: List[AllMessageValues], model: str, is_async: bool = False
    ) -> Union[List[AllMessageValues], Coroutine[Any, Any, List[AllMessageValues]]]:
        """
        DeepSeek does not support content in list format.
        Also merges reasoning_content into content for SGLang backends that
        do not natively process reasoning_content in input messages.
        """
        messages = handle_messages_with_content_list_to_str_conversion(messages)
        self._merge_reasoning_content_to_content(messages)
        if is_async:
            return super()._transform_messages(
                messages=messages, model=model, is_async=True
            )
        else:
            return super()._transform_messages(
                messages=messages, model=model, is_async=False
            )

    @staticmethod
    def _merge_reasoning_content_to_content(
        messages: List[AllMessageValues],
    ) -> None:
        """
        SGLang and some other DeepSeek-compatible backends ignore
        ``reasoning_content`` in input messages when constructing the prompt.
        This causes tool-call replay rounds to lose prior thinking context.

        Workaround: prepend ``reasoning_content`` into ``content`` so the
        model can see historical thinking text, then remove the
        ``reasoning_content`` and ``thinking_blocks`` fields so they don't
        leak to backends that reject unknown keys.
        """
        for message in messages:
            if message.get("role") != "assistant":
                continue
            reasoning = message.pop("reasoning_content", None)
            message.pop("thinking_blocks", None)
            if not reasoning:
                continue
            existing_content = message.get("content")
            if existing_content:
                message["content"] = reasoning + "\n\n" + str(existing_content)
            else:
                message["content"] = reasoning

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = (
            api_base
            or get_secret_str("DEEPSEEK_API_BASE")
            or "https://api.deepseek.com/beta"
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("DEEPSEEK_API_KEY")
        return api_base, dynamic_api_key

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        If api_base is not provided, use the default DeepSeek /chat/completions endpoint.
        """
        if not api_base:
            api_base = "https://api.deepseek.com/beta"

        if not api_base.endswith("/chat/completions"):
            api_base = f"{api_base}/chat/completions"

        return api_base
