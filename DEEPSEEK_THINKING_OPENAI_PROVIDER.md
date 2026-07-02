# DeepSeek V4 Thinking 参数支持 - OpenAI Provider 修改方案

## 问题描述

当前 LiteLLM 中，DeepSeek-V4-Flash 和 DeepSeek-V4-Pro 模型配置：
- `litellm_provider="openai"`（数据库配置）
- 不支持 `thinking` 和 `reasoning_effort` 参数

导致 `/v1/messages` 接口请求时，thinking 参数无法被正确处理。

## 解决方案

**保持 `litellm_provider="openai"` 不变**，在 OpenAIGPTConfig 中添加 DeepSeek V4 模型的特殊处理。

### 修改内容

#### 1. `litellm/llms/openai/chat/gpt_transformation.py`

**修改位置：**
- Line 137-187: `get_supported_openai_params()` - 添加 thinking 参数支持
- Line 213-225: `map_openai_params()` - 添加 thinking 参数处理
- Line 797+: 新增 `_is_deepseek_v4_model()` 和 `_handle_deepseek_thinking_params()` 方法

**核心逻辑：**
```python
# 1. 检测 DeepSeek V4 模型
def _is_deepseek_v4_model(model: str) -> bool:
    model_lower = model.lower()
    return any([
        "deepseek-v4" in model_lower,
        "deepseek-v3.2" in model_lower,
        ...
    ])

# 2. 为 DeepSeek 模型添加 thinking 参数支持
def get_supported_openai_params(model: str) -> list:
    if self._is_deepseek_v4_model(model):
        base_params.extend(["thinking", "reasoning_effort"])
    ...

# 3. 转换 thinking 参数为 chat_template_kwargs
def map_openai_params(...):
    if self._is_deepseek_v4_model(model):
        optional_params = self._handle_deepseek_thinking_params(...)
    ...

# 4. 生成 chat_template_kwargs（复用 DeepSeekChatConfig 逻辑）
def _handle_deepseek_thinking_params(...):
    # thinking={"type": "enabled"} →
    # extra_body.chat_template_kwargs = {
    #     "reasoning_effort": "high",
    #     "thinking": True,
    #     "enable_thinking": True
    # }
    ...
```

#### 2. 测试文件更新

**文件：** `tests/litellm/llms/deepseek/chat/test_deepseek_chat_transformation.py`

**新增测试类：** `TestOpenAIProviderDeepSeekThinking`
- 测试模型检测逻辑
- 测试 thinking 参数转换
- 测试 reasoning_effort 参数处理
- 测试非 DeepSeek 模型不处理 thinking

### 测试结果

```
✅ 所有测试通过
✓ DeepSeek model detection working correctly
✓ Thinking params added for DeepSeek models only
✓ thinking={'type': 'enabled'} generates correct chat_template_kwargs
✓ reasoning_effort='max' generates correct chat_template_kwargs
✓ adaptive thinking with reasoning_effort working
✓ Non-DeepSeek models don't get thinking processing
```

## 部署步骤

### 1. 代码部署

```bash
cd /data/home/cxd/2025/github/litellm-nanhu
git add litellm/llms/openai/chat/gpt_transformation.py
git add tests/litellm/llms/deepseek/chat/test_deepseek_chat_transformation.py
git commit -m "feat: Add DeepSeek V4 thinking parameter support for OpenAI provider"
git push origin fix/deepseek-thinking-maas
```

### 2. 重新部署 LiteLLM 服务

```bash
# 在 K8s litellm-system namespace 中重新部署
kubectl rollout restart deployment/litellm -n litellm-system
```

### 3. 验证修改

测试请求（使用数据库中的模型名称）：

```bash
curl "http://i-nhi.zhejianglab.org/maas/v1/messages" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1251529333842190336" \
  -d '{
    "model": "DeepSeek-V4-Flash",
    "messages": [{"role": "user", "content": [{"type": "text", "text": "你好"}]}],
    "thinking": {"type": "adaptive"},
    "max_tokens": 32000
  }'
```

预期结果：
```json
{
  "content": [
    {
      "type": "thinking",
      "thinking": "...思考内容..."
    },
    {
      "type": "text",
      "text": "...回复内容..."
    }
  ]
}
```

## 支持的参数格式

### 1. thinking 参数

```json
{
  "thinking": {"type": "enabled"}  // 启用思考模式
}
```

```json
{
  "thinking": {"type": "adaptive"}  // 自适应思考模式
}
```

### 2. reasoning_effort 参数

```json
{
  "reasoning_effort": "low"     // 低思考强度
}
```

```json
{
  "reasoning_effort": "medium"  // 中思考强度
}
```

```json
{
  "reasoning_effort": "high"    // 高思考强度（默认）
}
```

```json
{
  "reasoning_effort": "max"     // 最大思考强度
}
```

### 3. 组合使用

```json
{
  "thinking": {"type": "adaptive"},
  "reasoning_effort": "high"
}
```

## 内部转换逻辑

LiteLLM 会将参数转换为 DeepSeek API 要求的格式：

```json
{
  "extra_body": {
    "chat_template_kwargs": {
      "reasoning_effort": "high",
      "thinking": true,
      "enable_thinking": true
    },
    "thinking": {"type": "enabled"}  // 保留兼容性
  }
}
```

## 支持的模型

修改自动支持以下模型：
- `DeepSeek-V4-Flash`
- `DeepSeek-V4-Pro`
- `DeepSeek-V3.2-BF16-nodsa`
- 其他名称包含 "deepseek-v4" 或 "deepseek-v3.2" 的模型

## 优势

1. **零数据库改动** - 保持 `litellm_provider="openai"`
2. **自动生效** - 所有 DeepSeek V4 模型自动获得 thinking 支持
3. **向后兼容** - 不影响其他 OpenAI 模型
4. **集中维护** - DeepSeek 特殊逻辑集中一处
5. **完整测试覆盖** - 包含模型检测、参数转换、边界情况测试

## 相关文件

- 主代码：`litellm/llms/openai/chat/gpt_transformation.py`
- 测试：`tests/litellm/llms/deepseek/chat/test_deepseek_chat_transformation.py`
- DeepSeek 逻辑参考：`litellm/llms/deepseek/chat/transformation.py`