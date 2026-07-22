# KEP: 运管认证集成设计方案

> **版本**: v1.0  
> **日期**: 2026-06-30  
> **状态**: 已实现 (含待优化项)  
> **核心 Commit**: [`62ecab61`](https://github.com/BerriAI/litellm/commit/62ecab61f1f8e2cade32bf171efa1e726c612742) — 运管认证集成  
> **依赖 Commit**: [`38276fb4`](https://github.com/BerriAI/litellm/commit/38276fb40dccc912b3e95281465b7c0eb1c6ef3b) — Dockerfile 代理支持 + 流式断开检测

---

## 目录

1. [背景与动机](#1-背景与动机)
2. [LiteLLM API-Key 认证流程分析](#2-litellm-api-key-认证流程分析)
3. [Commit 62ecab61 改动详解](#3-commit-62ecab61-改动详解)
4. [运管认证兜底方案设计](#4-运管认证兜底方案设计)
5. [方案影响评估](#5-方案影响评估)
6. [与设计原则的差距分析 (待优化项)](#6-与设计原则的差距分析-待优化项)
7. [配置与部署](#7-配置与部署)

---

## 1. 背景与动机

### 1.1 问题描述

LiteLLM 推理网关中存在两套独立的 SK (Secret Key) 管理体系：

| 系统 | 管理方式 | 认证入口 |
|------|---------|---------|
| **LiteLLM 本地** | Virtual Key 机制，存储在 `LiteLLM_VerificationToken` 表 | 本地 DB / 缓存查询 |
| **运管系统** | 运管接口 `/apis/control-zjic/projectSk/check` | HTTP API 校验 |

双轨管理导致以下问题：

1. **数据一致性风险**：两套系统的 SK 状态、配额信息可能不同步（如运管禁用 SK 后 LiteLLM 仍可通过）
2. **重复管理开销**：需要在两个系统中分别维护 SK 和配额
3. **认证口径不统一**：可能出现 LiteLLM 通过但运管不通过（或反之）的情况

### 1.2 整体设计原则

为避免上述问题，整体业务边界重新划分：

| 职责 | 承担方 | 说明 |
|------|--------|------|
| **SK 认证校验** | **运管 (唯一入口)** | 每次请求校验 SK 存在、有效期内、配额足够 |
| **SK 本地同步** | LiteLLM | 同步到本地数据库，仅用于确认 SK 存在、方便查看请求历史 |
| **配额/时效** | LiteLLM 本地设为上限 | 实际额度由运管控制，本地不做限制 |
| **Team 管理** | LiteLLM | 按 SK 对应的 `projectId` 创建 Team，`team_alias = projectId` |

核心原则：**运管是认证校验的唯一入口**，LiteLLM 只是将认证通过的 SK 同步到本地以支持可观测性。

### 1.3 实体关系说明

**运管侧**：
- **Project**：项目，包含多个 SK，有独立的配额
- **SK**：密钥，属于某个 Project，可以属于多个 Project
- **配额共享**：同一 Project 内的多个 SK 共享该 Project 的配额

**LiteLLM 侧映射**：
- LiteLLM `Team` ↔ 运管 `Project`（一个 Team 对应一个 Project）
- LiteLLM `API Key` ↔ 运管 `SK`（一个 Key 对应一个 SK）
- 同一 Team 下的多个 Key 共享 Team 配额

**存在的设计冲突**（详见 §6.2 优化项 5）：
LiteLLM 中一个 Key 只能属于一个 Team（1:1），而运管中一个 SK 可以属于多个 Project（1:N）。当前按 `projectId` 创建 Team 的方式，如果同一个 SK 被用于不同 Project 的请求，会导致 Team 归属冲突。

---

## 2. LiteLLM API-Key 认证流程分析

### 2.1 认证入口

LiteLLM 的 API-Key 认证入口是 [`_user_api_key_auth_builder()`](litellm/proxy/auth/user_api_key_auth.py:1280)，在 `litellm/proxy/auth/user_api_key_auth.py` 中定义。

### 2.2 完整认证链路

```
请求进入 (HTTP Request)
  │
  ▼
┌─ 1. 提取 API Key ─────────────────────────────────────────────┐
│  _get_bearer_token_or_received_api_key()                       │
│  处理 Bearer / Basic / AWS4-HMAC-SHA256 等多种格式             │
│  [litellm/proxy/auth/user_api_key_auth.py:164]                 │
└────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─ 2. 检查是否为 Master Key ─────────────────────────────────────┐
│  secrets.compare_digest(api_key, master_key)                   │
│  [litellm/proxy/auth/user_api_key_auth.py:1196]                │
│  └─ 是 → 返回 PROXY_ADMIN 角色，流程结束                       │
└────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─ 3. 检查仅缓存 ───────────────────────────────────────────────┐
│  get_key_object(check_cache_only=True)                         │
│  [litellm/proxy/auth/user_api_key_auth.py:1106]                │
│  └─ 命中 → 返回 UserAPIKeyAuth                                 │
└────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─ 4. 检查 UI Hash Key ──────────────────────────────────────────┐
│  ExperimentalUIJWTToken.get_key_object_from_ui_hash_key()      │
│  [litellm/proxy/auth/user_api_key_auth.py:1120]                │
│  └─ 命中 → 返回 UserAPIKeyAuth                                 │
└────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─ 5. 校验 sk- 前缀 ────────────────────────────────────────────┐
│  必须以 sk- 开头，否则直接返回 401                             │
│  [litellm/proxy/auth/user_api_key_auth.py:1271]                │
└────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─ 6. SHA-256 哈希 ─────────────────────────────────────────────┐
│  api_key = hash_token(api_key)                                 │
│  [litellm/proxy/auth/user_api_key_auth.py:1290]                │
└────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─ 7. 数据库查询 ───────────────────────────────────────────────┐
│  get_key_object(hashed_token, prisma_client, ...)              │
│  [litellm/proxy/auth/user_api_key_auth.py:1294]                │
│  ┌─ 先查缓存 ──→ 命中: 返回 UserAPIKeyAuth                    │
│  │  [litellm/proxy/auth/auth_checks.py:2364]                   │
│  └─ 查数据库 ──→ _fetch_key_object_from_db_with_reconnect()   │
│     [litellm/proxy/auth/auth_checks.py:2377]                   │
│     └─ 未找到 → 抛出 ProxyException(code=401)                  │
│        [litellm/proxy/auth/auth_checks.py:2385]                │
└────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─ 8. 【运管认证兜底】 ← Commit 62ecab61 新增 ───────────────────┐
│  捕获 ProxyException(code=401)                                 │
│  └─ yunguan_auth_fallback(api_key, model, ...)                 │
│     [litellm/proxy/auth/user_api_key_auth.py:1322]             │
│     ├─ 运管通过: 创建 Team + Key → 返回 UserAPIKeyAuth         │
│     └─ 运管失败: 重新抛出原始异常 (含运管错误码)               │
└────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─ 9. 认证后检查 ───────────────────────────────────────────────┐
│  · end_user_id / tpm_limit / rpm_limit 更新                   │
│  · 临时预算增加 (_update_key_budget_with_temp_budget_increase) │
│  · 模型权限检查 (can_key_call_model)                           │
│  · Token 过期检查                                              │
│  · 预算/配额检查 (max_budget, model_max_budget)                │
│  · Team 预算检查                                               │
└────────────────────────────────────────────────────────────────┘
```

### 2.3 关键数据结构: `UserAPIKeyAuth`

认证成功后的核心返回对象：

```python
# 定义在 litellm/proxy/_types.py
class UserAPIKeyAuth(BaseModel):
    api_key: str           # hashed token (SHA-256)
    token: str             # hashed token (alias)
    key_alias: str         # 原始 SK 名 (用于 UI 显示)
    user_id: str           # 关联用户 ID
    team_id: str           # 关联 Team UUID
    team_alias: str        # Team 别名 (运管场景: projectId)
    models: List[str]      # 允许的模型列表, [] = 所有
    user_role: str         # PROXY_ADMIN / INTERNAL_USER 等
    blocked: bool          # 是否被禁用
    max_budget: float      # 最大预算, None = 无限制
    expires: datetime      # 过期时间
    metadata: dict         # 自定义元数据
```

---

## 3. Commit 62ecab61 改动详解

### 3.1 改动概览

```
Dockerfile.yunguan-patch                |  24 ++   (新增)
litellm/proxy/auth/user_api_key_auth.py |  53 ++-  (修改)
litellm/proxy/auth/yunguan_auth.py      | 567 ++++ (新增)
─────────────────────────────────────────────────
3 files changed, 641 insertions(+), 3 deletions(-)
```

### 3.2 新增文件: [`yunguan_auth.py`](litellm/proxy/auth/yunguan_auth.py) (567 行)

运管认证模块，包含以下核心组件：

#### 3.2.1 配置类: `YunguanAuthConfig`

```python
@dataclass
class YunguanAuthConfig:
    base_url: str                                    # 运管基地址
    auth_path: str = "/apis/control-zjic/projectSk/check"  # 认证路径
    timeout: float = 10.0                            # 请求超时

    @classmethod
    def from_general_settings(cls, general_settings) -> Optional["YunguanAuthConfig"]:
        """从 config.yaml 的 general_settings 读取配置"""
```

通过 `config.yaml` 的 `general_settings` 传入：

```yaml
general_settings:
  enable_yunguan_auth: true
  yunguan_base_url: "https://zjic.zhejianglab.com"
  yunguan_auth_path: "/apis/control-zjic/projectSk/check"
  yunguan_timeout: 10.0
```

#### 3.2.2 响应解析: `YunguanAuthResponse`

解析运管接口返回的 JSON 数据，处理 ISO 8601 过期时间格式（含时区偏移 `+00:00`）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `sk` | str | 原始 SK 值 |
| `token_total` | float | 总配额 (tokens) |
| `expire_time` | datetime | 过期时间 |
| `project_sk_id` | int | 运管 SK ID |
| `sk_type` | str | SK 类型 (如 `PROJECT_CUSTOM`) |
| `project_id` | str | 关联项目 ID |
| `quota_used` | float | 已用配额 |
| `free_model` | bool | 是否免费模型 |

#### 3.2.3 运管接口调用: `verify_sk_with_yunguan()`

- 使用 [`httpx.AsyncClient`](litellm/proxy/auth/yunguan_auth.py:162) 发起 HTTP GET 请求
- 传递 `sk` 和 `modelCode` 参数
- 完整的错误处理：超时 (`YUNGUAN_TIMEOUT`)、HTTP 错误、JSON 解析错误、业务错误码（如 `PROJECT_SK_NOT_EXIST`）
- 日志中只打印 SK 前缀（`sk[:8]...`），不泄露完整密钥

#### 3.2.4 Team 管理: `get_or_create_team_for_yunguan()`

创建策略：

1. **缓存查找** → key: `yunguan_project_team:{projectId}`
2. **DB 查找** → `team_alias == projectId`
3. **不存在则创建** → 新 Team，属性：
   - `team_alias = projectId` (运管 projectId)
   - `models = []` (空列表 = 允许所有模型)
   - `metadata` 包含运管来源标记 (`yunguan_source: True`)
4. **并发冲突处理** → 捕获 `Unique constraint` 异常后重新查找

#### 3.2.5 API Key 管理: `create_or_get_key_for_yunguan()`

创建策略：

1. 计算 SK 的 SHA-256 哈希 → [`hash_token(sk)`](litellm/proxy/auth/yunguan_auth.py:34)
2. 通过 `get_key_object()` 检查是否已存在
3. 不存在则创建新 Key：
   - `token = hash(sk)` (SHA-256)
   - `key_alias = sk` (原始 SK，便于 UI 识别)
   - `models = []` (允许所有模型)
   - `max_budget = None` (不限制，由运管控制)
   - `expires = 运管返回的过期时间`
   - `metadata.yunguan_sk = True`
   - `user_id = LITELLM_PROXY_ADMIN_NAME`
4. 返回 `UserAPIKeyAuth` → `user_role = INTERNAL_USER`
5. **并发冲突处理** → 捕获 `Unique constraint` 异常后重新查找

#### 3.2.6 主编排函数: `yunguan_auth_fallback()`

组合上述步骤的编排函数，流程如下：

```
1. 检查配置是否启用运管认证
   └─ 未启用 → 返回 (None, "YUNGUAN_NOT_ENABLED")

2. 提取模型名称 (从 request_data/route 中)
   └─ model 支持 str | List[str]，取第一个

3. 调用运管接口校验
   └─ verify_sk_with_yunguan(sk, model_name, config)
   └─ 失败 → 返回 (None, error_code) 如 "PROJECT_SK_NOT_EXIST"

4. 获取 projectId
   └─ 为空 → 返回 (None, "YUNGUAN_NO_PROJECT_ID")

5. 获取或创建 Team
   └─ get_or_create_team_for_yunguan(project_id, ...)

6. 创建或获取 Key
   └─ create_or_get_key_for_yunguan(sk, team_id, ...)

7. 返回 (UserAPIKeyAuth, None)
```

### 3.3 修改文件: [`user_api_key_auth.py`](litellm/proxy/auth/user_api_key_auth.py) (+53/-3)

在 `_user_api_key_auth_builder()` 的 401 异常处理分支中插入运管认证兜底逻辑：

```python
# L1301: 原代码只抛出异常
except ProxyException as e:
    if e.code == 401 or e.code == "401":
        # ===== 运管认证兜底逻辑 (新增) =====
        # 保存原始 api_key (运管需要原始 sk 值，而非 hash)
        original_api_key_for_yunguan = api_key   # L1288

        # 获取模型名称
        model_for_yunguan = _get_model_from_request_context(...)

        # 调用运管认证
        yunguan_valid_token, yunguan_error_code = await yunguan_auth_fallback(
            api_key=original_api_key_for_yunguan,
            model=model_for_yunguan,
            general_settings=general_settings,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
            parent_otel_span=parent_otel_span,
        )

        if yunguan_valid_token is not None:
            valid_token = yunguan_valid_token   # 认证成功：替换认证对象
        else:
            raise e  # 认证失败：返回原始异常
    else:
        raise e
```

### 3.4 新增文件: [`Dockerfile.yunguan-patch`](Dockerfile.yunguan-patch) (24 行)

基于现有运行镜像打补丁的 Dockerfile，用于快速部署运管认证功能：

```dockerfile
FROM 10.200.93.79:15080/gyjc/litellm:v1.85.0-38276fb

# 复制补丁文件到 venv 安装路径
COPY litellm/proxy/auth/yunguan_auth.py \
     /app/.venv/lib/python3.13/site-packages/litellm/proxy/auth/yunguan_auth.py
COPY litellm/proxy/auth/user_api_key_auth.py \
     /app/.venv/lib/python3.13/site-packages/litellm/proxy/auth/user_api_key_auth.py
# 复制到备用路径
COPY litellm/proxy/auth/yunguan_auth.py /app/litellm/proxy/auth/yunguan_auth.py
COPY litellm/proxy/auth/user_api_key_auth.py /app/litellm/proxy/auth/user_api_key_auth.py

# 清理 .pyc 缓存
RUN find /app -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
# 验证补丁完整性
RUN grep -c "运管认证" /app/.venv/lib/python3.13/site-packages/litellm/proxy/auth/user_api_key_auth.py
```

---

## 4. 运管认证兜底方案设计

### 4.1 方案流程图

```
请求进入 → 提取 api_key → LiteLLM 本地 DB 查询
                              │
                    ┌─────────┴──────────┐
                    ▼                    ▼
              本地 Key 存在         本地 Key 不存在 (401)
                    │                    │
                    ▼                    ▼
            返回 UserAPIKeyAuth    yunguan_auth_fallback()
                                        │
                              ┌─────────┴──────────┐
                              ▼                    ▼
                        运管认证通过          运管认证失败
                              │                    │
                              ▼                    ▼
                    ┌─ 获取/创建 Team      返回原始 401
                    │  (team_alias=projectId)    + 运管错误码
                    │
                    ▼
                    ┌─ 创建/获取 API Key
                    │  (token = hash(sk))
                    │  (models = [] 所有模型)
                    │  (max_budget = None 不限)
                    │  (expires = 运管过期时间)
                    │
                    ▼
              返回 UserAPIKeyAuth
              (user_role = INTERNAL_USER)

后续请求: 本地 DB/缓存命中 → 直接返回，不再调运管
```

### 4.2 运管接口协议

**请求**:
```
GET /apis/control-zjic/projectSk/check?sk={sk}&modelCode={model_code}
Host: https://zjic.zhejianglab.com
Content-Type: application/json
Accept: */*
```

**成功响应**:
```json
{
    "code": "200",
    "message": "成功",
    "requestId": "a3b702cd-dee9-4fff-b6a7-d8c8af72b8d1",
    "data": {
        "sk": "sk-1232007228247220224",
        "tokenTotal": 2.00002E14,
        "expireTime": "2026-06-30T15:59:59.000+00:00",
        "projectSkId": 283,
        "skType": "PROJECT_CUSTOM",
        "projectId": "pr-1228421016223940608",
        "quotaUsed": 1563,
        "freeModel": false
    },
    "failed": false,
    "success": true
}
```

**错误码映射**:

| 运管 `code` | 含义 | LiteLLM 处理 |
|-------------|------|-------------|
| `200` | 认证成功 | 创建 Team + Key |
| `PROJECT_SK_NOT_EXIST` | SK 不存在 | 返回 401 + `ZJIC.PROJECT_SK_NOT_EXIST` |
| 其他业务错误码 | 各种业务错误 | 返回 401 + `ZJIC.{error_code}` |
| 网络超时 | 无法连接运管 | 返回 401 + `YUNGUAN_TIMEOUT` |
| 网络异常 | 连接异常 | 返回 401 + `YUNGUAN_EXCEPTION` |

### 4.3 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 认证顺序 | 本地优先 → 运管兜底 | 最小化代码改动，不影响现有正常认证流程 |
| SK 存储方式 | SHA-256 哈希 | 与 LiteLLM 标准一致，不存储明文 SK |
| 本地配额 | `max_budget = None` | 实际配额由运管接口控制 |
| 本地过期时间 | 同步运管 `expireTime` | 仅首次同步，后续不更新 (待优化) |
| Key 角色 | `INTERNAL_USER` | 不可访问管理 API |
| Team 别名 | `projectId` | 便于通过运管项目 ID 关联 |

---

## 5. 方案影响评估

### 5.1 对现有模块的影响

| 模块 | 影响程度 | 说明 |
|------|----------|------|
| [`user_api_key_auth.py`](litellm/proxy/auth/user_api_key_auth.py) | **低** | 仅在 401 异常处理块中增加运管兜底调用，不改变正常认证路径 |
| [`auth_checks.py`](litellm/proxy/auth/auth_checks.py) | **无** | 未修改，`get_key_object()` 等函数保持不变 |
| [`common_request_processing.py`](litellm/proxy/common_request_processing.py) | **无** | 运管认证不影响请求处理流程 |
| 数据库 Schema | **无** | 使用现有 `LiteLLM_VerificationToken` 和 `LiteLLM_TeamTable` 表 |
| UI Dashboard | **无** | 运管创建的 Key/Team 以标准方式存储，UI 可正常展示 |
| Admin API (`/key/generate` 等) | **无** | 运管 Key 为 `INTERNAL_USER` 角色，不受管理 API 影响 |

### 5.2 性能影响

| 场景 | 频率 | 影响 |
|------|------|------|
| **本地 Key 命中** (正常路径) | 99%+ | **无影响**：不触发运管接口调用 |
| **本地 Key 未命中 + 运管未启用** | 取决于配置 | **无影响**：`YunguanAuthConfig.from_general_settings()` 快速返回 `None` |
| **本地 Key 未命中 + 运管启用 (首次)** | 极低 | **+10s (最坏)**：运管接口调用 + Team/Key 创建 |
| **本地 Key 未命中 + 运管启用 (后续)** | 极低 | **< 1ms**：本地缓存/数据库命中 |

### 5.3 安全性评估

| 关注点 | 评估 |
|--------|------|
| **SK 传输安全** | 通过 HTTPS 传输到运管接口，与用户原始请求一致 |
| **日志安全** | 日志中只打印 `sk[:8]...` 前缀，不泄露完整密钥 |
| **权限控制** | 运管 Key 创建为 `INTERNAL_USER` 角色，无法访问管理 API |
| **密钥存储** | 使用 SHA-256 哈希存储 token，与 LiteLLM 标准一致，不存储明文 |
| **防重放** | Token 已哈希，数据库唯一键约束防止重复创建 |

### 5.4 边界场景处理

| 场景 | 处理方式 |
|------|----------|
| 运管接口超时 (10s) | 返回原始 401，不阻塞请求 |
| 运管接口返回非 200 HTTP 状态 | 尝试解析 JSON 中的业务 `code`，返回对应错误码 |
| 运管 `projectId` 为空 | 返回 `YUNGUAN_NO_PROJECT_ID` 错误 |
| 并发创建 Team/Key 冲突 | 捕获 `Unique constraint` 异常，重新查找已存在的记录 |
| `prisma_client` 为 `None` | 返回 `YUNGUAN_NO_DB` 错误 (无数据库连接) |
| 配置未启用 (`enable_yunguan_auth: false`) | `from_general_settings()` 返回 `None`，跳过兜底 |
| 运管返回的 `expireTime` 格式异常 | 记录 warning 日志，不设置过期时间 |

### 5.5 兼容性

| 维度 | 评估 |
|------|------|
| **向后兼容** | ✅ 完全兼容。不启用运管时行为不变 |
| **API 兼容** | ✅ 不修改任何 API 接口 |
| **数据库兼容** | ✅ 使用现有表结构，不新增字段 |
| **Python 版本** | ✅ 使用 `datetime.fromisoformat`，Python 3.7+ |

---

## 6. 与设计原则的差距分析 (待优化项)

### 6.1 设计原则回顾

> **运管是认证校验的唯一入口**，LiteLLM 只是将认证通过的 SK 同步到本地以支持可观测性。

### 6.2 待优化项详情

---

#### 优化项 1: 认证顺序不符合"运管优先"原则 🔴 **高优先级**

| 维度 | 当前实现 | 期望实现 |
|------|----------|----------|
| 认证顺序 | LiteLLM 本地 DB → 运管 (兜底) | **运管 → LiteLLM 本地 DB** |
| 首次请求 | 先查本地，找不到才调运管 | 先调运管，通过后同步到本地 |
| SK 本地命中后 | 直接通过，不再校验运管 | 仍需校验运管状态 (或异步校验) |

**问题本质**：如果运管中 SK 状态变更（如禁用、配额用完），而 LiteLLM 本地 Key 仍然存在且有效，则该 SK 仍可通过 LiteLLM 本地认证，绕过了运管的管控。

**建议方案**：
- 将运管认证前置到 LiteLLM 本地认证之前
- 本地 Key 仅用于请求历史关联（`metadata` 标记），缓存超时时间应短于运管状态变更窗口
- 或设计运管状态异步同步机制（定时任务同步运管 SK 状态到本地）

**预估工作量**：中（需重构认证流程中的认证顺序逻辑）

---

#### 优化项 2: 缺少运管不可用时的降级策略 🔴 **高优先级**

| 场景 | 当前行为 | 问题 |
|------|----------|------|
| 运管接口宕机/超时 | 回退到 LiteLLM 本地认证 (兜底模式) | 如果改为运管优先，需要明确的降级策略 |
| 运管网络不可达 | 同上 | 同上 |

**建议方案**：如果改为"运管优先"模式，需定义降级策略：
- 方案 A：运管不可用时使用本地缓存的认证结果（需设置合理的缓存有效期）
- 方案 B：运管不可用时直接拒绝请求（严格模式，更安全）
- 建议默认方案 A，通过配置开关支持方案 B

**预估工作量**：中

#### 优化项 3: SK 多 Project 归属与 LiteLLM Team 1:1 模型冲突 🔴 **高优先级**

**问题分析**：

运管系统中一个 SK 可以属于多个 Project（1:N 关系），但 LiteLLM 中一个 API Key 只能属于一个 Team（1:1 关系）。当前实现按 `projectId`（即运管返回的 SK 所属 Project）创建 Team，当同一个 SK 用于不同 Project 的请求时，会出现以下问题：

| 场景 | 问题 |
|------|------|
| SK-1 属于 Project-A 和 Project-B | 第一次请求（Project-A）创建 Key → Team-A；第二次请求（Project-B）时 Key 已存在，直接复用 Team-A，丢失 Project-B 的关联 |
| Project-A 和 Project-B 配额独立 | 两个 Project 的配额在 LiteLLM 中被错误地合并到同一个 Team，无法区分 |

**建议方案**：

**将所有通过运管认证的 SK 统一归属到一个指定的运管专用 Team**，所有 SK 的认证授权完全由运管接口负责，LiteLLM 本地不做 Project 级别的配额区分。

```
运管 SK-1 (Project-A, Project-B)
运管 SK-2 (Project-C)
       │
       ▼  所有运管 SK 统一归属
┌─────────────────────────────┐
│ 运管专用 Team                │
│ team_alias: "yunguan"       │
│ models: [] (所有模型)        │
│ max_budget: None (不限制)    │
│                             │
│  ├─ Key: hash(sk-1)         │
│  │    metadata: {           │
│  │      yunguan_sk: true,   │
│  │      yunguan_projects:   │
│  │        ["pr-A", "pr-B"]  │
│  │    }                     │
│  └─ Key: hash(sk-2)         │
│       metadata: {           │
│         yunguan_sk: true,   │
│         yunguan_projects:   │
│           ["pr-C"]          │
│       }                     │
└─────────────────────────────┘
       │
       ▼  所有认证校验走运管
     运管接口 (唯一认证入口)
```

**优势**：
- 彻底消除 SK 多 Project 归属的 1:N 冲突
- Team 管理简化，无需按 projectId 动态创建
- 所有配额控制回归运管，LiteLLM 本地不做配额区分
- 每个 SK 参与的 Project 列表记录在 `metadata.yunguan_projects` 中，便于可观测性

**改动点**：
1. 新增配置项 `yunguan_team_alias`（默认 `"yunguan"`），指定运管专用 Team 别名
2. `get_or_create_team_for_yunguan()` 改为创建/获取固定的运管专用 Team，不再按 projectId 动态创建
3. `create_or_get_key_for_yunguan()` 在 `metadata.yunguan_projects` 中追加当前请求的 projectId
4. 如果运管专用 Team 不存在则自动创建（首次启动），`models=[]`、`max_budget=None`

**预估工作量**：中（涉及 Team 创建逻辑重构、配置新增、数据迁移考虑）

---

### 6.3 待优化项汇总

| # | 优化项 | 优先级 | 工作量 |
|---|--------|--------|--------|
| 1 | 运管认证从"兜底"改为"优先认证" | 🔴 高 | 中 |
| 2 | 运管不可用时的降级策略设计 | 🔴 高 | 中 |
| 3 | SK 多 Project 归属冲突 → 统一运管专用 Team | 🔴 高 | 中 |

---

## 7. 配置与部署

### 7.1 配置项

```yaml
general_settings:
  # 启用运管认证 (默认 false)
  enable_yunguan_auth: true

  # 运管接口基地址
  yunguan_base_url: "https://zjic.zhejianglab.com"

  # 运管认证接口路径 (可选，默认值: /apis/control-zjic/projectSk/check)
  yunguan_auth_path: "/apis/control-zjic/projectSk/check"

  # 运管接口超时秒数 (可选，默认值: 10.0)
  yunguan_timeout: 10.0

  # 运管专用 Team 别名 (可选，默认值: "yunguan")
  # 所有通过运管认证的 SK 统一归属到此 Team，避免 SK 多 Project 归属冲突
  # 参见 §6.2 优化项 5
  yunguan_team_alias: "yunguan"
```

### 7.2 Docker 补丁部署

```bash
# 构建运管补丁镜像
docker build -f Dockerfile.yunguan-patch -t litellm:yunguan-patch .

# 运行 (挂载包含运管配置的 config.yaml)
docker run -p 4000:4000 \
  -v /path/to/config.yaml:/app/config.yaml \
  litellm:yunguan-patch
```

### 7.3 Helm Chart 部署

通过 [`deploy/charts/litellm-helm/values.yaml`](deploy/charts/litellm-helm/values.yaml) 的 `proxy_config` 段传入：

```yaml
proxy_config:
  general_settings:
    enable_yunguan_auth: true
    yunguan_base_url: "https://zjic.zhejianglab.com"
  # ... 其他 proxy 配置
```

### 7.4 验证方法

```bash
# 1. 使用运管 SK 发送推理请求
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-1232007228247220224" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-Coder-Next-FP8",
    "messages": [{"role": "user", "content": "hello"}]
  }'

# 2. 预期行为:
#    - 首次请求: 触发运管认证 → 自动创建 Team + Key → 返回推理结果
#    - 后续请求: 命中本地缓存/DB → 直接返回推理结果

# 3. 在 LiteLLM Admin UI 中检查:
#    - Teams 列表出现新 Team (alias = 运管 projectId)
#    - Keys 列表出现新 Key (alias = 运管 sk 值, metadata.yunguan_sk = true)
```

---

> **相关文件索引**：
> - 设计文档: [`docs/kep/zjic_auth_integration_design.md`](docs/kep/zjic_auth_integration_design.md)
> - 运管认证模块: [`litellm/proxy/auth/yunguan_auth.py`](litellm/proxy/auth/yunguan_auth.py)
> - 认证入口: [`litellm/proxy/auth/user_api_key_auth.py`](litellm/proxy/auth/user_api_key_auth.py) (L1280-L1351)
> - 底层认证查询: [`litellm/proxy/auth/auth_checks.py`](litellm/proxy/auth/auth_checks.py) (L2340-L2419)
> - Docker 补丁: [`Dockerfile.yunguan-patch`](Dockerfile.yunguan-patch)
> - 示例配置: `/tmp/yunguan_config.yaml`
