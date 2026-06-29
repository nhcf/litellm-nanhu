"""
运管认证模块

当 LiteLLM 标准认证失败时，通过运管接口校验 SK 和模型名称。
如果运管认证通过则自动创建/复用 Team 和 API Key，使运管系统的密钥能够无缝接入 LiteLLM。

配置方式（config.yaml 的 general_settings）:
    enable_yunguan_auth: true
    yunguan_base_url: "https://zjic.zhejianglab.com"
    yunguan_auth_path: "/apis/control-zjic/projectSk/check"
"""

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import httpx

from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.constants import LITELLM_PROXY_ADMIN_NAME
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import _cache_key_object, get_key_object
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient, ProxyLogging
else:
    PrismaClient = Any
    ProxyLogging = Any


def hash_token(token: str) -> str:
    """计算 token 的 SHA-256 哈希值"""
    import hashlib
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@dataclass
class YunguanAuthConfig:
    """运管认证配置"""
    base_url: str
    auth_path: str = "/apis/control-zjic/projectSk/check"
    timeout: float = 10.0

    @classmethod
    def from_general_settings(
        cls, general_settings: Dict[str, Any]
    ) -> Optional["YunguanAuthConfig"]:
        """从 general_settings 创建配置"""
        if not general_settings.get("enable_yunguan_auth", False):
            return None

        base_url = general_settings.get("yunguan_base_url")
        if not base_url:
            verbose_proxy_logger.warning(
                "运管认证已启用但 yunguan_base_url 未配置"
            )
            return None

        return cls(
            base_url=base_url,
            auth_path=general_settings.get(
                "yunguan_auth_path", "/apis/control-zjic/projectSk/check"
            ),
            timeout=general_settings.get("yunguan_timeout", 10.0),
        )


@dataclass
class YunguanAuthResponse:
    """运管认证返回数据结构"""
    sk: str
    token_total: float
    expire_time: Optional[datetime]
    project_sk_id: int
    sk_type: str
    project_id: str
    quota_used: float
    free_model: bool
    success: bool = True

    @classmethod
    @classmethod
    def from_api_response(
        cls, data: Dict[str, Any]
    ) -> tuple[Optional["YunguanAuthResponse"], Optional[str]]:
        """
        解析运管接口返回数据
        
        Returns:
            tuple: (YunguanAuthResponse 或 None, error_code 或 None)
            - 成功时: (YunguanAuthResponse, None)
            - 失败时: (None, error_code) 如 "PROJECT_SK_NOT_EXIST"
        """
        if data.get("code") != "200" or not data.get("success", False):
            error_code = data.get("code", "UNKNOWN_ERROR")
            verbose_proxy_logger.warning(
                f"运管认证失败: {data.get('message', '未知错误')}, code={error_code}"
            )
            return None, error_code

        result_data = data.get("data", {})
        if not result_data:
            return None

        # 解析过期时间
        expire_time_str = result_data.get("expireTime")
        expire_time = None
        if expire_time_str:
            try:
                # 处理 ISO 8601 格式，如 "2026-06-30T15:59:59.000+00:00"
                # Python 3.7+ fromisoformat 不支持时区偏移，需要处理
                expire_time_str_clean = expire_time_str.replace("+00:00", "").replace("Z", "")
                expire_time = datetime.fromisoformat(expire_time_str_clean)
                expire_time = expire_time.replace(tzinfo=timezone.utc)
            except Exception as e:
                verbose_proxy_logger.warning(
                    f"解析运管过期时间失败: {expire_time_str}, error={e}"
                )

        return cls(
            sk=result_data.get("sk", ""),
            token_total=float(result_data.get("tokenTotal", 0)),
            expire_time=expire_time,
            project_sk_id=result_data.get("projectSkId", 0),
            sk_type=result_data.get("skType", ""),
            project_id=result_data.get("projectId", ""),
            quota_used=float(result_data.get("quotaUsed", 0)),
            free_model=result_data.get("freeModel", False),
            success=True,
        ), None


async def verify_sk_with_yunguan(
    sk: str,
    model_name: Optional[str],
    config: YunguanAuthConfig,
) -> tuple[Optional[YunguanAuthResponse], Optional[str]]:
    """
    调用运管接口校验 SK

    Args:
        sk: API Key（原始值，如 sk-xxx）
        model_name: 模型名称（从请求中提取）
        config: 运管配置

    Returns:
        tuple: (YunguanAuthResponse 或 None, error_code 或 None)
    """
    url = f"{config.base_url.rstrip('/')}{config.auth_path}"
    params = {"sk": sk}
    if model_name:
        params["modelCode"] = model_name

    # 仅打印 sk 的前缀用于调试，避免泄露完整密钥
    sk_preview = sk[:8] + "..." if len(sk) > 8 else sk
    verbose_proxy_logger.info(
        f"调用运管接口认证: url={url}, sk={sk_preview}, model={model_name}"
    )

    async with httpx.AsyncClient(timeout=config.timeout) as client:
        try:
            response = await client.get(
                url,
                params=params,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "*/*",
                },
            )
            
            # 记录响应状态和内容用于调试
            verbose_proxy_logger.debug(
                f"运管接口响应: status={response.status_code}, url={url}"
            )
            
            # 尝试解析 JSON 响应，即使 HTTP 状态码不是 200
            try:
                data = response.json()
                verbose_proxy_logger.debug(
                    f"运管接口响应内容: code={data.get('code')}, success={data.get('success')}"
                )
                
                # 如果运管返回了业务错误码，使用它而不是 HTTP 错误码
                if data.get("code") and data.get("code") != "200":
                    error_code = data.get("code")
                    verbose_proxy_logger.warning(
                        f"运管认证业务失败: code={error_code}, message={data.get('message')}"
                    )
                    return None, error_code
                    
                # HTTP 状态码正常且业务成功，解析认证数据
                return YunguanAuthResponse.from_api_response(data)
                
            except Exception as json_error:
                verbose_proxy_logger.error(
                    f"运管接口响应解析失败: status={response.status_code}, error={json_error}"
                )
                return None, "YUNGUAN_PARSE_ERROR"

        except httpx.TimeoutException:
            verbose_proxy_logger.error(
                f"运管接口超时: url={url}, timeout={config.timeout}s"
            )
            return None, "YUNGUAN_TIMEOUT"
        except httpx.HTTPStatusError as e:
            # HTTP 状态码错误时，尝试解析响应内容获取业务错误码
            verbose_proxy_logger.error(
                f"运管接口 HTTP 错误: status={e.response.status_code}, url={url}"
            )
            try:
                data = e.response.json()
                if data.get("code"):
                    return None, data.get("code")
            except Exception:
                pass
            return None, "YUNGUAN_HTTP_ERROR"
        except Exception as e:
            verbose_proxy_logger.error(f"运管接口异常: {e}")
            return None, "YUNGUAN_EXCEPTION"


async def get_or_create_team_for_yunguan(
    project_id: str,
    yunguan_response: YunguanAuthResponse,
    prisma_client: PrismaClient,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: Optional[ProxyLogging] = None,
) -> str:
    """
    根据 projectId 获取或创建 Team

    Args:
        project_id: 运管返回的 projectId
        yunguan_response: 运管认证响应数据
        prisma_client: Prisma 客户端
        user_api_key_cache: 缓存对象
        proxy_logging_obj: 日志对象

    Returns:
        team_id: UUID 格式的团队 ID
    """
    # 1. 尝试缓存查找（根据 projectId 映射）
    cache_key = f"yunguan_project_team:{project_id}"
    cached_team_id = await user_api_key_cache.async_get_cache(cache_key)
    if cached_team_id and isinstance(cached_team_id, str):
        verbose_proxy_logger.info(
            f"命中运管 Team 缓存: project={project_id}, team={cached_team_id}"
        )
        return cached_team_id

    # 2. 通过数据库查找（通过 team_alias 匹配 projectId）
    existing_team = await prisma_client.db.litellm_teamtable.find_first(
        where={"team_alias": project_id}
    )

    if existing_team:
        team_id = str(existing_team.team_id)
        await user_api_key_cache.async_set_cache(cache_key, team_id)
        verbose_proxy_logger.info(
            f"查找到现有 Team: project={project_id}, team={team_id}"
        )
        return team_id

    # 3. 创建新 Team
    new_team_id = str(uuid.uuid4())

    team_data = {
        "team_id": new_team_id,
        "team_alias": project_id,
        "models": [],  # 空列表表示允许所有模型
        "blocked": False,
        "metadata": {
            "yunguan_project_id": project_id,
            "yunguan_source": True,
            "yunguan_project_sk_id": yunguan_response.project_sk_id,
            "yunguan_sk_type": yunguan_response.sk_type,
        },
    }

    # 使用 prisma_client.jsonify_team_object 处理数据格式
    team_data = prisma_client.jsonify_team_object(db_data=team_data)

    verbose_proxy_logger.info(
        f"创建运管 Team: project={project_id}, team_id={new_team_id}"
    )

    try:
        team_row = await prisma_client.db.litellm_teamtable.create(
            data=team_data,
        )
        team_id = str(team_row.team_id)

        # 写入缓存
        await user_api_key_cache.async_set_cache(cache_key, team_id)
        verbose_proxy_logger.info(
            f"运管 Team 创建成功: project={project_id}, team_id={team_id}"
        )
        return team_id

    except Exception as e:
        # 处理并发创建导致的唯一键冲突
        error_str = str(e)
        if "Unique constraint" in error_str or "already exists" in error_str.lower():
            verbose_proxy_logger.warning(
                f"Team 创建冲突，重新查找: project={project_id}, error={e}"
            )
            # 重新查找已存在的 Team
            existing_team = await prisma_client.db.litellm_teamtable.find_first(
                where={"team_alias": project_id}
            )
            if existing_team:
                team_id = str(existing_team.team_id)
                await user_api_key_cache.async_set_cache(cache_key, team_id)
                return team_id

        verbose_proxy_logger.error(
            f"创建运管 Team 失败: project={project_id}, error={e}"
        )
        raise e


async def create_or_get_key_for_yunguan(
    sk: str,
    team_id: str,
    project_id: str,
    yunguan_response: YunguanAuthResponse,
    prisma_client: PrismaClient,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: Optional[ProxyLogging] = None,
    parent_otel_span: Optional[Any] = None,
) -> Optional[UserAPIKeyAuth]:
    """
    为运管用户创建或获取 LiteLLM API Key

    Args:
        sk: 原始 sk（作为 token 值存储）
        team_id: 关联的 team UUID
        project_id: 运管 projectId
        yunguan_response: 运管认证响应数据
        prisma_client: Prisma 客户端
        user_api_key_cache: 缓存对象
        proxy_logging_obj: 日志对象
        parent_otel_span: OpenTelemetry span

    Returns:
        UserAPIKeyAuth: 认证对象，失败返回 None
    """
    # 1. 计算 sk 的哈希值
    sk_hash = hash_token(sk)

    # 2. 先检查数据库中是否已存在此 sk 创建的 key
    try:
        existing_key = await get_key_object(
            hashed_token=sk_hash,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=parent_otel_span,
            proxy_logging_obj=proxy_logging_obj,
        )
        verbose_proxy_logger.info(
            f"查找到现有运管 Key: {sk[:8]}..., team={team_id}"
        )
        existing_key.parent_otel_span = parent_otel_span
        return existing_key
    except Exception:
        verbose_proxy_logger.info(
            f"运管 Key 不存在，创建新 Key: {sk[:8]}..., team={team_id}"
        )

    # 3. 创建新的 Key
    # key_name 设置为原始 sk (API Key)，便于在 UI 中识别和管理
    key_alias = sk

    key_data = {
        "token": sk_hash,
        "key_alias": key_alias,
        "user_id": LITELLM_PROXY_ADMIN_NAME,
        "team_id": team_id,
        "models": [],  # 空列表表示允许所有模型
        "max_budget": None,  # 不限制预算，由运管控制
        "blocked": False,
        "metadata": {
            "yunguan_sk": True,
            "yunguan_project_id": project_id,
            "yunguan_sk_type": yunguan_response.sk_type,
            "yunguan_project_sk_id": yunguan_response.project_sk_id,
        },
    }

    # 如果运管返回了过期时间，同步到 Key 上
    if yunguan_response.expire_time:
        key_data["expires"] = yunguan_response.expire_time

    # 保存原始 metadata 用于 UserAPIKeyAuth 对象
    original_metadata = key_data["metadata"]

    # 使用 prisma_client.jsonify_object 处理数据格式（用于数据库存储）
    key_data = prisma_client.jsonify_object(data=key_data)

    try:
        await prisma_client.db.litellm_verificationtoken.create(
            data=key_data,
        )
        verbose_proxy_logger.info(
            f"运管 Key 创建成功: sk={sk[:8]}..., team={team_id}, key_alias={key_alias}"
        )

        # 构建返回的 UserAPIKeyAuth（使用原始 metadata 字典）
        user_api_key_auth = UserAPIKeyAuth(
            api_key=sk_hash,
            token=sk_hash,
            key_alias=key_alias,
            user_id=LITELLM_PROXY_ADMIN_NAME,
            team_id=team_id,
            team_alias=project_id,
            models=[],  # 允许所有模型
            user_role=LitellmUserRoles.INTERNAL_USER,  # 运管用户为内部用户角色
            blocked=False,
            max_budget=None,
            metadata=original_metadata,
            expires=yunguan_response.expire_time,
            parent_otel_span=parent_otel_span,
        )

        # 写入缓存
        await _cache_key_object(
            hashed_token=sk_hash,
            user_api_key_obj=user_api_key_auth,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

        return user_api_key_auth

    except Exception as e:
        # 处理并发创建的唯一键冲突
        error_str = str(e)
        if "Unique constraint" in error_str and "token" in error_str:
            verbose_proxy_logger.warning(
                f"Key 创建冲突，重新查找: sk={sk[:8]}..., error={e}"
            )
            # 重新查找已存在的 Key
            try:
                existing_key = await get_key_object(
                    hashed_token=sk_hash,
                    prisma_client=prisma_client,
                    user_api_key_cache=user_api_key_cache,
                    parent_otel_span=parent_otel_span,
                    proxy_logging_obj=proxy_logging_obj,
                )
                existing_key.parent_otel_span = parent_otel_span
                return existing_key
            except Exception as lookup_error:
                verbose_proxy_logger.error(f"重新查找 Key 失败: {lookup_error}")
                return None

        verbose_proxy_logger.error(f"创建运管 Key 失败: sk={sk[:8]}..., error={e}")
        return None


async def yunguan_auth_fallback(
    api_key: str,
    model: Optional[Union[str, List[str]]],
    general_settings: Dict[str, Any],
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: Optional[ProxyLogging] = None,
    parent_otel_span: Optional[Any] = None,
) -> tuple[Optional[UserAPIKeyAuth], Optional[str]]:
    """
    运管认证兜底逻辑

    在 LiteLLM 标准认证失败后，尝试通过运管接口校验 SK。

    Args:
        api_key: 原始 API Key（如 sk-xxx）
        model: 请求中的模型名称
        general_settings: Proxy 配置的 general_settings
        prisma_client: Prisma 客户端
        user_api_key_cache: 缓存对象
        proxy_logging_obj: 日志对象
        parent_otel_span: OpenTelemetry span

    Returns:
        tuple: (UserAPIKeyAuth 或 None, error_code 或 None)
        - 认证通过: (UserAPIKeyAuth, None)
        - 认证失败: (None, error_code) 如 "PROJECT_SK_NOT_EXIST"
    """
    # 1. 检查配置是否启用运管认证
    import logging
    _yg_logger = logging.getLogger("yunguan_auth")
    _yg_logger.setLevel(logging.INFO)
    
    config = YunguanAuthConfig.from_general_settings(general_settings)
    if config is None:
        _yg_logger.warning(f"运管配置未启用或无效: enable_yunguan_auth={general_settings.get('enable_yunguan_auth')}, base_url={general_settings.get('yunguan_base_url')}")
        return None, "YUNGUAN_NOT_ENABLED"
    
    _yg_logger.info(f"运管配置已启用: base_url={config.base_url}")
    
    if prisma_client is None:
        _yg_logger.warning("prisma_client 为 None，无法继续运管认证")
        return None, "YUNGUAN_NO_DB"

    # 2. 提取模型名称
    model_name: Optional[str] = None
    if model:
        if isinstance(model, list):
            model_name = model[0] if model else None
        else:
            model_name = model
    
    _yg_logger.info(f"调用运管接口: sk={api_key[:8]}..., model={model_name}")

    # 3. 调用运管接口校验
    yunguan_response, yunguan_error_code = await verify_sk_with_yunguan(
        sk=api_key,
        model_name=model_name,
        config=config,
    )

    if yunguan_response is None:
        _yg_logger.warning(f"运管接口返回 None (认证失败), error_code={yunguan_error_code}")
        return None, yunguan_error_code

    _yg_logger.info(f"运管接口认证通过: project_id={yunguan_response.project_id}")

    # 4. 获取 projectId
    project_id = yunguan_response.project_id
    if not project_id:
        print(f"[YUNGUAN-FB] project_id 为空", file=sys.stderr, flush=True)
        verbose_proxy_logger.warning("运管认证返回的数据中没有 projectId")
        return None, "YUNGUAN_NO_PROJECT_ID"

    verbose_proxy_logger.info(
        f"运管认证通过: sk={api_key[:8]}..., project={project_id}, "
        f"quota_used={yunguan_response.quota_used}, token_total={yunguan_response.token_total}"
    )

    # 5. 获取或创建 Team
    try:
        team_id = await get_or_create_team_for_yunguan(
            project_id=project_id,
            yunguan_response=yunguan_response,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
    except Exception as e:
        verbose_proxy_logger.error(f"获取/创建运管 Team 失败: {e}")
        return None, "YUNGUAN_TEAM_ERROR"

    # 6. 创建或获取 Key
    user_api_key_auth = await create_or_get_key_for_yunguan(
        sk=api_key,
        team_id=team_id,
        project_id=project_id,
        yunguan_response=yunguan_response,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
        parent_otel_span=parent_otel_span,
    )

    return user_api_key_auth, None