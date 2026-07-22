"""
单元测试: 运管认证模块

覆盖 yunguan_auth.py 中的核心函数:
- hash_token
- YunguanAuthConfig.from_general_settings
- YunguanAuthResponse.from_api_response
- verify_sk_with_yunguan (mock httpx)
- get_or_create_team_for_yunguan
- create_or_get_key_for_yunguan
- yunguan_auth_fallback (编排集成)
"""

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.yunguan_auth import (
    YunguanAuthConfig,
    YunguanAuthResponse,
    create_or_get_key_for_yunguan,
    get_or_create_team_for_yunguan,
    hash_token,
    verify_sk_with_yunguan,
    yunguan_auth_fallback,
)
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def valid_config() -> YunguanAuthConfig:
    return YunguanAuthConfig(
        base_url="https://zjic.zhejianglab.com",
        auth_path="/apis/control-zjic/projectSk/check",
        timeout=5.0,
    )


@pytest.fixture
def valid_api_response_data() -> Dict[str, Any]:
    return {
        "code": "200",
        "message": "成功",
        "data": {
            "sk": "sk-test12345678",
            "tokenTotal": 2.00002e14,
            "expireTime": "2026-06-30T15:59:59.000+00:00",
            "projectSkId": 283,
            "skType": "PROJECT_CUSTOM",
            "projectId": "pr-1228421016223940608",
            "quotaUsed": 1563,
            "freeModel": False,
        },
        "success": True,
    }


@pytest.fixture
def valid_yunguan_response(valid_api_response_data) -> YunguanAuthResponse:
    resp, _ = YunguanAuthResponse.from_api_response(valid_api_response_data)
    assert resp is not None
    return resp


@pytest.fixture
def mock_prisma_client():
    client = MagicMock()
    client.db = MagicMock()
    client.db.litellm_teamtable = MagicMock()
    client.db.litellm_verificationtoken = MagicMock()
    client.jsonify_team_object = MagicMock(side_effect=lambda db_data: db_data)
    client.jsonify_object = MagicMock(side_effect=lambda data: data)
    return client


@pytest.fixture
def mock_user_api_key_cache():
    cache = MagicMock(spec=UserApiKeyCache)
    cache.async_get_cache = AsyncMock(return_value=None)
    cache.async_set_cache = AsyncMock()
    return cache


# ── hash_token ────────────────────────────────────────────────


class TestHashToken:
    def test_should_produce_sha256_hex(self):
        result = hash_token("sk-test123")
        expected = hashlib.sha256("sk-test123".encode("utf-8")).hexdigest()
        assert result == expected

    def test_should_be_deterministic(self):
        assert hash_token("abc") == hash_token("abc")

    def test_should_differ_for_different_inputs(self):
        assert hash_token("abc") != hash_token("def")


# ── YunguanAuthConfig.from_general_settings ───────────────────


class TestYunguanAuthConfig:
    def test_should_return_none_when_not_enabled(self):
        assert YunguanAuthConfig.from_general_settings({"enable_yunguan_auth": False}) is None

    def test_should_return_none_when_enabled_but_no_base_url(self):
        assert YunguanAuthConfig.from_general_settings({"enable_yunguan_auth": True}) is None

    def test_should_use_defaults_when_not_specified(self):
        cfg = YunguanAuthConfig.from_general_settings({
            "enable_yunguan_auth": True,
            "yunguan_base_url": "https://example.com",
        })
        assert cfg is not None
        assert cfg.base_url == "https://example.com"
        assert cfg.auth_path == "/apis/control-zjic/projectSk/check"
        assert cfg.timeout == 10.0

    def test_should_use_custom_settings(self):
        cfg = YunguanAuthConfig.from_general_settings({
            "enable_yunguan_auth": True,
            "yunguan_base_url": "https://custom.example.com",
            "yunguan_auth_path": "/custom/check",
            "yunguan_timeout": 3.0,
        })
        assert cfg is not None
        assert cfg.base_url == "https://custom.example.com"
        assert cfg.auth_path == "/custom/check"
        assert cfg.timeout == 3.0

    def test_should_return_none_when_key_missing(self):
        assert YunguanAuthConfig.from_general_settings({}) is None


# ── YunguanAuthResponse.from_api_response ─────────────────────


class TestYunguanAuthResponse:
    def test_should_parse_success_response(self, valid_api_response_data):
        resp, error = YunguanAuthResponse.from_api_response(valid_api_response_data)
        assert error is None
        assert resp is not None
        assert resp.sk == "sk-test12345678"
        assert resp.token_total == 2.00002e14
        assert resp.project_id == "pr-1228421016223940608"
        assert resp.project_sk_id == 283
        assert resp.sk_type == "PROJECT_CUSTOM"
        assert resp.quota_used == 1563
        assert resp.free_model is False
        # 过期时间解析
        assert resp.expire_time is not None
        assert resp.expire_time.tzinfo == timezone.utc
        assert resp.expire_time.year == 2026

    def test_should_return_error_for_failed_response(self):
        data = {"code": "PROJECT_SK_NOT_EXIST", "message": "SK不存在", "success": False}
        resp, error = YunguanAuthResponse.from_api_response(data)
        assert resp is None
        assert error == "PROJECT_SK_NOT_EXIST"

    def test_should_return_error_when_code_not_200(self):
        data = {"code": "500", "message": "服务器错误", "success": False}
        resp, error = YunguanAuthResponse.from_api_response(data)
        assert resp is None
        assert error == "500"

    def test_should_return_error_when_success_false(self):
        data = {"code": "200", "message": "成功", "success": False}
        resp, error = YunguanAuthResponse.from_api_response(data)
        assert resp is None

    def test_should_handle_missing_data_field(self):
        """当 data 字段缺失时，from_api_response 返回 None（非 tuple）"""
        result = YunguanAuthResponse.from_api_response({"code": "200", "success": True})
        assert result is None

    def test_should_handle_malformed_expire_time(self):
        data = {
            "code": "200", "success": True,
            "data": {"sk": "sk-test", "tokenTotal": 100, "expireTime": "not-a-date",
                     "projectSkId": 1, "skType": "T", "projectId": "pr", "quotaUsed": 0, "freeModel": False},
        }
        resp, error = YunguanAuthResponse.from_api_response(data)
        assert error is None
        assert resp is not None
        assert resp.expire_time is None  # 解析失败 -> None


# ── verify_sk_with_yunguan ────────────────────────────────────


class TestVerifySkWithYunguan:
    async def _mock_httpx(self, return_value=None, side_effect=None):
        mock_resp = MagicMock(spec=httpx.Response) if return_value else None
        if mock_resp is not None:
            mock_resp.status_code = 200
            mock_resp.json.return_value = return_value

        class _Ctx:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass

        ctx = _Ctx()
        ctx.get = AsyncMock(
            return_value=mock_resp, side_effect=side_effect
        )
        return ctx

    @pytest.mark.asyncio
    async def test_should_return_response_on_success(self, valid_config, valid_api_response_data):
        ctx = await self._mock_httpx(return_value=valid_api_response_data)
        with patch("httpx.AsyncClient", return_value=ctx):
            resp, error = await verify_sk_with_yunguan("sk-test12345678", "test-model", valid_config)
        assert error is None
        assert resp is not None
        assert resp.project_id == "pr-1228421016223940608"

    @pytest.mark.asyncio
    async def test_should_return_error_on_timeout(self, valid_config):
        ctx = await self._mock_httpx(side_effect=httpx.TimeoutException("timeout"))
        with patch("httpx.AsyncClient", return_value=ctx):
            resp, error = await verify_sk_with_yunguan("sk-test", None, valid_config)
        assert resp is None
        assert error == "YUNGUAN_TIMEOUT"

    @pytest.mark.asyncio
    async def test_should_return_error_on_exception(self, valid_config):
        ctx = await self._mock_httpx(side_effect=RuntimeError("unexpected"))
        with patch("httpx.AsyncClient", return_value=ctx):
            resp, error = await verify_sk_with_yunguan("sk-test", None, valid_config)
        assert resp is None
        assert error == "YUNGUAN_EXCEPTION"

    @pytest.mark.asyncio
    async def test_should_return_business_error_code(self, valid_config):
        error_data = {"code": "PROJECT_SK_NOT_EXIST", "message": "SK不存在", "success": False}
        ctx = await self._mock_httpx(return_value=error_data)
        with patch("httpx.AsyncClient", return_value=ctx):
            resp, error = await verify_sk_with_yunguan("sk-test", None, valid_config)
        assert resp is None
        assert error == "PROJECT_SK_NOT_EXIST"

    @pytest.mark.asyncio
    async def test_should_pass_model_code_param(self, valid_config):
        """验证 model_name 正确传递为 modelCode 查询参数"""
        data = {"code": "200", "success": True, "data": {
            "sk": "s", "tokenTotal": 1, "expireTime": None, "projectSkId": 1,
            "skType": "T", "projectId": "p", "quotaUsed": 0, "freeModel": False}}
        ctx = await self._mock_httpx(return_value=data)
        with patch("httpx.AsyncClient", return_value=ctx):
            await verify_sk_with_yunguan("sk-test", "Qwen3-Coder-Next-FP8", valid_config)
        assert ctx.get.call_args is not None
        assert ctx.get.call_args.kwargs["params"]["modelCode"] == "Qwen3-Coder-Next-FP8"

    @pytest.mark.asyncio
    async def test_should_handle_json_parse_error(self, valid_config):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.side_effect = ValueError("bad json")

        class _Ctx:
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass

        ctx = _Ctx()
        ctx.get = AsyncMock(return_value=mock_resp)
        with patch("httpx.AsyncClient", return_value=ctx):
            resp, error = await verify_sk_with_yunguan("sk-test", None, valid_config)
        assert resp is None
        assert error == "YUNGUAN_PARSE_ERROR"


# ── get_or_create_team_for_yunguan ────────────────────────────


class TestGetOrCreateTeamForYunguan:
    @pytest.mark.asyncio
    async def test_should_return_cached_team(self, mock_prisma_client, mock_user_api_key_cache, valid_yunguan_response):
        mock_user_api_key_cache.async_get_cache.return_value = "cached-uuid"
        team_id = await get_or_create_team_for_yunguan("pr-test", valid_yunguan_response,
                                                        mock_prisma_client, mock_user_api_key_cache)
        assert team_id == "cached-uuid"
        mock_prisma_client.db.litellm_teamtable.find_first.assert_not_called()

    @pytest.mark.asyncio
    async def test_should_return_db_team(self, mock_prisma_client, mock_user_api_key_cache, valid_yunguan_response):
        mock_user_api_key_cache.async_get_cache.return_value = None
        mock_team = MagicMock(team_id="db-uuid")
        mock_prisma_client.db.litellm_teamtable.find_first = AsyncMock(return_value=mock_team)
        team_id = await get_or_create_team_for_yunguan("pr-test", valid_yunguan_response,
                                                        mock_prisma_client, mock_user_api_key_cache)
        assert team_id == "db-uuid"
        mock_user_api_key_cache.async_set_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_should_create_new_team(self, mock_prisma_client, mock_user_api_key_cache, valid_yunguan_response):
        mock_user_api_key_cache.async_get_cache.return_value = None
        mock_prisma_client.db.litellm_teamtable.find_first = AsyncMock(return_value=None)
        mock_team = MagicMock(team_id="new-uuid")
        mock_prisma_client.db.litellm_teamtable.create = AsyncMock(return_value=mock_team)

        team_id = await get_or_create_team_for_yunguan("pr-new", valid_yunguan_response,
                                                        mock_prisma_client, mock_user_api_key_cache)
        assert team_id == "new-uuid"
        create_call = mock_prisma_client.db.litellm_teamtable.create.call_args
        data = create_call.kwargs.get("data", create_call.args[0] if create_call.args else {})
        assert data.get("team_alias") == "pr-new"
        assert data.get("models") == []

    @pytest.mark.asyncio
    async def test_should_handle_concurrent_creation_conflict(self, mock_prisma_client, mock_user_api_key_cache, valid_yunguan_response):
        mock_user_api_key_cache.async_get_cache.return_value = None
        mock_prisma_client.db.litellm_teamtable.find_first = AsyncMock(return_value=None)
        mock_prisma_client.db.litellm_teamtable.create = AsyncMock(
            side_effect=Exception("Unique constraint failed"))
        mock_team = MagicMock(team_id="retry-uuid")
        mock_prisma_client.db.litellm_teamtable.find_first = AsyncMock(return_value=mock_team)

        team_id = await get_or_create_team_for_yunguan("pr-test", valid_yunguan_response,
                                                        mock_prisma_client, mock_user_api_key_cache)
        assert team_id == "retry-uuid"

    @pytest.mark.asyncio
    async def test_should_raise_on_unknown_create_error(self, mock_prisma_client, mock_user_api_key_cache, valid_yunguan_response):
        mock_user_api_key_cache.async_get_cache.return_value = None
        mock_prisma_client.db.litellm_teamtable.find_first = AsyncMock(return_value=None)
        mock_prisma_client.db.litellm_teamtable.create = AsyncMock(side_effect=RuntimeError("db down"))
        with pytest.raises(RuntimeError, match="db down"):
            await get_or_create_team_for_yunguan("pr-test", valid_yunguan_response,
                                                  mock_prisma_client, mock_user_api_key_cache)


# ── create_or_get_key_for_yunguan ─────────────────────────────


class TestCreateOrGetKeyForYunguan:
    @pytest.mark.asyncio
    async def test_should_return_existing_key(self, mock_prisma_client, mock_user_api_key_cache, valid_yunguan_response):
        existing = UserAPIKeyAuth(api_key=hash_token("sk-x"), token=hash_token("sk-x"),
                                  key_alias="sk-x", user_id="admin", team_id="t1",
                                  user_role=LitellmUserRoles.INTERNAL_USER)
        with patch("litellm.proxy.auth.yunguan_auth.get_key_object", AsyncMock(return_value=existing)):
            result = await create_or_get_key_for_yunguan("sk-x", "t1", "pr", valid_yunguan_response,
                                                          mock_prisma_client, mock_user_api_key_cache)
            assert result is not None
            assert result.api_key == hash_token("sk-x")

    @pytest.mark.asyncio
    async def test_should_create_new_key(self, mock_prisma_client, mock_user_api_key_cache, valid_yunguan_response):
        with patch("litellm.proxy.auth.yunguan_auth.get_key_object", AsyncMock(side_effect=Exception("not found"))):
            mock_prisma_client.db.litellm_verificationtoken.create = AsyncMock()
            result = await create_or_get_key_for_yunguan("sk-new", "team-uuid", "pr-test",
                                                          valid_yunguan_response,
                                                          mock_prisma_client, mock_user_api_key_cache)
        assert result is not None
        assert result.team_id == "team-uuid"
        assert result.user_role == LitellmUserRoles.INTERNAL_USER
        assert result.key_alias == "sk-new"
        assert result.models == []
        assert result.metadata.get("yunguan_sk") is True

    @pytest.mark.asyncio
    async def test_should_sync_expires(self, mock_prisma_client, mock_user_api_key_cache, valid_yunguan_response):
        with patch("litellm.proxy.auth.yunguan_auth.get_key_object", AsyncMock(side_effect=Exception("not found"))):
            mock_prisma_client.db.litellm_verificationtoken.create = AsyncMock()
            result = await create_or_get_key_for_yunguan("sk-e", "t", "p", valid_yunguan_response,
                                                          mock_prisma_client, mock_user_api_key_cache)
        assert result is not None
        assert result.expires is not None
        assert result.expires.tzinfo == timezone.utc

    @pytest.mark.asyncio
    async def test_should_handle_concurrent_key_conflict(self, mock_prisma_client, mock_user_api_key_cache, valid_yunguan_response):
        existing = UserAPIKeyAuth(api_key=hash_token("sk-c"), token=hash_token("sk-c"),
                                  key_alias="sk-c", user_id="admin", team_id="t",
                                  user_role=LitellmUserRoles.INTERNAL_USER)
        with patch("litellm.proxy.auth.yunguan_auth.get_key_object",
                   AsyncMock(side_effect=[Exception("not found"), existing])):
            mock_prisma_client.db.litellm_verificationtoken.create = AsyncMock(
                side_effect=Exception("Unique constraint on token"))
            result = await create_or_get_key_for_yunguan("sk-c", "t", "p", valid_yunguan_response,
                                                          mock_prisma_client, mock_user_api_key_cache)
        assert result is not None


# ── yunguan_auth_fallback ─────────────────────────────────────


class TestYunguanAuthFallback:
    @pytest.mark.asyncio
    async def test_should_return_error_when_not_enabled(self, mock_prisma_client, mock_user_api_key_cache):
        result, error = await yunguan_auth_fallback("sk-test", "m", {"enable_yunguan_auth": False},
                                                     mock_prisma_client, mock_user_api_key_cache)
        assert result is None
        assert error == "YUNGUAN_NOT_ENABLED"

    @pytest.mark.asyncio
    async def test_should_return_error_when_no_db(self, mock_user_api_key_cache):
        gs = {"enable_yunguan_auth": True, "yunguan_base_url": "https://x.com"}
        result, error = await yunguan_auth_fallback("sk-test", "m", gs, None, mock_user_api_key_cache)
        assert result is None
        assert error == "YUNGUAN_NO_DB"

    @pytest.mark.asyncio
    async def test_should_return_auth_on_success(self, valid_yunguan_response,
                                                  mock_prisma_client, mock_user_api_key_cache):
        gs = {"enable_yunguan_auth": True, "yunguan_base_url": "https://zjic.zhejianglab.com"}
        mock_auth = UserAPIKeyAuth(api_key=hash_token("sk-t"), token=hash_token("sk-t"),
                                   key_alias="sk-t", user_id="admin", team_id="t-123",
                                   team_alias="pr-x", user_role=LitellmUserRoles.INTERNAL_USER)

        with patch("litellm.proxy.auth.yunguan_auth.verify_sk_with_yunguan",
                   AsyncMock(return_value=(valid_yunguan_response, None))), \
             patch("litellm.proxy.auth.yunguan_auth.get_or_create_team_for_yunguan",
                   AsyncMock(return_value="t-123")), \
             patch("litellm.proxy.auth.yunguan_auth.create_or_get_key_for_yunguan",
                   AsyncMock(return_value=mock_auth)):
            result, error = await yunguan_auth_fallback("sk-test12345678", "Qwen3-Coder-Next-FP8",
                                                         gs, mock_prisma_client, mock_user_api_key_cache)
        assert error is None
        assert result is not None
        assert result.team_id == "t-123"

    @pytest.mark.asyncio
    async def test_should_return_error_when_verify_fails(self, mock_prisma_client, mock_user_api_key_cache):
        gs = {"enable_yunguan_auth": True, "yunguan_base_url": "https://zjic.zhejianglab.com"}
        with patch("litellm.proxy.auth.yunguan_auth.verify_sk_with_yunguan",
                   AsyncMock(return_value=(None, "PROJECT_SK_NOT_EXIST"))):
            result, error = await yunguan_auth_fallback("sk-bad", "m", gs, mock_prisma_client, mock_user_api_key_cache)
        assert result is None
        assert error == "PROJECT_SK_NOT_EXIST"

    @pytest.mark.asyncio
    async def test_should_return_error_when_project_id_empty(self, mock_prisma_client, mock_user_api_key_cache):
        gs = {"enable_yunguan_auth": True, "yunguan_base_url": "https://zjic.zhejianglab.com"}
        empty_resp = YunguanAuthResponse(sk="s", token_total=1, expire_time=None, project_sk_id=1,
                                          sk_type="T", project_id="", quota_used=0, free_model=False)
        with patch("litellm.proxy.auth.yunguan_auth.verify_sk_with_yunguan",
                   AsyncMock(return_value=(empty_resp, None))):
            result, error = await yunguan_auth_fallback("sk-t", "m", gs, mock_prisma_client, mock_user_api_key_cache)
        assert result is None
        assert error == "YUNGUAN_NO_PROJECT_ID"

    @pytest.mark.asyncio
    async def test_should_return_error_when_team_creation_fails(self, valid_yunguan_response,
                                                                 mock_prisma_client, mock_user_api_key_cache):
        gs = {"enable_yunguan_auth": True, "yunguan_base_url": "https://zjic.zhejianglab.com"}
        with patch("litellm.proxy.auth.yunguan_auth.verify_sk_with_yunguan",
                   AsyncMock(return_value=(valid_yunguan_response, None))), \
             patch("litellm.proxy.auth.yunguan_auth.get_or_create_team_for_yunguan",
                   AsyncMock(side_effect=RuntimeError("db error"))):
            result, error = await yunguan_auth_fallback("sk-t", "m", gs, mock_prisma_client, mock_user_api_key_cache)
        assert result is None
        assert error == "YUNGUAN_TEAM_ERROR"

    @pytest.mark.asyncio
    async def test_should_extract_first_model_from_list(self, valid_yunguan_response,
                                                         mock_prisma_client, mock_user_api_key_cache):
        """model 为 list 时取第一个元素传给运管接口"""
        gs = {"enable_yunguan_auth": True, "yunguan_base_url": "https://zjic.zhejianglab.com"}
        mock_auth = UserAPIKeyAuth(api_key=hash_token("sk-t"), token=hash_token("sk-t"),
                                   user_role=LitellmUserRoles.INTERNAL_USER)
        with patch("litellm.proxy.auth.yunguan_auth.verify_sk_with_yunguan",
                   AsyncMock(return_value=(valid_yunguan_response, None))) as mock_verify, \
             patch("litellm.proxy.auth.yunguan_auth.get_or_create_team_for_yunguan",
                   AsyncMock(return_value="t")), \
             patch("litellm.proxy.auth.yunguan_auth.create_or_get_key_for_yunguan",
                   AsyncMock(return_value=mock_auth)):
            await yunguan_auth_fallback("sk-t", ["model-a", "model-b"], gs,
                                        mock_prisma_client, mock_user_api_key_cache)
        assert mock_verify.call_args.kwargs["model_name"] == "model-a"
