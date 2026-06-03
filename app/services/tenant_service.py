"""多租户隔离服务 - 管理租户上下文和数据隔离"""

from typing import Optional
from contextvars import ContextVar

from loguru import logger


_tenant_context: ContextVar[Optional[dict]] = ContextVar("tenant_context", default=None)


class TenantContext:
    """租户上下文"""

    def __init__(self, tenant_id: str, user_id: Optional[str] = None):
        self.tenant_id = tenant_id
        self.user_id = user_id

    def to_dict(self) -> dict:
        return {
            "_tenant_id": self.tenant_id,
            "_user_id": self.user_id,
        }

    def __repr__(self):
        return f"TenantContext(tenant_id={self.tenant_id}, user_id={self.user_id})"


class TenantService:
    """多租户隔离服务"""

    DEFAULT_TENANT_ID = "default"
    DEFAULT_USER_ID = "anonymous"
    PUBLIC_TENANT_ID = "public"

    def __init__(self):
        self._enabled = True
        self._include_public = True
        logger.info("多租户隔离服务初始化完成")

    def set_context(
        self,
        tenant_id: str,
        user_id: Optional[str] = None,
        include_public: bool = True
    ) -> TenantContext:
        """
        设置当前请求的租户上下文

        Args:
            tenant_id: 租户ID
            user_id: 用户ID（可选）
            include_public: 是否包含公共知识库（默认 True）

        Returns:
            TenantContext: 租户上下文对象
        """
        if not tenant_id:
            tenant_id = self.DEFAULT_TENANT_ID
            logger.warning(f"tenant_id 为空，使用默认值: {tenant_id}")

        if not user_id:
            user_id = self.DEFAULT_USER_ID

        self._include_public = include_public

        context = TenantContext(tenant_id=tenant_id, user_id=user_id)
        _tenant_context.set(context)
        logger.debug(f"设置租户上下文: {context}, include_public={include_public}")
        return context

    def get_context(self) -> TenantContext:
        """
        获取当前请求的租户上下文

        Returns:
            TenantContext: 租户上下文对象（如果未设置则返回默认值）
        """
        context = _tenant_context.get()
        if context is None:
            logger.warning("租户上下文未设置，使用默认值")
            return TenantContext(self.DEFAULT_TENANT_ID, self.DEFAULT_USER_ID)
        return context

    def clear_context(self):
        """清除当前请求的租户上下文"""
        _tenant_context.set(None)
        logger.debug("清除租户上下文")

    def get_tenant_id(self) -> str:
        """获取当前租户ID"""
        return self.get_context().tenant_id

    def get_user_id(self) -> str:
        """获取当前用户ID"""
        return self.get_context().user_id

    def get_metadata_filter(self) -> dict:
        """
        获取租户过滤条件（用于 Milvus 查询）

        Returns:
            dict: 包含 _tenant_id 和 _user_id 的过滤条件
        """
        context = self.get_context()
        return {
            "_tenant_id": context.tenant_id,
            "_user_id": context.user_id,
        }

    def build_milvus_filter(self, additional_filter: Optional[str] = None) -> str:
        """
        构建 Milvus 过滤表达式（支持公共知识库）

        Args:
            additional_filter: 额外的过滤条件

        Returns:
            str: Milvus 过滤表达式
        """
        context = self.get_context()

        tenant_filter = f'metadata["_tenant_id"] == "{context.tenant_id}"'
        public_filter = f'metadata["_tenant_id"] == "{self.PUBLIC_TENANT_ID}"'
        no_tenant_filter = 'metadata["_tenant_id"] == ""'

        if self._include_public:
            base_filter = f"({tenant_filter} or {public_filter} or {no_tenant_filter})"
        else:
            base_filter = f"({tenant_filter} or {no_tenant_filter})"

        if additional_filter:
            return f"{base_filter} and ({additional_filter})"

        return base_filter

    def enrich_metadata(self, metadata: dict, is_public: bool = False) -> dict:
        """
        为元数据添加租户信息

        Args:
            metadata: 原始元数据
            is_public: 是否为公共知识库

        Returns:
            dict: 添加了租户信息的元数据
        """
        context = self.get_context()
        enriched = dict(metadata)

        if is_public:
            enriched["_tenant_id"] = self.PUBLIC_TENANT_ID
        else:
            enriched["_tenant_id"] = context.tenant_id

        enriched["_user_id"] = context.user_id
        return enriched

    def is_public_tenant(self) -> bool:
        """检查当前租户是否为公共租户"""
        return self.get_context().tenant_id == self.PUBLIC_TENANT_ID

    def is_include_public(self) -> bool:
        """检查是否包含公共知识库"""
        return self._include_public

    def set_public_context(self, user_id: Optional[str] = None):
        """
        设置公共知识库上下文（用于上传公共知识）

        Args:
            user_id: 用户ID
        """
        self.set_context(tenant_id=self.PUBLIC_TENANT_ID, user_id=user_id, include_public=False)
        logger.info(f"设置公共知识库上下文, user_id={user_id}")

    def is_enabled(self) -> bool:
        """检查多租户隔离是否启用"""
        return self._enabled

    def enable(self):
        """启用多租户隔离"""
        self._enabled = True
        logger.info("多租户隔离已启用")

    def disable(self):
        """禁用多租户隔离（仅用于测试或特殊场景）"""
        self._enabled = False
        logger.warning("多租户隔离已禁用，数据可能串租户！")


tenant_service = TenantService()


def get_tenant_id() -> str:
    """快捷方法：获取当前租户ID"""
    return tenant_service.get_tenant_id()


def get_user_id() -> str:
    """快捷方法：获取当前用户ID"""
    return tenant_service.get_user_id()


def with_tenant(tenant_id: str, user_id: Optional[str] = None):
    """
    装饰器：设置租户上下文

    Usage:
        @with_tenant("tenant_001", "user_123")
        def my_function():
            ...
    """
    def decorator(func):
        from functools import wraps
        @wraps(func)
        def wrapper(*args, **kwargs):
            tenant_service.set_context(tenant_id, user_id)
            try:
                return func(*args, **kwargs)
            finally:
                tenant_service.clear_context()
        return wrapper
    return decorator
