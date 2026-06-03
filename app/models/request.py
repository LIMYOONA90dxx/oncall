"""请求数据模型

定义 API 请求的 Pydantic 模型
"""

from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """对话请求"""

    id: str = Field(..., description="会话 ID", alias="Id")
    question: str = Field(..., description="用户问题", alias="Question")
    tenant_id: Optional[str] = Field(None, description="租户 ID", alias="TenantId")
    user_id: Optional[str] = Field(None, description="用户 ID", alias="UserId")
    include_public: bool = Field(True, description="是否包含公共知识库", alias="IncludePublic")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "Id": "session-123",
                "Question": "什么是向量数据库？",
                "TenantId": "tenant_001",
                "UserId": "user_123",
                "IncludePublic": True
            }
        }


class ClearRequest(BaseModel):
    """清空会话请求"""

    session_id: str = Field(..., description="会话 ID", alias="sessionId")
    tenant_id: Optional[str] = Field(None, description="租户 ID", alias="TenantId")
    user_id: Optional[str] = Field(None, description="用户 ID", alias="UserId")

    class Config:
        populate_by_name = True
