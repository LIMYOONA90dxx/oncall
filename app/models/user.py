"""用户模型"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum


class UserRole(str, Enum):
    """用户角色"""
    ADMIN = "admin"
    USER = "user"


class User(BaseModel):
    """用户模型"""
    id: str
    username: str
    password_hash: str
    tenant_id: str
    role: UserRole = UserRole.USER
    created_at: datetime = Field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_active: bool = True

    class Config:
        use_enum_values = True


class UserCreate(BaseModel):
    """用户注册请求"""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=100)
    tenant_id: Optional[str] = Field(None, min_length=1, max_length=100)


class UserLogin(BaseModel):
    """用户登录请求"""
    username: str
    password: str


class Token(BaseModel):
    """Token 响应"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_info: dict


class UserInfo(BaseModel):
    """用户信息响应"""
    id: str
    username: str
    tenant_id: str
    role: str
    is_active: bool
