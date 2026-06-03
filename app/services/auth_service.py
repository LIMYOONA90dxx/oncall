"""JWT 认证服务"""

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from loguru import logger

from app.config import config
from app.models.user import User, UserInfo
from app.services.user_db_service import user_db
from app.services.tenant_service import tenant_service


SECRET_KEY = config.jwt_secret_key
ALGORITHM = config.jwt_algorithm
ACCESS_TOKEN_EXPIRE_HOURS = config.jwt_expire_hours

security = HTTPBearer(auto_error=False)


class AuthService:
    """JWT 认证服务"""

    @staticmethod
    def create_access_token(user: User, expires_delta: Optional[timedelta] = None) -> str:
        """创建 JWT Token"""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)

        payload = {
            "sub": user.id,
            "username": user.username,
            "tenant_id": user.tenant_id,
            "role": user.role,
            "exp": expire,
            "iat": datetime.utcnow()
        }

        token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
        logger.debug(f"创建 Token: username={user.username}, tenant_id={user.tenant_id}")
        return token

    @staticmethod
    def decode_token(token: str) -> Optional[dict]:
        """解码 JWT Token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except JWTError as e:
            logger.warning(f"Token 解码失败: {e}")
            return None

    @staticmethod
    def get_current_user(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
    ) -> Optional[User]:
        """获取当前用户（可选认证）"""
        if not credentials:
            return None

        token = credentials.credentials
        payload = AuthService.decode_token(token)

        if not payload:
            return None

        user_id = payload.get("sub")
        if not user_id:
            return None

        user = user_db.get_user_by_id(user_id)
        return user

    @staticmethod
    def require_auth(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
    ) -> User:
        """要求用户认证（必须登录）"""
        if not credentials:
            raise HTTPException(
                status_code=401,
                detail="未登录，请先登录",
                headers={"WWW-Authenticate": "Bearer"}
            )

        token = credentials.credentials
        payload = AuthService.decode_token(token)

        if not payload:
            raise HTTPException(
                status_code=401,
                detail="Token 无效或已过期",
                headers={"WWW-Authenticate": "Bearer"}
            )

        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Token 格式错误",
                headers={"WWW-Authenticate": "Bearer"}
            )

        user = user_db.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=401,
                detail="用户不存在",
                headers={"WWW-Authenticate": "Bearer"}
            )

        if not user.is_active:
            raise HTTPException(
                status_code=403,
                detail="用户已被禁用"
            )

        tenant_service.set_context(
            tenant_id=user.tenant_id,
            user_id=user.id
        )

        return user

    @staticmethod
    def require_admin(current_user: User = Depends(require_auth)) -> User:
        """要求管理员权限"""
        if current_user.role != "admin":
            raise HTTPException(
                status_code=403,
                detail="需要管理员权限"
            )
        return current_user


auth_service = AuthService()
require_auth = AuthService.require_auth
require_admin = AuthService.require_admin
get_current_user = AuthService.get_current_user
