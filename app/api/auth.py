"""认证 API 路由"""

from datetime import timedelta
from fastapi import APIRouter, HTTPException, Depends

from loguru import logger

from app.models.user import UserCreate, UserLogin, Token, UserInfo, User, UserRole
from app.services.user_db_service import user_db
from app.services.auth_service import auth_service, require_auth
from app.services.tenant_service import tenant_service
from app.config import config

router = APIRouter()


@router.post("/auth/register", response_model=Token)
async def register(user_data: UserCreate):
    """用户注册

    创建新用户账号，每个账号属于一个租户

    Returns:
        Token: 包含 access_token 和用户信息
    """
    try:
        user = user_db.create_user(
            username=user_data.username,
            password=user_data.password,
            tenant_id=user_data.tenant_id,
            role=UserRole.USER
        )

        access_token = auth_service.create_access_token(user)

        logger.info(f"用户注册成功: username={user.username}, tenant_id={user.tenant_id}")

        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=config.jwt_expire_hours * 3600,
            user_info={
                "id": user.id,
                "username": user.username,
                "tenant_id": user.tenant_id,
                "role": user.role
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"注册失败: {e}")
        raise HTTPException(status_code=500, detail="注册失败")


@router.post("/auth/login", response_model=Token)
async def login(login_data: UserLogin):
    """用户登录

    验证用户名和密码，返回 JWT Token

    Returns:
        Token: 包含 access_token 和用户信息
    """
    user = user_db.authenticate(login_data.username, login_data.password)

    if not user:
        raise HTTPException(
            status_code=401,
            detail="用户名或密码错误"
        )

    access_token = auth_service.create_access_token(user)

    logger.info(f"用户登录成功: username={user.username}, tenant_id={user.tenant_id}")

    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=config.jwt_expire_hours * 3600,
        user_info={
            "id": user.id,
            "username": user.username,
            "tenant_id": user.tenant_id,
            "role": user.role
        }
    )


@router.get("/auth/me", response_model=UserInfo)
async def get_current_user_info(current_user: User = Depends(require_auth)):
    """获取当前登录用户信息

    需要在 Header 中携带 Bearer Token

    Returns:
        UserInfo: 当前用户信息
    """
    return UserInfo(
        id=current_user.id,
        username=current_user.username,
        tenant_id=current_user.tenant_id,
        role=current_user.role,
        is_active=current_user.is_active
    )


@router.post("/auth/logout")
async def logout(current_user: User = Depends(require_auth)):
    """用户登出

    客户端需要删除本地存储的 Token

    Returns:
        dict: 成功消息
    """
    logger.info(f"用户登出: username={current_user.username}")
    tenant_service.clear_context()

    return {
        "code": 200,
        "message": "登出成功",
        "data": None
    }


@router.get("/auth/tenant/users", response_model=list[UserInfo])
async def list_tenant_users(current_user: User = Depends(require_auth)):
    """列出当前租户下的所有用户（需要管理员权限）

    Returns:
        list[UserInfo]: 用户列表
    """
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="需要管理员权限")

    users = user_db.list_users_by_tenant(current_user.tenant_id)

    return [
        UserInfo(
            id=u.id,
            username=u.username,
            tenant_id=u.tenant_id,
            role=u.role,
            is_active=u.is_active
        )
        for u in users
    ]
