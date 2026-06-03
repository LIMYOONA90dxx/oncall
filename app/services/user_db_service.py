"""用户数据库服务 - 管理用户数据"""

import json
import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict
from loguru import logger

from app.models.user import User, UserRole


class UserDatabase:
    """用户数据库（基于 JSON 文件，生产环境可替换为真实数据库）"""

    def __init__(self, db_path: str = "./data/users.json"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_db()

    def _ensure_db(self):
        """确保数据库文件存在"""
        if not self.db_path.exists():
            self._save_users({})
            logger.info(f"创建用户数据库: {self.db_path}")

    def _load_users(self) -> Dict[str, dict]:
        """加载所有用户"""
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载用户数据库失败: {e}")
            return {}

    def _save_users(self, users: Dict[str, dict]):
        """保存所有用户"""
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(users, f, ensure_ascii=False, indent=2, default=str)

    @staticmethod
    def hash_password(password: str) -> str:
        """密码哈希"""
        return hashlib.sha256(password.encode()).hexdigest()

    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """验证密码"""
        return UserDatabase.hash_password(password) == password_hash

    def create_user(
        self,
        username: str,
        password: str,
        tenant_id: Optional[str] = None,
        role: UserRole = UserRole.USER
    ) -> User:
        """创建用户"""
        users = self._load_users()

        if username in users:
            raise ValueError(f"用户名已存在: {username}")

        user_id = str(uuid.uuid4())
        password_hash = self.hash_password(password)
        
        if tenant_id is None:
            tenant_id = f"tenant_{user_id[:8]}"

        user = User(
            id=user_id,
            username=username,
            password_hash=password_hash,
            tenant_id=tenant_id,
            role=role,
            created_at=datetime.now(),
            is_active=True
        )

        users[username] = user.model_dump()
        self._save_users(users)

        logger.info(f"创建用户成功: username={username}, tenant_id={tenant_id}")
        return user

    def get_user_by_username(self, username: str) -> Optional[User]:
        """根据用户名获取用户"""
        users = self._load_users()
        user_data = users.get(username)
        if user_data:
            return User(**user_data)
        return None

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """根据用户ID获取用户"""
        users = self._load_users()
        for user_data in users.values():
            if user_data.get("id") == user_id:
                return User(**user_data)
        return None

    def update_last_login(self, username: str):
        """更新最后登录时间"""
        users = self._load_users()
        if username in users:
            users[username]["last_login"] = datetime.now().isoformat()
            self._save_users(users)

    def authenticate(self, username: str, password: str) -> Optional[User]:
        """验证用户登录"""
        user = self.get_user_by_username(username)
        if not user:
            return None

        if not self.verify_password(password, user.password_hash):
            return None

        if not user.is_active:
            return None

        self.update_last_login(username)
        return user

    def list_users_by_tenant(self, tenant_id: str) -> list[User]:
        """列出租户下的所有用户"""
        users = self._load_users()
        return [
            User(**data) for data in users.values()
            if data.get("tenant_id") == tenant_id
        ]


user_db = UserDatabase()
