"""文件上传接口模块（支持多租户隔离）"""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.services.vector_index_service import vector_index_service
from app.services.document_loader_service import document_loader_service
from app.services.tenant_service import tenant_service
from loguru import logger

router = APIRouter()

UPLOAD_DIR = Path("./uploads")
MAX_FILE_SIZE = 10 * 1024 * 1024


def _get_allowed_extensions() -> list[str]:
    """获取支持的文件扩展名列表"""
    return [ext.lstrip(".") for ext in document_loader_service.get_supported_extensions()]


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    tenant_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    is_public: bool = Form(False)
):
    """
    上传文件并自动创建向量索引（支持多租户隔离和公共知识库）

    Args:
        file: 上传的文件
        tenant_id: 租户ID（表单字段）
        user_id: 用户ID（表单字段）
        is_public: 是否为公共知识库（默认 False）

    Returns:
        JSONResponse: 上传结果
    """
    try:
        if is_public:
            tenant_service.set_public_context(user_id=user_id)
        else:
            tenant_service.set_context(tenant_id=tenant_id, user_id=user_id)

        if not file.filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")

        safe_filename = _sanitize_filename(file.filename)

        file_extension = _get_file_extension(safe_filename)
        allowed_extensions = _get_allowed_extensions()
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件格式，仅支持: {', '.join(allowed_extensions)}",
            )

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        file_path = UPLOAD_DIR / safe_filename

        if file_path.exists():
            logger.info(f"文件已存在，将覆盖: {file_path}")
            file_path.unlink()

        content = await file.read()

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"文件大小超过限制（最大 {MAX_FILE_SIZE} 字节）")

        file_path.write_bytes(content)

        current_tenant_id = tenant_service.get_tenant_id()
        knowledge_type = "公共" if is_public else "私有"
        logger.info(f"文件上传成功: {file_path}, 类型={knowledge_type}, tenant_id={current_tenant_id}")

        try:
            logger.info(f"开始为上传文件创建向量索引: {file_path}")
            vector_index_service.index_single_file(str(file_path), is_public=is_public)
            logger.info(f"向量索引创建成功: {file_path}")
        except Exception as e:
            logger.error(f"向量索引创建失败: {file_path}, 错误: {e}")

        return JSONResponse(
            status_code=200,
            content={
                "code": 200,
                "message": "success",
                "data": {
                    "filename": safe_filename,
                    "file_path": str(file_path),
                    "size": len(content),
                    "tenant_id": current_tenant_id,
                    "is_public": is_public,
                },
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"文件上传失败: {e}")
    finally:
        tenant_service.clear_context()


@router.post("/index_directory")
async def index_directory(
    directory_path: str = None,
    tenant_id: Optional[str] = None,
    user_id: Optional[str] = None
):
    """
    索引指定目录下的所有文件（支持多租户隔离）

    Args:
        directory_path: 目录路径（可选，默认使用 uploads 目录）
        tenant_id: 租户ID
        user_id: 用户ID

    Returns:
        JSONResponse: 索引结果
    """
    try:
        tenant_service.set_context(tenant_id=tenant_id, user_id=user_id)

        current_tenant_id = tenant_service.get_tenant_id()
        logger.info(f"开始索引目录: {directory_path or 'uploads'}, tenant_id={current_tenant_id}")

        result = vector_index_service.index_directory(directory_path)

        return JSONResponse(
            status_code=200,
            content={
                "code": 200,
                "message": "success" if result.success else "partial_success",
                "data": {
                    **result.to_dict(),
                    "tenant_id": current_tenant_id,
                },
            },
        )

    except Exception as e:
        logger.error(f"索引目录失败: {e}")
        raise HTTPException(status_code=500, detail=f"索引目录失败: {e}")
    finally:
        tenant_service.clear_context()


def _get_file_extension(filename: str) -> str:
    """
    获取文件扩展名

    Args:
        filename: 文件名

    Returns:
        str: 扩展名（小写，不含点）
    """
    parts = filename.rsplit(".", 1)
    if len(parts) == 2:
        return parts[1].lower()
    return ""


def _sanitize_filename(filename: str) -> str:
    """
    规范化文件名，去除空格和特殊字符

    Args:
        filename: 原始文件名

    Returns:
        str: 规范化后的文件名
    """
    # 去除空格
    sanitized = filename.replace(" ", "_")
    # 去除其他可能导致问题的字符
    for char in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
        sanitized = sanitized.replace(char, "_")
    return sanitized
