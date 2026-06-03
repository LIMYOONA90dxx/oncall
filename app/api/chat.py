"""对话接口

提供基于 RAG Agent 的普通对话和流式对话接口
"""

import json
from fastapi import APIRouter, HTTPException, Depends
from sse_starlette.sse import EventSourceResponse
from app.models.request import ChatRequest, ClearRequest
from app.models.response import SessionInfoResponse, ApiResponse
from app.models.user import User
from app.services.rag_agent_service import rag_agent_service
from app.services.tenant_service import tenant_service
from app.services.auth_service import get_current_user
from loguru import logger

router = APIRouter()


@router.post("/chat")
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user)
):
    """快速对话接口

    支持两种认证方式：
    1. Bearer Token（推荐）：从 Token 中自动获取 tenant_id
    2. 请求参数：从请求体中获取 tenant_id（兼容模式）

    Returns:
        统一格式的对话响应
    """
    try:
        if current_user:
            tenant_id = current_user.tenant_id
            user_id = current_user.id
            include_public = request.include_public
        else:
            tenant_id = request.tenant_id
            user_id = request.user_id
            include_public = request.include_public

        tenant_service.set_context(
            tenant_id=tenant_id,
            user_id=user_id,
            include_public=include_public
        )

        logger.info(f"[会话 {request.id}] 收到快速对话请求: {request.question}, tenant_id={tenant_id}")

        answer = await rag_agent_service.query(
            request.question,
            session_id=request.id
        )

        logger.info(f"[会话 {request.id}] 快速对话完成")

        return {
            "code": 200,
            "message": "success",
            "data": {
                "success": True,
                "answer": answer,
                "errorMessage": None
            }
        }

    except Exception as e:
        logger.error(f"对话接口错误: {e}")
        return {
            "code": 500,
            "message": "error",
            "data": {
                "success": False,
                "answer": None,
                "errorMessage": str(e)
            }
        }
    finally:
        tenant_service.clear_context()


@router.post("/chat_stream")
async def chat_stream(
    request: ChatRequest,
    current_user: User = Depends(get_current_user)
):
    """流式对话接口（基于 RAG Agent，SSE）

    支持两种认证方式：
    1. Bearer Token（推荐）：从 Token 中自动获取 tenant_id
    2. 请求参数：从请求体中获取 tenant_id（兼容模式）

    Returns:
        SSE 事件流
    """
    if current_user:
        tenant_id = current_user.tenant_id
        user_id = current_user.id
        include_public = request.include_public
    else:
        tenant_id = request.tenant_id
        user_id = request.user_id
        include_public = request.include_public

    tenant_service.set_context(
        tenant_id=tenant_id,
        user_id=user_id,
        include_public=include_public
    )

    logger.info(f"[会话 {request.id}] 收到流式对话请求: {request.question}, tenant_id={tenant_id}")

    async def event_generator():
        try:
            async for chunk in rag_agent_service.query_stream(request.question, session_id=request.id):
                chunk_type = chunk.get("type", "unknown")
                chunk_data = chunk.get("data", None)

                if chunk_type == "debug":
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "debug",
                            "node": chunk.get("node", "unknown"),
                            "message_type": chunk.get("message_type", "unknown")
                        }, ensure_ascii=False)
                    }
                elif chunk_type == "tool_call":
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "tool_call",
                            "data": chunk_data
                        }, ensure_ascii=False)
                    }
                elif chunk_type == "search_results":
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "search_results",
                            "data": chunk_data
                        }, ensure_ascii=False)
                    }
                elif chunk_type == "content":
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "content",
                            "data": chunk_data
                        }, ensure_ascii=False)
                    }
                elif chunk_type == "complete":
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "done",
                            "data": chunk_data
                        }, ensure_ascii=False)
                    }
                elif chunk_type == "error":
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "error",
                            "data": str(chunk_data)
                        }, ensure_ascii=False)
                    }

            logger.info(f"[会话 {request.id}] 流式对话完成")

        except Exception as e:
            logger.error(f"流式对话接口错误: {e}")
            yield {
                "event": "message",
                "data": json.dumps({
                    "type": "error",
                    "data": str(e)
                }, ensure_ascii=False)
            }
        finally:
            tenant_service.clear_context()

    return EventSourceResponse(event_generator())


@router.post("/chat/clear", response_model=ApiResponse)
async def clear_session(request: ClearRequest):
    """清空会话历史

    Args:
        request: 清空请求

    Returns:
        操作结果
    """
    try:
        success = rag_agent_service.clear_session(request.session_id)
        logger.info(f"清空会话: {request.session_id}, 结果: {success}")

        return ApiResponse(
            status="success" if success else "error",
            message="会话已清空" if success else "清空会话失败",
            data=None
        )

    except Exception as e:
        logger.error(f"清空会话错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/session/{session_id}", response_model=SessionInfoResponse)
async def get_session_info(session_id: str) -> SessionInfoResponse:
    """查询会话历史

    Args:
        session_id: 会话 ID

    Returns:
        会话信息
    """
    try:
        history = rag_agent_service.get_session_history(session_id)

        return SessionInfoResponse(
            session_id=session_id,
            message_count=len(history),
            history=history
        )

    except Exception as e:
        logger.error(f"获取会话信息错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tenant/info")
async def get_tenant_info(tenant_id: str = None, user_id: str = None):
    """获取当前租户信息

    用于测试和调试，查看租户上下文设置情况

    Args:
        tenant_id: 租户ID（查询参数）
        user_id: 用户ID（查询参数）

    Returns:
        租户上下文信息
    """
    try:
        tenant_service.set_context(tenant_id=tenant_id, user_id=user_id)

        context = tenant_service.get_context()
        filter_expr = tenant_service.build_milvus_filter()

        return {
            "code": 200,
            "message": "success",
            "data": {
                "tenant_id": context.tenant_id,
                "user_id": context.user_id,
                "is_public_tenant": tenant_service.is_public_tenant(),
                "include_public": tenant_service.is_include_public(),
                "milvus_filter": filter_expr,
                "public_tenant_id": tenant_service.PUBLIC_TENANT_ID,
            }
        }
    finally:
        tenant_service.clear_context()
