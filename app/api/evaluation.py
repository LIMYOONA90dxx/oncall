"""RAG 评测 API 路由"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional

from loguru import logger

from app.models.evaluation import (
    EvaluationRequest,
    EvaluationResponse,
    EvaluationSample,
    RAGTestRequest,
    EvaluationResult,
)
from app.models.user import User
from app.services.auth_service import require_auth
from app.services.rag_agent_service import rag_agent_service
from app.services.tenant_service import tenant_service
from pydantic import BaseModel, Field

router = APIRouter()


def _get_ragas_service():
    """延迟导入 ragas 服务"""
    from app.services.ragas_evaluation_service import ragas_service
    return ragas_service


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_rag(
    request: EvaluationRequest,
    current_user: User = Depends(require_auth)
):
    """评测 RAG 样本
    
    提供问题、答案、上下文和标准答案，评测 RAG 系统的质量
    
    支持的评测指标：
    - faithfulness: 忠实度，答案是否基于检索到的上下文
    - answer_relevancy: 答案相关性，答案与问题的相关程度
    - context_precision: 上下文精确度，检索到的上下文是否精确
    - context_recall: 上下文召回率，检索到的上下文是否完整
    - answer_correctness: 答案正确性，答案的正确程度
    - answer_similarity: 答案相似度，答案与标准答案的相似程度
    
    需要登录认证
    """
    try:
        if not request.samples:
            raise HTTPException(status_code=400, detail="评测样本不能为空")
        
        for i, sample in enumerate(request.samples):
            if not sample.question:
                raise HTTPException(status_code=400, detail=f"样本 {i+1} 的问题不能为空")
        
        logger.info(
            f"收到评测请求: {len(request.samples)} 个样本, "
            f"指标: {request.metrics or '默认'}, "
            f"tenant_id={current_user.tenant_id}"
        )
        
        ragas_service = _get_ragas_service()
        response = await ragas_service.evaluate_samples(
            samples=request.samples,
            metric_names=request.metrics
        )
        
        return response
        
    except ValueError as e:
        logger.error(f"评测参数错误: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"评测服务不可用: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"评测失败: {e}")
        raise HTTPException(status_code=500, detail=f"评测失败: {str(e)}")


@router.post("/test", response_model=EvaluationResponse)
async def test_rag(
    request: RAGTestRequest,
    current_user: User = Depends(require_auth)
):
    """自动运行 RAG 并评测
    
    提供问题和标准答案，自动运行 RAG 系统获取答案和上下文，然后评测
    
    需要登录认证
    """
    from app.services.vector_search_service import vector_search_service
    
    try:
        if not request.samples:
            raise HTTPException(status_code=400, detail="测试样本不能为空")
        
        tenant_service.set_context(
            tenant_id=current_user.tenant_id,
            user_id=current_user.id,
            include_public=True
        )
        
        logger.info(
            f"收到 RAG 测试请求: {len(request.samples)} 个样本, "
            f"tenant_id={current_user.tenant_id}"
        )
        
        evaluation_samples = []
        
        for sample in request.samples:
            answer = await rag_agent_service.query(
                session_id=f"eval_{current_user.id}",
                question=sample.question,
            )
            
            contexts = []
            try:
                search_results = vector_search_service.search_similar_documents(sample.question, top_k=3)
                contexts = [result.content for result in search_results]
            except Exception as e:
                logger.warning(f"获取上下文失败: {e}")
            
            eval_sample = EvaluationSample(
                question=sample.question,
                answer=answer if answer else "",
                contexts=contexts if contexts else None,
                ground_truth=sample.ground_truth,
            )
            evaluation_samples.append(eval_sample)
        
        ragas_service = _get_ragas_service()
        response = await ragas_service.evaluate_samples(
            samples=evaluation_samples,
            metric_names=request.metrics
        )
        
        return response
        
    except ValueError as e:
        logger.error(f"测试参数错误: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"评测服务不可用: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"RAG 测试失败: {e}")
        raise HTTPException(status_code=500, detail=f"RAG 测试失败: {str(e)}")
    finally:
        tenant_service.clear_context()


@router.post("/evaluate/single", response_model=EvaluationResult)
async def evaluate_single(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: Optional[str] = None,
    metrics: Optional[list[str]] = None,
    current_user: User = Depends(require_auth)
):
    """评测单个 RAG 样本
    
    简化的单样本评测接口
    
    需要登录认证
    """
    try:
        ragas_service = _get_ragas_service()
        result = await ragas_service.evaluate_single(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
            metric_names=metrics
        )
        
        return result
        
    except RuntimeError as e:
        logger.error(f"评测服务不可用: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"单样本评测失败: {e}")
        raise HTTPException(status_code=500, detail=f"评测失败: {str(e)}")


@router.get("/metrics")
async def list_metrics():
    """列出支持的评测指标"""
    return {
        "code": 200,
        "message": "success",
        "data": {
            "metrics": [
                {
                    "name": "faithfulness",
                    "description": "忠实度，衡量答案是否基于检索到的上下文",
                    "requires": ["question", "answer", "contexts"],
                    "range": "0-1"
                },
                {
                    "name": "answer_relevancy",
                    "description": "答案相关性，衡量答案与问题的相关程度",
                    "requires": ["question", "answer"],
                    "range": "0-1"
                },
                {
                    "name": "context_precision",
                    "description": "上下文精确度，衡量检索到的上下文是否精确",
                    "requires": ["question", "contexts", "ground_truth"],
                    "range": "0-1"
                },
                {
                    "name": "context_recall",
                    "description": "上下文召回率，衡量检索到的上下文是否完整",
                    "requires": ["question", "contexts", "ground_truth"],
                    "range": "0-1"
                },
                {
                    "name": "answer_correctness",
                    "description": "答案正确性，衡量答案的正确程度",
                    "requires": ["question", "answer", "ground_truth"],
                    "range": "0-1"
                },
                {
                    "name": "answer_similarity",
                    "description": "答案相似度，衡量答案与标准答案的相似程度",
                    "requires": ["answer", "ground_truth"],
                    "range": "0-1"
                }
            ],
            "default_metrics": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        }
    }


class AutoEvaluateRequest(BaseModel):
    """自动评测请求"""
    num_samples: int = Field(default=5, ge=1, le=20, description="评测样本数量")
    source_filter: Optional[str] = Field(default=None, description="来源文件过滤")
    metrics: Optional[list[str]] = Field(default=None, description="评测指标列表")


@router.post("/auto", response_model=EvaluationResponse)
async def auto_evaluate(
    request: AutoEvaluateRequest = AutoEvaluateRequest(),
    current_user: User = Depends(require_auth)
):
    """自动从知识库生成评测数据并评测
    
    流程：
    1. 从知识库获取文档片段
    2. 使用 LLM 生成问答对
    3. 运行 RAG 系统获取答案和上下文
    4. 进行评测
    
    需要登录认证
    """
    try:
        from app.services.auto_evaluation_service import auto_eval_service
        
        tenant_service.set_context(
            tenant_id=current_user.tenant_id,
            user_id=current_user.id,
            include_public=True
        )
        
        logger.info(
            f"收到自动评测请求: num_samples={request.num_samples}, "
            f"source_filter={request.source_filter}, "
            f"tenant_id={current_user.tenant_id}"
        )
        
        response = await auto_eval_service.auto_evaluate(
            num_samples=request.num_samples,
            source_filter=request.source_filter,
            metrics=request.metrics
        )
        
        return response
        
    except ValueError as e:
        logger.error(f"自动评测参数错误: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"评测服务不可用: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"自动评测失败: {e}")
        raise HTTPException(status_code=500, detail=f"自动评测失败: {str(e)}")
    finally:
        tenant_service.clear_context()


@router.get("/knowledge-base/stats")
async def get_knowledge_base_stats(
    current_user: User = Depends(require_auth)
):
    """获取知识库统计信息
    
    需要登录认证
    """
    try:
        from app.core.milvus_client import milvus_manager
        
        tenant_service.set_context(
            tenant_id=current_user.tenant_id,
            user_id=current_user.id,
            include_public=True
        )
        
        milvus_manager.connect()
        collection = milvus_manager.get_collection()
        stats = collection.num_entities
        
        return {
            "code": 200,
            "message": "success",
            "data": {
                "total_documents": stats,
                "tenant_id": current_user.tenant_id
            }
        }
        
    except Exception as e:
        logger.error(f"获取知识库统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取统计失败: {str(e)}")
    finally:
        tenant_service.clear_context()
