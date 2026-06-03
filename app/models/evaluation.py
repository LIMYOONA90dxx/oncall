"""RAG 评测数据模型"""

from typing import Optional
from pydantic import BaseModel, Field


class EvaluationSample(BaseModel):
    """评测样本"""
    question: str = Field(..., description="用户问题")
    answer: Optional[str] = Field(None, description="生成的回答")
    contexts: Optional[list[str]] = Field(None, description="检索到的上下文")
    ground_truth: Optional[str] = Field(None, description="标准答案")


class EvaluationRequest(BaseModel):
    """评测请求"""
    samples: list[EvaluationSample] = Field(..., description="评测样本列表")
    metrics: Optional[list[str]] = Field(
        default=None,
        description="评测指标列表，默认使用全部指标"
    )


class EvaluationResult(BaseModel):
    """单个评测结果"""
    question: str
    answer: Optional[str] = None
    contexts: Optional[list[str]] = None
    ground_truth: Optional[str] = None
    
    faithfulness: Optional[float] = Field(None, description="忠实度")
    answer_relevancy: Optional[float] = Field(None, description="答案相关性")
    context_precision: Optional[float] = Field(None, description="上下文精确度")
    context_recall: Optional[float] = Field(None, description="上下文召回率")
    answer_correctness: Optional[float] = Field(None, description="答案正确性")
    answer_similarity: Optional[float] = Field(None, description="答案相似度")


class EvaluationResponse(BaseModel):
    """评测响应"""
    code: int = 200
    message: str = "success"
    total_samples: int = Field(..., description="总样本数")
    avg_faithfulness: Optional[float] = Field(None, description="平均忠实度")
    avg_answer_relevancy: Optional[float] = Field(None, description="平均答案相关性")
    avg_context_precision: Optional[float] = Field(None, description="平均上下文精确度")
    avg_context_recall: Optional[float] = Field(None, description="平均上下文召回率")
    avg_answer_correctness: Optional[float] = Field(None, description="平均答案正确性")
    avg_answer_similarity: Optional[float] = Field(None, description="平均答案相似度")
    detailed_results: list[EvaluationResult] = Field(default_factory=list, description="详细结果")


class RAGTestSample(BaseModel):
    """RAG 测试样本（用于自动生成评测数据）"""
    question: str = Field(..., description="用户问题")
    ground_truth: Optional[str] = Field(None, description="标准答案")


class RAGTestRequest(BaseModel):
    """RAG 测试请求（自动运行 RAG 并评测）"""
    samples: list[RAGTestSample] = Field(..., description="测试样本列表")
    metrics: Optional[list[str]] = Field(
        default=None,
        description="评测指标列表"
    )
