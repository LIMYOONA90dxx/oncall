"""Ragas RAG 评测服务

使用 Ragas 框架对 RAG 系统进行评测，支持多种评测指标：
- Faithfulness (忠实度): 答案是否基于检索到的上下文
- Answer Relevancy (答案相关性): 答案与问题的相关程度
- Context Precision (上下文精确度): 检索到的上下文是否精确
- Context Recall (上下文召回率): 检索到的上下文是否完整
- Answer Correctness (答案正确性): 答案的正确程度
- Answer Similarity (答案相似度): 答案与标准答案的相似程度
"""

from typing import Optional, Any
import asyncio
from loguru import logger

from app.models.evaluation import (
    EvaluationSample,
    EvaluationResult,
    EvaluationResponse,
)
from app.config import config


class RagasEvaluationService:
    """Ragas 评测服务 - 使用延迟导入避免启动时加载问题"""
    
    DEFAULT_METRICS = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    
    def __init__(self):
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._evaluate = None
        self._metrics = {}
        self._SingleTurnSample = None
        self._EvaluationDataset = None
    
    async def _ensure_initialized(self):
        """确保服务已初始化（延迟导入 ragas）"""
        if self._initialized:
            return
        
        async with self._init_lock:
            if self._initialized:
                return
            
            logger.info("Ragas 评测服务初始化中...")
            
            try:
                from ragas import evaluate
                from ragas.metrics import (
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                    answer_correctness,
                    answer_similarity,
                )
                from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
                from ragas.llms import LangchainLLMWrapper
                from ragas.embeddings import LangchainEmbeddingsWrapper
                from ragas import RunConfig
                
                self._evaluate = evaluate
                self._SingleTurnSample = SingleTurnSample
                self._EvaluationDataset = EvaluationDataset
                
                self._metrics = {
                    "faithfulness": faithfulness,
                    "answer_relevancy": answer_relevancy,
                    "context_precision": context_precision,
                    "context_recall": context_recall,
                    "answer_correctness": answer_correctness,
                    "answer_similarity": answer_similarity,
                }
                
                from langchain_openai import ChatOpenAI, OpenAIEmbeddings
                
                base_llm = ChatOpenAI(
                    model="deepseek-chat",
                    openai_api_key="sk-e219885341a7484889b5ee8299a0432d",
                    openai_api_base="https://api.deepseek.com/v1",
                    temperature=0,
                    model_kwargs={"extra_body": {"n": 1}},
                )
                
                base_embeddings = OpenAIEmbeddings(
                    model="text-embedding-v4",
                    openai_api_key=config.dashscope_api_key,
                    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                )
                
                llm = LangchainLLMWrapper(base_llm)
                embeddings = LangchainEmbeddingsWrapper(base_embeddings)
                
                run_config = RunConfig(
                    max_workers=1,
                    max_wait=60,
                )
                
                for metric in self._metrics.values():
                    metric.llm = llm
                    metric.embeddings = embeddings
                
                self._run_config = run_config
                self._initialized = True
                logger.info("Ragas 评测服务初始化完成")
                
            except ImportError as e:
                logger.error(f"Ragas 评测服务初始化失败，缺少依赖: {e}")
                raise RuntimeError(f"Ragas 评测服务不可用，请确保已安装 ragas 和相关依赖: {e}")
            except Exception as e:
                logger.error(f"Ragas 评测服务初始化失败: {e}")
                raise
    
    def _get_metrics(self, metric_names: Optional[list[str]] = None) -> list[Any]:
        """获取评测指标"""
        if metric_names is None:
            metric_names = self.DEFAULT_METRICS
        
        metrics = []
        for name in metric_names:
            if name in self._metrics:
                metrics.append(self._metrics[name])
            else:
                logger.warning(f"不支持的评测指标: {name}")
        
        return metrics
    
    def _samples_to_dataset(self, samples: list[EvaluationSample]) -> Any:
        """将评测样本转换为 Ragas 数据集"""
        ragas_samples = []
        
        for sample in samples:
            ragas_sample = self._SingleTurnSample(
                user_input=sample.question,
                response=sample.answer,
                retrieved_contexts=sample.contexts,
                reference=sample.ground_truth,
            )
            ragas_samples.append(ragas_sample)
        
        return self._EvaluationDataset(samples=ragas_samples)
    
    async def evaluate_samples(
        self,
        samples: list[EvaluationSample],
        metric_names: Optional[list[str]] = None
    ) -> EvaluationResponse:
        """评测样本
        
        Args:
            samples: 评测样本列表
            metric_names: 评测指标名称列表
            
        Returns:
            EvaluationResponse: 评测结果
        """
        await self._ensure_initialized()
        
        metrics = self._get_metrics(metric_names)
        if not metrics:
            raise ValueError("没有有效的评测指标")
        
        logger.info(f"开始评测 {len(samples)} 个样本，指标: {[m.name for m in metrics]}")
        
        dataset = self._samples_to_dataset(samples)
        
        try:
            results = self._evaluate(
                dataset=dataset,
                metrics=metrics,
                run_config=self._run_config,
            )
            
            logger.info(f"评测完成，结果: {results}")
            
            if isinstance(results, dict):
                avg_scores = results
            elif hasattr(results, 'to_pandas'):
                df = results.to_pandas()
                avg_scores = df.mean(numeric_only=True).to_dict()
            else:
                avg_scores = {}
            
            logger.info(f"平均分数: {avg_scores}")
            
            detailed_results = []
            for i, sample in enumerate(samples):
                result = EvaluationResult(
                    question=sample.question,
                    answer=sample.answer,
                    contexts=sample.contexts,
                    ground_truth=sample.ground_truth,
                )
                
                if isinstance(results, dict):
                    result.faithfulness = results.get('faithfulness')
                    result.answer_relevancy = results.get('answer_relevancy')
                    result.context_precision = results.get('context_precision')
                    result.context_recall = results.get('context_recall')
                    result.answer_correctness = results.get('answer_correctness')
                    result.answer_similarity = results.get('answer_similarity')
                elif hasattr(results, 'to_pandas'):
                    df = results.to_pandas()
                    if i < len(df):
                        row = df.iloc[i]
                        result.faithfulness = float(row.get('faithfulness', 0)) if 'faithfulness' in row else None
                        result.answer_relevancy = float(row.get('answer_relevancy', 0)) if 'answer_relevancy' in row else None
                        result.context_precision = float(row.get('context_precision', 0)) if 'context_precision' in row else None
                        result.context_recall = float(row.get('context_recall', 0)) if 'context_recall' in row else None
                        result.answer_correctness = float(row.get('answer_correctness', 0)) if 'answer_correctness' in row else None
                        result.answer_similarity = float(row.get('answer_similarity', 0)) if 'answer_similarity' in row else None
                
                detailed_results.append(result)
            
            response = EvaluationResponse(
                code=200,
                message="success",
                total_samples=len(samples),
                avg_faithfulness=float(avg_scores.get('faithfulness')) if avg_scores.get('faithfulness') is not None else None,
                avg_answer_relevancy=float(avg_scores.get('answer_relevancy')) if avg_scores.get('answer_relevancy') is not None else None,
                avg_context_precision=float(avg_scores.get('context_precision')) if avg_scores.get('context_precision') is not None else None,
                avg_context_recall=float(avg_scores.get('context_recall')) if avg_scores.get('context_recall') is not None else None,
                avg_answer_correctness=float(avg_scores.get('answer_correctness')) if avg_scores.get('answer_correctness') is not None else None,
                avg_answer_similarity=float(avg_scores.get('answer_similarity')) if avg_scores.get('answer_similarity') is not None else None,
                detailed_results=detailed_results,
            )
            
            return response
            
        except Exception as e:
            logger.error(f"评测失败: {e}")
            raise
    
    async def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: Optional[str] = None,
        metric_names: Optional[list[str]] = None
    ) -> EvaluationResult:
        """评测单个样本
        
        Args:
            question: 用户问题
            answer: 生成的回答
            contexts: 检索到的上下文
            ground_truth: 标准答案
            metric_names: 评测指标名称列表
            
        Returns:
            EvaluationResult: 评测结果
        """
        sample = EvaluationSample(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
        )
        
        response = await self.evaluate_samples([sample], metric_names)
        
        if response.detailed_results:
            return response.detailed_results[0]
        
        return EvaluationResult(question=question)


ragas_service = RagasEvaluationService()
