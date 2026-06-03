"""自动评测数据生成服务

从知识库中自动生成评测数据，包括：
1. 从向量数据库获取文档片段
2. 使用 LLM 生成问答对
3. 运行 RAG 系统获取答案和上下文
4. 进行评测
"""

from typing import Optional
import asyncio
import random
from loguru import logger

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from app.config import config
from app.services.vector_store_manager import vector_store_manager
from app.services.rag_agent_service import rag_agent_service
from app.services.tenant_service import tenant_service
from app.services.ragas_evaluation_service import ragas_service
from app.models.evaluation import EvaluationSample, EvaluationResponse


class AutoEvaluationService:
    """自动评测数据生成服务"""
    
    def __init__(self):
        self._llm = None
    
    def _get_llm(self):
        """获取 LLM 实例"""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model="deepseek-chat",
                openai_api_key="sk-e219885341a7484889b5ee8299a0432d",
                openai_api_base="https://api.deepseek.com/v1",
                temperature=0.7,
            )
        return self._llm
    
    async def get_documents_from_knowledge_base(
        self,
        limit: int = 10,
        source_filter: Optional[str] = None
    ) -> list[Document]:
        """从知识库获取文档
        
        Args:
            limit: 获取文档数量
            source_filter: 来源文件过滤
            
        Returns:
            文档列表
        """
        try:
            from app.core.milvus_client import milvus_manager
            milvus_manager.connect()
            collection = milvus_manager.get_collection()
            
            query_result = collection.query(
                expr='',
                output_fields=["content", "metadata"],
                limit=limit
            )
            
            documents = []
            if query_result:
                for item in query_result:
                    doc = Document(
                        page_content=item.get('content', ''),
                        metadata=item.get('metadata', {})
                    )
                    if source_filter:
                        if source_filter in doc.metadata.get('source', ''):
                            documents.append(doc)
                    else:
                        documents.append(doc)
            
            logger.info(f"从知识库获取了 {len(documents)} 个文档")
            return documents
            
        except Exception as e:
            logger.error(f"获取文档失败: {e}")
            return []
    
    async def generate_qa_pairs(
        self,
        documents: list[Document],
        num_questions: int = 3
    ) -> list[dict]:
        """从文档生成问答对
        
        Args:
            documents: 文档列表
            num_questions: 每个文档生成的问题数量
            
        Returns:
            问答对列表 [{"question": "...", "ground_truth": "...", "context": "..."}]
        """
        llm = self._get_llm()
        qa_pairs = []
        
        for doc in documents[:5]:
            context = doc.page_content[:2000]
            
            prompt = f"""基于以下文本内容，生成 {num_questions} 个问题和对应的答案。

文本内容：
{context}

请按照以下 JSON 格式输出：
[
  {{"question": "问题1", "answer": "答案1"}},
  {{"question": "问题2", "answer": "答案2"}},
  {{"question": "问题3", "answer": "答案3"}}
]

要求：
1. 问题应该基于文本内容，具有实际意义
2. 答案应该准确、简洁，完全来自文本内容
3. 问题类型要多样化（事实性问题、概念解释、对比分析等）
4. 只输出 JSON 数组，不要有其他内容"""

            try:
                response = await llm.ainvoke(prompt)
                content = response.content
                
                import json
                import re
                
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    qa_list = json.loads(json_match.group())
                    for qa in qa_list:
                        qa_pairs.append({
                            "question": qa.get("question", ""),
                            "ground_truth": qa.get("answer", ""),
                            "context": context
                        })
                
            except Exception as e:
                logger.error(f"生成问答对失败: {e}")
                continue
        
        logger.info(f"生成了 {len(qa_pairs)} 个问答对")
        return qa_pairs
    
    async def run_rag_and_evaluate(
        self,
        qa_pairs: list[dict],
        metrics: Optional[list[str]] = None
    ) -> EvaluationResponse:
        """运行 RAG 系统并评测
        
        Args:
            qa_pairs: 问答对列表
            metrics: 评测指标
            
        Returns:
            评测结果
        """
        from app.services.vector_search_service import vector_search_service
        
        evaluation_samples = []
        
        for qa in qa_pairs:
            question = qa["question"]
            ground_truth = qa.get("ground_truth", "")
            
            try:
                answer = await rag_agent_service.query(
                    session_id="auto_eval",
                    question=question
                )
                
                contexts = []
                try:
                    search_results = vector_search_service.search_similar_documents(question, top_k=3)
                    contexts = [result.content for result in search_results]
                except Exception as e:
                    logger.warning(f"获取上下文失败: {e}")
                
                sample = EvaluationSample(
                    question=question,
                    answer=answer if answer else "",
                    contexts=contexts if contexts else None,
                    ground_truth=ground_truth if ground_truth else None,
                )
                evaluation_samples.append(sample)
                
            except Exception as e:
                logger.error(f"RAG 查询失败: {e}")
                continue
        
        if not evaluation_samples:
            raise ValueError("没有有效的评测样本")
        
        response = await ragas_service.evaluate_samples(
            samples=evaluation_samples,
            metric_names=metrics
        )
        
        return response
    
    async def auto_evaluate(
        self,
        num_samples: int = 5,
        source_filter: Optional[str] = None,
        metrics: Optional[list[str]] = None
    ) -> EvaluationResponse:
        """自动评测完整流程
        
        Args:
            num_samples: 评测样本数量
            source_filter: 来源文件过滤
            metrics: 评测指标
            
        Returns:
            评测结果
        """
        logger.info(f"开始自动评测，目标样本数: {num_samples}")
        
        documents = await self.get_documents_from_knowledge_base(
            limit=num_samples * 2,
            source_filter=source_filter
        )
        
        if not documents:
            raise ValueError("知识库中没有文档，请先上传文件")
        
        qa_pairs = await self.generate_qa_pairs(
            documents=documents,
            num_questions=1
        )
        
        if not qa_pairs:
            raise ValueError("无法生成问答对")
        
        qa_pairs = qa_pairs[:num_samples]
        
        response = await self.run_rag_and_evaluate(
            qa_pairs=qa_pairs,
            metrics=metrics
        )
        
        return response


auto_eval_service = AutoEvaluationService()
