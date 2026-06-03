"""向量存储管理器 - 封装 Milvus VectorStore 操作（支持多租户隔离）"""

from typing import Any, List, Optional

from langchain_core.documents import Document
from langchain_milvus import Milvus
from loguru import logger

from app.config import config
from app.services.vector_embedding_service import vector_embedding_service
from app.services.tenant_service import tenant_service


COLLECTION_NAME = "biz"


class VectorStoreManager:
    """向量存储管理器"""

    def __init__(self):
        """初始化向量存储管理器"""
        self.vector_store: Milvus | None = None
        self.collection_name = COLLECTION_NAME
        self._initialized = False

    def _ensure_vector_store(self):
        """延迟初始化 VectorStore"""
        if self._initialized and self.vector_store is not None:
            return

        try:
            from app.core.milvus_client import milvus_manager
            milvus_manager.connect()

            connection_args = {
                "host": config.milvus_host,
                "port": config.milvus_port,
            }

            self.vector_store = Milvus(
                embedding_function=vector_embedding_service,
                collection_name=self.collection_name,
                connection_args=connection_args,
                auto_id=False,
                drop_old=False,
                text_field="content",
                vector_field="vector",
                primary_field="id",
                metadata_field="metadata",
            )

            self._initialized = True
            logger.info(
                f"VectorStore 初始化成功: {config.milvus_host}:{config.milvus_port}, "
                f"collection: {self.collection_name}"
            )

        except Exception as e:
            logger.error(f"VectorStore 初始化失败: {e}")
            raise

    def add_documents(self, documents: List[Document], is_public: bool = False) -> List[str]:
        """
        批量添加文档到向量存储（自动添加租户隔离元数据）

        Args:
            documents: 文档列表
            is_public: 是否为公共知识库

        Returns:
            List[str]: 文档 ID 列表
        """
        try:
            from app.core.milvus_client import milvus_manager
            milvus_manager.connect()
            collection = milvus_manager.get_collection()

            import time
            import uuid
            start_time = time.time()

            all_ids = []
            all_entities = []
            texts_to_embed = [doc.page_content for doc in documents]
            embeddings = vector_embedding_service.embed_documents(texts_to_embed)

            if len(embeddings) != len(documents):
                raise RuntimeError("Embedding count does not match document count")

            for doc, embedding in zip(documents, embeddings):
                doc_id = str(uuid.uuid4())
                all_ids.append(doc_id)

                enriched_metadata = tenant_service.enrich_metadata(doc.metadata, is_public=is_public)

                entity = {
                    "id": doc_id,
                    "content": doc.page_content,
                    "vector": embedding,
                    "metadata": enriched_metadata
                }
                all_entities.append(entity)

            batch_size = 10
            for i in range(0, len(all_entities), batch_size):
                batch = all_entities[i:i + batch_size]
                collection.insert(batch)

            elapsed = time.time() - start_time
            tenant_id = tenant_service.get_tenant_id()
            knowledge_type = "公共" if is_public else "私有"
            logger.info(
                f"批量添加 {len(documents)} 个文档到 VectorStore 完成, "
                f"类型={knowledge_type}, tenant_id={tenant_id}, 耗时: {elapsed:.2f}秒, 平均: {elapsed/len(documents):.2f}秒/个"
            )
            return all_ids

        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise

    def delete_by_source(self, file_path: str, is_public: bool = False) -> int:
        """
        删除指定文件的所有文档（带租户隔离）

        Args:
            file_path: 文件路径
            is_public: 是否为公共知识库

        Returns:
            int: 删除的文档数量
        """
        try:
            from app.core.milvus_client import milvus_manager
            collection = milvus_manager.get_collection()

            if is_public:
                base_filter = f'metadata["_tenant_id"] == "{tenant_service.PUBLIC_TENANT_ID}"'
            else:
                base_filter = tenant_service.build_milvus_filter()

            file_filter = f'{base_filter} and metadata["_source"] == "{file_path}"'

            result: Any = collection.delete(file_filter)
            deleted_count = getattr(result, "delete_count", 0)

            tenant_id = tenant_service.get_tenant_id()
            knowledge_type = "公共" if is_public else "私有"
            logger.info(f"删除文件旧数据: {file_path}, 类型={knowledge_type}, tenant_id={tenant_id}, 删除数量: {deleted_count}")
            return deleted_count

        except Exception as e:
            logger.warning(f"删除旧数据失败 (可能是首次索引): {e}")
            return 0

    def get_source_file_hash(self, file_path: str, is_public: bool = False) -> Optional[str]:
        """获取已索引文件的内容哈希，用于跳过未变化的重复索引。"""
        try:
            from app.core.milvus_client import milvus_manager
            collection = milvus_manager.get_collection()

            if is_public:
                base_filter = f'metadata["_tenant_id"] == "{tenant_service.PUBLIC_TENANT_ID}"'
            else:
                base_filter = tenant_service.build_milvus_filter()

            file_filter = f'{base_filter} and metadata["_source"] == "{file_path}"'
            results = collection.query(
                expr=file_filter,
                output_fields=["metadata"],
                limit=1,
            )

            if not results:
                return None

            metadata = results[0].get("metadata", {}) or {}
            return metadata.get("_file_hash")

        except Exception as e:
            logger.warning(f"鑾峰彇鏂囦欢鍝堝笇澶辫触: {file_path}, 閿欒: {e}")
            return None

    def get_vector_store(self) -> Milvus:
        """
        获取 VectorStore 实例

        Returns:
            Milvus: VectorStore 实例
        """
        self._ensure_vector_store()
        if self.vector_store is None:
            raise RuntimeError("VectorStore not initialized")
        return self.vector_store

    def similarity_search(
        self,
        query: str,
        k: int = 3,
        filter_expr: Optional[str] = None,
        enforce_tenant: bool = True
    ) -> List[Document]:
        """
        相似度搜索（默认强制租户隔离）

        Args:
            query: 查询文本
            k: 返回结果数量
            filter_expr: 额外的过滤条件
            enforce_tenant: 是否强制租户过滤（默认 True，禁止关闭）

        Returns:
            List[Document]: 相关文档列表
        """
        self._ensure_vector_store()
        try:
            if enforce_tenant or tenant_service.is_enabled():
                final_filter = tenant_service.build_milvus_filter(filter_expr)
            else:
                final_filter = filter_expr

            docs = self.vector_store.similarity_search(
                query,
                k=k,
                filter=final_filter
            )

            tenant_id = tenant_service.get_tenant_id()
            logger.debug(f"相似度搜索完成: query='{query}', tenant_id={tenant_id}, 结果数={len(docs)}")
            return docs
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            return []


# 全局单例
vector_store_manager = VectorStoreManager()
