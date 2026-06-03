"""Hybrid vector search service with parent-child result aggregation."""

import math
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

from loguru import logger

from app.core.milvus_client import milvus_manager
from app.services.vector_embedding_service import vector_embedding_service
from app.services.tenant_service import tenant_service


class SearchResult:
    """Search result."""

    def __init__(
        self,
        id: str,
        content: str,
        score: float,
        metadata: Dict[str, Any],
    ):
        self.id = id
        self.content = content
        self.score = score
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
        }


class BM25:
    """A small BM25 implementation for lexical recall."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: Dict[str, str] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length: float = 0
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.num_docs: int = 0

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(r"\w+", text)
        return [token for token in tokens if len(token) > 1]

    def fit(self, documents: List[Tuple[str, str]]):
        self.documents = {doc_id: content for doc_id, content in documents}
        self.num_docs = len(documents)

        doc_lengths = []
        doc_freqs = Counter()

        for doc_id, content in documents:
            tokens = self._tokenize(content)
            self.doc_lengths[doc_id] = len(tokens)
            doc_lengths.append(len(tokens))

            for token in set(tokens):
                doc_freqs[token] += 1

        self.avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0
        self.doc_freqs = dict(doc_freqs)

        for term, df in doc_freqs.items():
            self.idf[term] = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if not self.documents:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        doc_scores: Dict[str, float] = {}

        for doc_id, content in self.documents.items():
            doc_tokens = self._tokenize(content)
            doc_len = self.doc_lengths.get(doc_id, 0)
            doc_tf = Counter(doc_tokens)

            score = 0.0
            for term in query_tokens:
                if term not in self.idf:
                    continue

                tf = doc_tf.get(term, 0)
                idf = self.idf[term]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                score += idf * (numerator / denominator)

            if score > 0:
                doc_scores[doc_id] = score

        return sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]


class CrossEncoderReranker:
    """Optional reranker."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self.model = None
        self._loaded = False
        logger.info("Cross encoder reranker initialized: {}", model_name)

    def _load_model(self):
        if self._loaded:
            return

        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name)
            self._loaded = True
            logger.info("Cross encoder reranker loaded")
        except ImportError:
            logger.warning("sentence-transformers not installed, rerank disabled")
            self._loaded = False
        except Exception as exc:
            logger.warning("Cross encoder load failed: {}", exc)
            self._loaded = False

    def rerank(
        self,
        query: str,
        documents: List[SearchResult],
        top_k: int = 3,
    ) -> List[SearchResult]:
        if not documents or len(documents) <= top_k:
            return documents

        self._load_model()
        if not self._loaded or self.model is None:
            return documents[:top_k]

        try:
            pairs = [[query, doc.content] for doc in documents]
            scores = self.model.predict(pairs)

            for index, doc in enumerate(documents):
                doc.score = float(scores[index])

            return sorted(documents, key=lambda item: item.score, reverse=True)[:top_k]
        except Exception as exc:
            logger.warning("Rerank failed: {}", exc)
            return documents[:top_k]


class HybridSearchService:
    """Hybrid retrieval over child chunks with parent-level aggregation."""

    def __init__(
        self,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        use_rerank: bool = True,
        rerank_top_k: int = 10,
    ):
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.use_rerank = use_rerank
        self.rerank_top_k = rerank_top_k
        self.bm25 = BM25()
        self._bm25_loaded = False
        self.reranker = CrossEncoderReranker()

        logger.info(
            "Hybrid search initialized: vector_weight={}, bm25_weight={}, rerank={}",
            vector_weight,
            bm25_weight,
            use_rerank,
        )

    def _load_bm25_index(self):
        if self._bm25_loaded:
            return

        try:
            collection = milvus_manager.get_collection()
            tenant_filter = tenant_service.build_milvus_filter()

            all_docs = collection.query(
                expr=tenant_filter,
                output_fields=["id", "content", "metadata"],
                limit=10000,
            )

            if all_docs:
                grouped_documents: Dict[str, List[str]] = {}
                for doc in all_docs:
                    metadata = doc.get("metadata", {}) or {}
                    parent_id = metadata.get("_parent_id") or doc["id"]
                    grouped_documents.setdefault(parent_id, []).append(doc["content"])

                documents = [
                    (parent_id, "\n".join(contents))
                    for parent_id, contents in grouped_documents.items()
                ]
                self.bm25.fit(documents)
                self._bm25_loaded = True
                logger.info("BM25 index loaded with {} parent chunks", len(documents))
        except Exception as exc:
            logger.warning("BM25 index load failed: {}", exc)
            self._bm25_loaded = False

    def invalidate_bm25_index(self):
        """Force BM25 to rebuild on next search."""
        self.bm25 = BM25()
        self._bm25_loaded = False

    def search(
        self,
        query: str,
        top_k: int = 3,
        hybrid_top_k: int = 10,
    ) -> List[SearchResult]:
        try:
            vector_results = self._vector_search(query, hybrid_top_k)
            self._load_bm25_index()
            bm25_results = self._bm25_search(query, hybrid_top_k)
            fused_results = self._fusion(vector_results, bm25_results, hybrid_top_k)

            if self.use_rerank and len(fused_results) > top_k:
                candidates = fused_results[: self.rerank_top_k]
                return self.reranker.rerank(query, candidates, top_k)

            return fused_results[:top_k]
        except Exception as exc:
            logger.error("Hybrid search failed: {}, fallback to vector search", exc)
            return list(self._vector_search(query, top_k).values())[:top_k]

    def _vector_search(self, query: str, top_k: int) -> Dict[str, SearchResult]:
        query_vector = vector_embedding_service.embed_query(query)
        collection = milvus_manager.get_collection()
        tenant_filter = tenant_service.build_milvus_filter()
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }

        results = collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=tenant_filter,
            output_fields=["id", "content", "metadata"],
        )

        vector_dict = self._collect_parent_results(results)

        if not vector_dict:
            fallback_results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["id", "content", "metadata"],
            )
            vector_dict = self._collect_parent_results(fallback_results)

        return vector_dict

    def _collect_parent_results(self, results) -> Dict[str, SearchResult]:
        vector_dict: Dict[str, SearchResult] = {}

        for hits in results:
            for hit in hits:
                metadata = hit.entity.get("metadata", {}) or {}
                result_id = metadata.get("_parent_id") or hit.entity.get("id")
                result_content = metadata.get("_parent_content") or hit.entity.get("content")
                result = SearchResult(
                    id=result_id,
                    content=result_content,
                    score=hit.distance,
                    metadata=metadata,
                )
                existing = vector_dict.get(result.id)
                if existing is None or result.score < existing.score:
                    vector_dict[result.id] = result

        return vector_dict

    def _bm25_search(self, query: str, top_k: int) -> Dict[str, float]:
        if not self._bm25_loaded:
            return {}
        return {doc_id: score for doc_id, score in self.bm25.search(query, top_k)}

    def _fusion(
        self,
        vector_results: Dict[str, SearchResult],
        bm25_results: Dict[str, float],
        top_k: int,
    ) -> List[SearchResult]:
        if not vector_results and not bm25_results:
            return []
        if not bm25_results:
            return list(vector_results.values())[:top_k]
        if not vector_results:
            return []

        all_doc_ids = set(vector_results.keys()) | set(bm25_results.keys())
        max_vector_score = max(result.score for result in vector_results.values()) or 1.0
        max_bm25_score = max(bm25_results.values()) or 1.0

        fused_scores: Dict[str, float] = {}
        for doc_id in all_doc_ids:
            vector_score = 0.0
            if doc_id in vector_results:
                vector_score = 1.0 - (vector_results[doc_id].score / max_vector_score)

            bm25_score = 0.0
            if doc_id in bm25_results:
                bm25_score = bm25_results[doc_id] / max_bm25_score

            fused_scores[doc_id] = (
                self.vector_weight * vector_score + self.bm25_weight * bm25_score
            )

        top_doc_ids = [
            doc_id
            for doc_id, _score in sorted(
                fused_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:top_k]
        ]

        fused_results: List[SearchResult] = []
        for doc_id in top_doc_ids:
            if doc_id in vector_results:
                result = vector_results[doc_id]
                result.score = fused_scores[doc_id]
                fused_results.append(result)

        return fused_results


class VectorSearchService:
    """Search similar documents."""

    def __init__(self):
        self.hybrid_service = HybridSearchService()
        logger.info("Vector search service initialized")

    def invalidate_cache(self):
        """Invalidate search-side caches after reindexing."""
        self.hybrid_service.invalidate_bm25_index()

    def search_similar_documents(self, query: str, top_k: int = 3) -> List[SearchResult]:
        try:
            return self.hybrid_service.search(query, top_k=top_k)
        except Exception as exc:
            logger.error("Search similar documents failed: {}", exc)
            raise RuntimeError(f"Search failed: {exc}") from exc


vector_search_service = VectorSearchService()
