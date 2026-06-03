"""Document splitting service with parent-child chunking."""

import hashlib
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from loguru import logger

from app.config import config


class DocumentSplitterService:
    """Split documents into retrievable child chunks with parent context."""

    def __init__(self):
        self.parent_chunk_size = config.parent_chunk_size
        self.child_chunk_size = config.chunk_max_size
        self.chunk_overlap = config.chunk_overlap

        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
            ],
            strip_headers=False,
        )

        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        logger.info(
            "Document splitter initialized: parent_chunk_size={}, child_chunk_size={}, overlap={}",
            self.parent_chunk_size,
            self.child_chunk_size,
            self.chunk_overlap,
        )

    def split_markdown(self, content: str, file_path: str = "") -> List[Document]:
        """Split markdown by headings first, then build parent-child chunks."""
        if not content or not content.strip():
            logger.warning("Markdown content is empty: {}", file_path)
            return []

        try:
            heading_docs = self.markdown_splitter.split_text(content)
            parent_docs = self.parent_splitter.split_documents(heading_docs)
            final_docs = self._build_parent_child_chunks(parent_docs, file_path, ".md")
            logger.info("Markdown split complete: {} -> {} child chunks", file_path, len(final_docs))
            return final_docs
        except Exception as exc:
            logger.error("Markdown split failed: {}, error={}", file_path, exc)
            raise

    def split_text(self, content: str, file_path: str = "") -> List[Document]:
        """Split plain text into parent-child chunks."""
        if not content or not content.strip():
            logger.warning("Text content is empty: {}", file_path)
            return []

        try:
            parent_docs = self.parent_splitter.create_documents(
                texts=[content],
                metadatas=[
                    {
                        "_source": file_path,
                        "_extension": Path(file_path).suffix,
                        "_file_name": Path(file_path).name,
                    }
                ],
            )

            docs = self._build_parent_child_chunks(
                parent_docs,
                file_path,
                Path(file_path).suffix,
            )
            logger.info("Text split complete: {} -> {} child chunks", file_path, len(docs))
            return docs
        except Exception as exc:
            logger.error("Text split failed: {}, error={}", file_path, exc)
            raise

    def split_document(self, content: str, file_path: str = "") -> List[Document]:
        """Route to the appropriate splitter by file extension."""
        if file_path.endswith(".md"):
            return self.split_markdown(content, file_path)
        return self.split_text(content, file_path)

    def _build_parent_child_chunks(
        self,
        parent_docs: List[Document],
        file_path: str,
        extension: str,
    ) -> List[Document]:
        """Build child chunks and attach parent context metadata."""
        normalized_parents = self._merge_small_chunks(parent_docs, min_size=300)
        child_docs: List[Document] = []

        for parent_index, parent_doc in enumerate(normalized_parents):
            parent_content = parent_doc.page_content.strip()
            if not parent_content:
                continue

            parent_id = self._build_parent_id(file_path, parent_index, parent_content)
            parent_metadata = dict(parent_doc.metadata)
            parent_metadata["_source"] = file_path
            parent_metadata["_extension"] = extension
            parent_metadata["_file_name"] = Path(file_path).name

            raw_child_docs = self.child_splitter.split_documents([parent_doc])
            normalized_children = self._merge_small_chunks(
                raw_child_docs,
                min_size=max(150, self.child_chunk_size // 3),
            )

            for child_index, child_doc in enumerate(normalized_children):
                child_content = child_doc.page_content.strip()
                if not child_content:
                    continue

                child_metadata = dict(parent_metadata)
                child_metadata.update(child_doc.metadata)
                child_metadata["_chunk_type"] = "child"
                child_metadata["_parent_id"] = parent_id
                child_metadata["_parent_index"] = parent_index
                child_metadata["_child_index"] = child_index
                child_metadata["_parent_content"] = parent_content

                child_docs.append(
                    Document(page_content=child_content, metadata=child_metadata)
                )

        return child_docs

    def _build_parent_id(self, file_path: str, parent_index: int, content: str) -> str:
        digest = hashlib.md5(
            f"{file_path}:{parent_index}:{content}".encode("utf-8")
        ).hexdigest()
        return f"parent_{digest}"

    def _merge_small_chunks(
        self,
        documents: List[Document],
        min_size: int = 300,
    ) -> List[Document]:
        """Merge undersized chunks to reduce fragmentation."""
        if not documents:
            return []

        merged_docs: List[Document] = []
        current_doc: Document | None = None
        merge_limit = max(self.parent_chunk_size, self.child_chunk_size)

        for doc in documents:
            doc_size = len(doc.page_content)

            if current_doc is None:
                current_doc = doc
            elif doc_size < min_size and len(current_doc.page_content) < merge_limit:
                current_doc.page_content += "\n\n" + doc.page_content
            else:
                merged_docs.append(current_doc)
                current_doc = doc

        if current_doc is not None:
            merged_docs.append(current_doc)

        return merged_docs


document_splitter_service = DocumentSplitterService()
