"""文本清洗服务 - 清理文档中的噪音数据"""

import re
from typing import List

from langchain_core.documents import Document
from loguru import logger


class TextCleanerService:
    """文本清洗服务 - 清理文档中的噪音数据"""

    def __init__(self):
        """初始化文本清洗服务"""
        self._compiled_patterns = {
            "multiple_spaces": re.compile(r"[ \t]+"),
            "multiple_newlines": re.compile(r"\n{3,}"),
            "control_chars": re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"),
            "unicode_control": re.compile(r"[\u200b-\u200f\u2028-\u202f\u205f-\u206f]"),
            "page_numbers": re.compile(r"^\s*\d+\s*$", re.MULTILINE),
            "common_headers": re.compile(
                r"^(第\s*\d+\s*页|Page\s*\d+|页码[:：]?\s*\d+)\s*$",
                re.MULTILINE | re.IGNORECASE
            ),
        }
        logger.info("文本清洗服务初始化完成")

    def clean_text(self, text: str, aggressive: bool = False) -> str:
        """
        清洗文本内容

        Args:
            text: 原始文本
            aggressive: 是否使用激进清洗模式（会移除更多内容）

        Returns:
            str: 清洗后的文本
        """
        if not text:
            return ""

        cleaned = text

        cleaned = self._compiled_patterns["control_chars"].sub("", cleaned)
        cleaned = self._compiled_patterns["unicode_control"].sub("", cleaned)

        cleaned = self._compiled_patterns["multiple_spaces"].sub(" ", cleaned)

        cleaned = self._compiled_patterns["multiple_newlines"].sub("\n\n", cleaned)

        cleaned = self._fix_encoding_issues(cleaned)

        if aggressive:
            cleaned = self._compiled_patterns["page_numbers"].sub("", cleaned)
            cleaned = self._compiled_patterns["common_headers"].sub("", cleaned)

        cleaned = self._remove_repeated_lines(cleaned)

        cleaned = cleaned.strip()

        return cleaned

    def clean_document(self, document: Document, aggressive: bool = False) -> Document:
        """
        清洗单个文档

        Args:
            document: 原始文档
            aggressive: 是否使用激进清洗模式

        Returns:
            Document: 清洗后的文档
        """
        cleaned_content = self.clean_text(document.page_content, aggressive=aggressive)
        return Document(
            page_content=cleaned_content,
            metadata=document.metadata
        )

    def clean_documents(self, documents: List[Document], aggressive: bool = False) -> List[Document]:
        """
        批量清洗文档

        Args:
            documents: 文档列表
            aggressive: 是否使用激进清洗模式

        Returns:
            List[Document]: 清洗后的文档列表
        """
        cleaned_docs = []
        for doc in documents:
            cleaned_doc = self.clean_document(doc, aggressive=aggressive)
            if cleaned_doc.page_content.strip():
                cleaned_docs.append(cleaned_doc)

        logger.info(f"文本清洗完成: {len(documents)} -> {len(cleaned_docs)} 个文档")
        return cleaned_docs

    def _fix_encoding_issues(self, text: str) -> str:
        """修复常见的编码问题"""
        replacements = {
            "\ufffd": "",
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2013": "-",
            "\u2014": "-",
            "\u2026": "...",
            "\u00a0": " ",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

    def _remove_repeated_lines(self, text: str, threshold: int = 3) -> str:
        """
        移除重复的行（常见于页眉页脚）

        Args:
            text: 文本内容
            threshold: 重复次数阈值

        Returns:
            str: 处理后的文本
        """
        lines = text.split("\n")
        line_counts = {}
        for line in lines:
            stripped = line.strip()
            if stripped and len(stripped) < 100:
                line_counts[stripped] = line_counts.get(stripped, 0) + 1

        repeated_lines = {
            line for line, count in line_counts.items()
            if count >= threshold
        }

        if repeated_lines:
            filtered_lines = [
                line for line in lines
                if line.strip() not in repeated_lines
            ]
            return "\n".join(filtered_lines)

        return text

    def extract_main_content(self, text: str, min_paragraph_length: int = 50) -> str:
        """
        提取主要内容（移除过短的段落）

        Args:
            text: 文本内容
            min_paragraph_length: 最小段落长度

        Returns:
            str: 主要内容
        """
        paragraphs = text.split("\n\n")
        main_paragraphs = [
            p.strip() for p in paragraphs
            if len(p.strip()) >= min_paragraph_length
        ]
        return "\n\n".join(main_paragraphs)


text_cleaner_service = TextCleanerService()
