"""文档加载器服务 - 支持多种文档格式"""

from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from loguru import logger

from app.services.text_cleaner_service import text_cleaner_service


class DocumentLoaderService:
    """文档加载器服务 - 支持 TXT、Markdown、PDF、Word 等格式"""

    SUPPORTED_EXTENSIONS = {
        ".txt": "text",
        ".md": "markdown",
        ".pdf": "pdf",
        ".docx": "word",
        ".doc": "word",
    }

    def __init__(self):
        """初始化文档加载器服务"""
        self._pdf_loader_available = self._check_pdf_loader()
        self._word_loader_available = self._check_word_loader()
        logger.info(
            f"文档加载器服务初始化完成, "
            f"PDF支持: {self._pdf_loader_available}, "
            f"Word支持: {self._word_loader_available}"
        )

    def _check_pdf_loader(self) -> bool:
        """检查 PDF 加载器是否可用"""
        try:
            from langchain_community.document_loaders import PyPDFLoader
            return True
        except ImportError:
            logger.warning("PyPDFLoader 不可用，请安装 pypdf: pip install pypdf")
            return False

    def _check_word_loader(self) -> bool:
        """检查 Word 加载器是否可用"""
        try:
            from langchain_community.document_loaders import Docx2txtLoader
            return True
        except ImportError:
            logger.warning("Docx2txtLoader 不可用，请安装 docx2txt: pip install docx2txt")
            return False

    def get_supported_extensions(self) -> List[str]:
        """获取支持的文件扩展名列表"""
        extensions = [".txt", ".md"]
        if self._pdf_loader_available:
            extensions.append(".pdf")
        if self._word_loader_available:
            extensions.extend([".docx", ".doc"])
        return extensions

    def load_document(self, file_path: str, clean: bool = True) -> List[Document]:
        """
        加载文档（根据文件类型自动选择加载器）

        Args:
            file_path: 文件路径
            clean: 是否清洗文本内容

        Returns:
            List[Document]: 文档列表

        Raises:
            ValueError: 不支持的文件类型
        """
        path = Path(file_path)
        extension = path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支持的文件类型: {extension}")

        doc_type = self.SUPPORTED_EXTENSIONS[extension]

        if doc_type == "text":
            documents = self._load_text(file_path)
        elif doc_type == "markdown":
            documents = self._load_text(file_path)
        elif doc_type == "pdf":
            documents = self._load_pdf(file_path)
        elif doc_type == "word":
            documents = self._load_word(file_path)
        else:
            raise ValueError(f"未知的文档类型: {doc_type}")

        if clean:
            documents = text_cleaner_service.clean_documents(documents)
            logger.info(f"文档清洗完成: {file_path}")

        return documents

    def _load_text(self, file_path: str) -> List[Document]:
        """加载文本文件"""
        path = Path(file_path)
        content = path.read_text(encoding="utf-8")

        return [Document(
            page_content=content,
            metadata={
                "_source": path.as_posix(),
                "_extension": path.suffix,
                "_file_name": path.name,
            }
        )]

    def _load_pdf(self, file_path: str) -> List[Document]:
        """加载 PDF 文件"""
        if not self._pdf_loader_available:
            raise RuntimeError("PDF 加载器不可用，请安装 pypdf: pip install pypdf")

        from langchain_community.document_loaders import PyPDFLoader

        path = Path(file_path)
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        for doc in documents:
            doc.metadata["_source"] = path.as_posix()
            doc.metadata["_extension"] = ".pdf"
            doc.metadata["_file_name"] = path.name

        logger.info(f"PDF 加载完成: {file_path}, 共 {len(documents)} 页")
        return documents

    def _load_word(self, file_path: str) -> List[Document]:
        """加载 Word 文件"""
        if not self._word_loader_available:
            raise RuntimeError("Word 加载器不可用，请安装 docx2txt: pip install docx2txt")

        from langchain_community.document_loaders import Docx2txtLoader

        path = Path(file_path)
        loader = Docx2txtLoader(file_path)
        documents = loader.load()

        for doc in documents:
            doc.metadata["_source"] = path.as_posix()
            doc.metadata["_extension"] = path.suffix
            doc.metadata["_file_name"] = path.name

        logger.info(f"Word 加载完成: {file_path}, 共 {len(documents)} 个文档")
        return documents

    def is_supported(self, file_path: str) -> bool:
        """
        检查文件是否支持

        Args:
            file_path: 文件路径

        Returns:
            bool: 是否支持
        """
        extension = Path(file_path).suffix.lower()
        return extension in self.get_supported_extensions()


document_loader_service = DocumentLoaderService()
