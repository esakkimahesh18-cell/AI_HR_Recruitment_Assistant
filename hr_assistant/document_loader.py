from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)


def load_resume(path: str):
    """Load a PDF, DOCX, or TXT resume into LangChain documents."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return PyPDFLoader(str(file_path)).load()

    if suffix == ".docx":
        return Docx2txtLoader(str(file_path)).load()

    if suffix == ".txt":
        return TextLoader(str(file_path), encoding="utf-8").load()

    raise ValueError("Supported resume formats: PDF, DOCX, TXT")


def documents_to_text(documents: List) -> str:
    return "\n\n".join(doc.page_content for doc in documents)
