

   
import os
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


BASE_DIR = Path(__file__).resolve().parent.parent

KNOWLEDGE_DIR = (
    BASE_DIR
    / "data"
    / "knowledge_base"
)


def get_embeddings():

    model_name = os.getenv(
        "OLLAMA_EMBEDDING_MODEL",
        "nomic-embed-text"
    )

    ollama_url = os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434"
    )

    return OllamaEmbeddings(
        model=model_name,
        base_url=ollama_url,
    )


@lru_cache(maxsize=1)
def get_vector_db():

    documents = []

    for file_path in KNOWLEDGE_DIR.glob("*.txt"):

        text = file_path.read_text(
            encoding="utf-8"
        )

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": file_path.name
                }
            )
        )

    if not documents:
        raise ValueError(
            "No knowledge-base documents found."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = splitter.split_documents(
        documents
    )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
    )

    return db


def build_vector_db():

    return get_vector_db()


def search_knowledge(query, k=4):

    db = get_vector_db()

    return db.similarity_search(
        query,
        k=k
    )
