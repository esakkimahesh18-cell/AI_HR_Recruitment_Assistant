from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = BASE_DIR / "data" / "knowledge_base"
CHROMA_DIR = BASE_DIR / ".chroma"


def get_embeddings():
    return OllamaEmbeddings(
        model="nomic-embed-text"
    )


def build_vector_db():
    documents = []

    for file_path in KNOWLEDGE_DIR.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        documents.append(
            Document(
                page_content=text,
                metadata={"source": file_path.name}
            )
        )

    if not documents:
        raise ValueError("No knowledge-base documents found.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    db = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=str(CHROMA_DIR)
    )

    return db


def get_vector_db():
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings()
    )


def search_knowledge(query, k=4):
    db = get_vector_db()
    results = db.similarity_search(query, k=k)

    return results