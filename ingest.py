from dotenv import load_dotenv

from hr_assistant.rag import build_vector_db


load_dotenv()

if __name__ == "__main__":
    print("Building HR knowledge vector database...")
    build_vector_db()
    print("Done. Chroma database created in ./.chroma")
