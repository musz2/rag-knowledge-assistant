from pathlib import Path

from app.config import Settings
from app.embeddings import LocalHashEmbedding
from app.llm import LocalExtractiveChat
from app.rag import RAGPipeline
from app.vector_store import JsonVectorStore


def test_ingest_and_ask(tmp_path: Path) -> None:
    document = tmp_path / "notes.txt"
    document.write_text("FAISS and ChromaDB store embeddings for RAG search.", encoding="utf-8")

    settings = Settings(
        uploads_dir=tmp_path / "uploads",
        index_path=tmp_path / "index.json",
        chunk_size=120,
        chunk_overlap=20,
    )
    pipeline = RAGPipeline(
        settings=settings,
        embeddings=LocalHashEmbedding(),
        vector_store=JsonVectorStore(settings.index_path),
        chat_model=LocalExtractiveChat(),
    )

    chunks = pipeline.ingest(document)
    response = pipeline.ask("What stores embeddings?")

    assert chunks == 1
    assert "FAISS" in response["answer"]
    assert response["sources"][0]["source"] == "notes.txt"
