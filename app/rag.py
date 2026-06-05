from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from app.config import Settings
from app.document_loader import chunk_text, load_document
from app.embeddings import EmbeddingModel
from app.llm import ChatModel
from app.vector_store import JsonVectorStore, VectorRecord


class RAGPipeline:
    def __init__(
        self,
        settings: Settings,
        embeddings: EmbeddingModel,
        vector_store: JsonVectorStore,
        chat_model: ChatModel,
    ) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.chat_model = chat_model

    def ingest(self, path: Path) -> int:
        text = load_document(path)
        chunks = chunk_text(text, self.settings.chunk_size, self.settings.chunk_overlap)
        vectors = self.embeddings.embed(chunks)
        records = [
            VectorRecord(
                id=str(uuid4()),
                text=chunk,
                source=path.name,
                embedding=embedding,
            )
            for chunk, embedding in zip(chunks, vectors)
        ]
        self.vector_store.add(records)
        return len(records)

    def ask(self, question: str) -> dict:
        query_embedding = self.embeddings.embed([question])[0]
        contexts = self.vector_store.search(query_embedding, self.settings.top_k)
        answer = self.chat_model.answer(question, contexts)
        return {
            "answer": answer,
            "sources": [{"source": item.source, "text": item.text} for item in contexts],
        }
