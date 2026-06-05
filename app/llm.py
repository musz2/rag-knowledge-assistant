from __future__ import annotations

from abc import ABC, abstractmethod

from app.config import Settings
from app.vector_store import VectorRecord


class ChatModel(ABC):
    @abstractmethod
    def answer(self, question: str, contexts: list[VectorRecord]) -> str:
        raise NotImplementedError


class LocalExtractiveChat(ChatModel):
    def answer(self, question: str, contexts: list[VectorRecord]) -> str:
        if not contexts:
            return "I do not have enough indexed knowledge to answer that yet."
        citations = ", ".join(sorted({context.source for context in contexts}))
        evidence = "\n\n".join(context.text for context in contexts)
        return (
            f"Based on the indexed documents ({citations}), the most relevant context is:\n\n"
            f"{evidence[:1600]}"
        )


class OpenAIChat(ChatModel):
    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI

        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_chat_model

    def answer(self, question: str, contexts: list[VectorRecord]) -> str:
        context_text = "\n\n".join(
            f"Source: {context.source}\n{context.text}" for context in contexts
        )
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": "Answer only from the provided context. Cite source filenames.",
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context_text}\n\nQuestion: {question}",
                },
            ],
        )
        return response.output_text


class LlamaChat(ChatModel):
    def __init__(self, settings: Settings) -> None:
        import requests

        self.requests = requests
        self.base_url = settings.llama_base_url.rstrip("/")
        self.model = settings.llama_model

    def answer(self, question: str, contexts: list[VectorRecord]) -> str:
        context_text = "\n\n".join(
            f"Source: {context.source}\n{context.text}" for context in contexts
        )
        prompt = (
            "Answer only from the provided context and cite source filenames.\n\n"
            f"Context:\n{context_text}\n\nQuestion: {question}"
        )
        response = self.requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=60,
        )
        response.raise_for_status()
        return response.json().get("response", "")


def build_chat_model(settings: Settings) -> ChatModel:
    provider = settings.llm_provider.lower()
    if provider == "openai":
        return OpenAIChat(settings)
    if provider == "llama":
        return LlamaChat(settings)
    return LocalExtractiveChat()
