from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.config import Settings, get_settings
from app.embeddings import build_embedding_model
from app.llm import build_chat_model
from app.rag import RAGPipeline
from app.vector_store import JsonVectorStore


class QuestionRequest(BaseModel):
    question: str


def build_pipeline(settings: Settings = Depends(get_settings)) -> RAGPipeline:
    return RAGPipeline(
        settings=settings,
        embeddings=build_embedding_model(settings),
        vector_store=JsonVectorStore(settings.index_path),
        chat_model=build_chat_model(settings),
    )


app = FastAPI(title="RAG Knowledge Assistant")


@app.get("/health")
def health(settings: Settings = Depends(get_settings)) -> dict:
    store = JsonVectorStore(settings.index_path)
    return {"status": "ok", "indexed_chunks": store.count()}


@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    pipeline: RAGPipeline = Depends(build_pipeline),
) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".docx", ".txt", ".md"}:
        raise HTTPException(status_code=400, detail="Upload a PDF, DOCX, TXT, or MD file.")

    target = settings.uploads_dir / Path(file.filename or f"upload{suffix}").name
    target.write_bytes(await file.read())
    chunks = pipeline.ingest(target)
    return {"filename": target.name, "chunks_indexed": chunks}


@app.post("/ask")
def ask(request: QuestionRequest, pipeline: RAGPipeline = Depends(build_pipeline)) -> dict:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    return pipeline.ask(request.question)


@app.get("/", response_class=HTMLResponse)
def web_ui() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RAG Knowledge Assistant</title>
  <style>
    :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
    body { margin: 0; background: #f5f7fb; color: #172033; }
    main { max-width: 980px; margin: 0 auto; padding: 32px 20px; }
    h1 { font-size: 32px; margin: 0 0 8px; }
    p { color: #586174; }
    section { background: white; border: 1px solid #dfe5ef; border-radius: 8px; padding: 18px; margin-top: 18px; }
    input, textarea, button { font: inherit; }
    textarea { width: 100%; min-height: 110px; box-sizing: border-box; padding: 12px; border: 1px solid #cbd5e1; border-radius: 6px; }
    button { background: #116466; color: white; border: 0; border-radius: 6px; padding: 10px 14px; cursor: pointer; }
    button:disabled { opacity: .6; cursor: wait; }
    pre { white-space: pre-wrap; background: #111827; color: #f9fafb; padding: 14px; border-radius: 6px; }
  </style>
</head>
<body>
  <main>
    <h1>RAG Knowledge Assistant</h1>
    <p>Upload documents, index them, and ask questions grounded in your knowledge base.</p>
    <section>
      <h2>Ingest Document</h2>
      <input id="file" type="file" accept=".pdf,.docx,.txt,.md" />
      <button onclick="ingest()">Upload & Index</button>
      <pre id="ingestResult"></pre>
    </section>
    <section>
      <h2>Ask</h2>
      <textarea id="question" placeholder="Ask a question about your indexed documents"></textarea>
      <button onclick="ask()">Ask</button>
      <pre id="answer"></pre>
    </section>
  </main>
  <script>
    async function ingest() {
      const file = document.getElementById('file').files[0];
      if (!file) return;
      const body = new FormData();
      body.append('file', file);
      const res = await fetch('/ingest', { method: 'POST', body });
      document.getElementById('ingestResult').textContent = JSON.stringify(await res.json(), null, 2);
    }
    async function ask() {
      const question = document.getElementById('question').value;
      const res = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
      });
      document.getElementById('answer').textContent = JSON.stringify(await res.json(), null, 2);
    }
  </script>
</body>
</html>
"""
