# RAG Knowledge Assistant

An intermediate production-style RAG project that ingests PDFs, DOCX, TXT, and Markdown files, indexes document chunks with embeddings, retrieves relevant context, and answers through OpenAI, a Llama/Ollama endpoint, or a local deterministic demo mode.

```text
PDFs / Docs
      ↓
Embedding Model
      ↓
Vector Store
      ↓
RAG Pipeline
      ↓
OpenAI / Llama / Local
      ↓
Web UI
```

## Features

- FastAPI web UI and JSON API
- Document ingestion for PDF, DOCX, TXT, and Markdown
- Persistent JSON vector store for local development and CI
- OpenAI embeddings and chat support
- Llama/Ollama chat support
- Docker and Docker Compose
- GitHub Actions CI for tests and Docker image builds
- AWS EC2 bootstrap script and deployment guide

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Open `http://localhost:8000`, upload a document, and ask a question.

## Docker

```bash
cp .env.example .env
docker compose up --build
```

The app will be available at `http://localhost:8000`.

## API

Health check:

```bash
curl http://localhost:8000/health
```

Ingest a document:

```bash
curl -F "file=@/path/to/document.pdf" http://localhost:8000/ingest
```

Ask a question:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What does the document say about RAG architecture?"}'
```

## Provider Modes

Local demo mode works without external services:

```env
EMBEDDING_PROVIDER=local
LLM_PROVIDER=local
```

OpenAI mode:

```env
EMBEDDING_PROVIDER=openai
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

Llama/Ollama chat mode:

```env
EMBEDDING_PROVIDER=local
LLM_PROVIDER=llama
LLAMA_BASE_URL=http://localhost:11434
LLAMA_MODEL=llama3.1
```

## CI/CD

GitHub Actions runs on pushes and pull requests to `main`:

- Install Python dependencies
- Run tests
- Build the Docker image

See `.github/workflows/ci.yml`.

## AWS EC2 Deployment

See [docs/aws-ec2-deployment.md](docs/aws-ec2-deployment.md) for the EC2 setup flow.

## What You Learn

- Production deployment
- Containers
- CI/CD
- RAG architecture
