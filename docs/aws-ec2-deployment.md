# AWS EC2 Deployment

## 1. Create the instance

Create an Ubuntu LTS EC2 instance, then open inbound TCP ports:

- `22` for SSH from your IP
- `8000` for the web app, or `80`/`443` if you later add a reverse proxy

Use at least `t3.small` for small demos. Choose a larger instance if you run a local Llama model on the same host.

## 2. Bootstrap Docker and the app

SSH into the instance and run:

```bash
export REPO_URL=https://github.com/YOUR_USER/YOUR_REPO.git
curl -fsSL https://raw.githubusercontent.com/YOUR_USER/YOUR_REPO/main/scripts/ec2_bootstrap.sh | bash
```

Then edit `/opt/rag-knowledge-assistant/.env`:

```bash
cd /opt/rag-knowledge-assistant
nano .env
docker compose up -d --build
```

## 3. Production settings

For OpenAI-backed RAG:

```bash
EMBEDDING_PROVIDER=openai
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

For local deterministic demo mode, leave both providers as `local`.

## 4. GitHub Actions deployment option

The included CI workflow tests and builds the Docker image. To deploy automatically, add these repository secrets:

- `EC2_HOST`
- `EC2_USER`
- `EC2_SSH_KEY`

Then extend `.github/workflows/ci.yml` with an SSH deploy step after the Docker build.
