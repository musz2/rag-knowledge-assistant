#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/rag-knowledge-assistant}"
REPO_URL="${REPO_URL:?Set REPO_URL to your GitHub repository URL}"

sudo apt-get update
sudo apt-get install -y ca-certificates curl git

if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER"
fi

sudo mkdir -p "$APP_DIR"
sudo chown "$USER:$USER" "$APP_DIR"

if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" pull --ff-only
else
  git clone "$REPO_URL" "$APP_DIR"
fi

cd "$APP_DIR"
cp -n .env.example .env
docker compose up -d --build
