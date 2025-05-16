#!/bin/bash

set -euo pipefail


LLM_MODEL="tinyllama"
EMBEDDING_MODEL="nomic-embed-text"

echo "[INFO] Starting Ollama in background..."
ollama serve &
OLLAMA_PID=$!

echo "[INFO] Waiting for Ollama API to be ready..."
until curl -s -o /dev/null -w "%{http_code}" http://localhost:11434 | grep -q "200"; do
  sleep 1
done

echo "[INFO] Checking if LLM model '$LLM_MODEL' is available..."
if ! ollama list | grep -q "$LLM_MODEL"; then
  echo "[INFO] LLM model '$LLM_MODEL' not found. Pulling..."
  ollama pull "$LLM_MODEL"
else
  echo "[INFO] LLM model '$LLM_MODEL' already present."
fi

echo "[INFO] Checking if Embedding model '$EMBEDDING_MODEL' is available..."
if ! ollama list | grep -q "$EMBEDDING_MODEL"; then
  echo "[INFO] Embedding model '$EMBEDDING_MODEL' not found. Pulling..."
  ollama pull "$EMBEDDING_MODEL"
else
  echo "[INFO] Embedding model '$EMBEDDING_MODEL' already present."
fi

echo "[INFO] Ollama is ready. LLM model '$LLM_MODEL' and Embedding model '$EMBEDDING_MODEL' are loaded."

# Keep container alive
wait $OLLAMA_PID
