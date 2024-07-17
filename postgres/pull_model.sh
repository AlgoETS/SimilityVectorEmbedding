#!/bin/sh
apt-get update && apt-get install -y curl
/bin/ollama serve
# curl -X POST http://localhost:11434/api/pull -d '{"name": "llama3"}'
