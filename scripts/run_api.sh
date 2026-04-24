#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

echo "Starting API at http://${HOST}:${PORT}"
python -m uvicorn src.api.main:app --host "$HOST" --port "$PORT" --reload
