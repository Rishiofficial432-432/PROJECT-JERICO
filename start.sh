#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  PROJECT JERICO — One-shot launcher
#  Starts the FastAPI backend + HTML frontend static server
# ─────────────────────────────────────────────────────────────

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$ROOT/.venv/bin/python"

# Load local environment variables (e.g., GEMINI_API_KEY) if present.
if [ -f "$ROOT/.env" ]; then
    set -a
    . "$ROOT/.env"
    set +a
fi

# Verify venv exists
if [ ! -x "$PYTHON" ]; then
    echo "❌  .venv not found. Run first:"
    echo "    python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

# ── Function to find available port ────────────────────────────
find_free_port() {
    local port=$1
    while true; do
        if ! lsof -i :$port >/dev/null 2>&1; then
            echo "$port"
            return 0
        fi
        port=$((port + 1))
        if [ "$port" -gt 8050 ]; then
            echo "8000" # fallback
            return 1
        fi
    done
}

API_PORT=$(find_free_port 8000)

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   PROJECT JERICO — MULTI-INSTANCE LCH    ║"
echo "╚══════════════════════════════════════════╝"
echo "📡  Selecting Port: $API_PORT"
echo ""

# ── 1. Unified Jerico Server (FastAPI + HTML) ─────────────────
echo "🚀  Starting Unified Jerico Server  →  http://localhost:$API_PORT"
"$PYTHON" -m uvicorn frontend.api:app \
    --host 0.0.0.0 \
    --port "$API_PORT" \
    --reload \
    --app-dir "$ROOT" \
    2>&1 | sed 's/^/[SERVER] /' &
SERVER_PID=$!

echo ""
echo "────────────────────────────────────────────"
echo "  Main Dashboard  →  http://localhost:$API_PORT/index.html"
echo "  Upload Page     →  http://localhost:$API_PORT/upload.html"
echo "  API Docs        →  http://localhost:$API_PORT/docs"
echo "────────────────────────────────────────────"
echo "  Press  Ctrl+C  to stop."
echo ""

# ── Open browser (macOS) ──────────────────────────────────────
sleep 3
open "http://localhost:$API_PORT/index.html" 2>/dev/null || true

# ── Graceful shutdown ─────────────────────────────────────────
cleanup() {
    echo ""
    echo "🛑  Shutting down server..."
    kill "$SERVER_PID" 2>/dev/null
    
    # Wait for up to 3 seconds for graceful shutdown
    for i in {1..3}; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "✅  Server stopped gracefully."
            exit 0
        fi
        sleep 1
    done

    # Force kill if still running
    kill -9 "$SERVER_PID" 2>/dev/null
    echo "✅  Server stopped."
    exit 0
}

trap cleanup INT TERM
wait
