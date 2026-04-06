#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  PROJECT JERICO — One-shot launcher
#  Starts the FastAPI backend + HTML frontend static server
# ─────────────────────────────────────────────────────────────

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$ROOT/.venv/bin/python"

# Verify venv exists
if [ ! -x "$PYTHON" ]; then
    echo "❌  .venv not found. Run first:"
    echo "    python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

API_PORT=8000
FRONTEND_PORT=3000

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║       PROJECT JERICO  —  LAUNCHER        ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── 1. FastAPI backend ────────────────────────────────────────
echo "🚀  Starting FastAPI backend  →  http://localhost:$API_PORT"
"$PYTHON" -m uvicorn frontend.api:app \
    --host 0.0.0.0 \
    --port "$API_PORT" \
    --reload \
    --app-dir "$ROOT" \
    2>&1 | sed 's/^/[API] /' &
API_PID=$!

# ── 2. Static file server for HTML frontend ───────────────────
echo "🌐  Starting HTML frontend    →  http://localhost:$FRONTEND_PORT"
"$PYTHON" -m http.server "$FRONTEND_PORT" \
    --directory "$ROOT/frontend" \
    2>&1 | sed 's/^/[WEB] /' &
WEB_PID=$!

echo ""
echo "────────────────────────────────────────────"
echo "  API  Backend   →  http://localhost:$API_PORT"
echo "  HTML Frontend  →  http://localhost:$FRONTEND_PORT/index.html"
echo "  Upload Page    →  http://localhost:$FRONTEND_PORT/upload.html"
echo "  API Docs       →  http://localhost:$API_PORT/docs"
echo "────────────────────────────────────────────"
echo "  Press  Ctrl+C  to stop everything."
echo ""

# ── Open browser (macOS) ──────────────────────────────────────
sleep 2
open "http://localhost:$FRONTEND_PORT/index.html" 2>/dev/null || true

# ── Graceful shutdown ─────────────────────────────────────────
cleanup() {
    echo ""
    echo "🛑  Shutting down all servers..."
    kill "$API_PID" "$WEB_PID" 2>/dev/null
    
    # Wait for up to 3 seconds for graceful shutdown
    for i in {1..3}; do
        if ! kill -0 "$API_PID" 2>/dev/null && ! kill -0 "$WEB_PID" 2>/dev/null; then
            echo "✅  All servers stopped gracefully."
            exit 0
        fi
        sleep 1
    done

    # Force kill if still running
    echo "⚠️  Force killing stuck processes..."
    kill -9 "$API_PID" "$WEB_PID" 2>/dev/null
    echo "✅  All servers stopped."
    exit 0
}

trap cleanup INT TERM
wait
