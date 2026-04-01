#!/bin/bash
set -e

echo "Starting CCTV Security System..."

# Single source of truth: project-local .venv only.
if [ ! -x ".venv/bin/python" ]; then
	echo "Error: .venv not found. Create it with: python3.12 -m venv .venv"
	exit 1
fi

PYTHON_BIN=".venv/bin/python"

echo "Using Python: $PYTHON_BIN"

# Validate core imports. Install requirements once if any are missing.
MISSING="$($PYTHON_BIN - <<'PY'
import importlib.util
required = ["streamlit", "cv2", "torch", "scipy", "ultralytics", "transformers"]
missing = [m for m in required if importlib.util.find_spec(m) is None]
print(" ".join(missing))
PY
)"

if [ -n "$MISSING" ]; then
	echo "Missing modules: $MISSING"
	echo "Installing dependencies from requirements.txt..."
	"$PYTHON_BIN" -m pip install -r requirements.txt
fi

echo "Launching dashboard on http://localhost:8501"
"$PYTHON_BIN" -m streamlit run src/dashboard.py --server.port=8501
