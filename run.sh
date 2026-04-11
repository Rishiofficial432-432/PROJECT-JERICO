#!/bin/bash
set -e

# Load local environment variables (e.g., GEMINI_API_KEY) if present.
if [ -f ".env" ]; then
    set -a
    . ".env"
    set +a
fi

echo "Starting CCTV Security System..."

# Priority: 1. System/Global python3 (often modern), 2. Local .venv
# Check if global python3 supports Torch 2.4+ (needed for Florence-2)
TORCH_OK=$(python3 -c "import torch; from packaging import version; print(version.parse(torch.__version__) >= version.parse('2.4'))" 2>/dev/null || echo "False")

if [ "$TORCH_OK" = "True" ]; then
    PYTHON_BIN="python3"
elif [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
else
    echo "Error: No suitable Python environment found. Core models require Torch >= 2.4."
    exit 1
fi

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
"$PYTHON_BIN" -m streamlit run dashboard.py --server.port=8501
