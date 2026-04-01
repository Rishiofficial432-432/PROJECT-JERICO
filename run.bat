@echo off
setlocal

echo Starting CCTV Security System...

if not exist .venv\Scripts\python.exe (
	echo Error: .venv not found. Create it with: python -m venv .venv
	exit /b 1
)

set PYTHON_BIN=.venv\Scripts\python.exe

echo Using Python: %PYTHON_BIN%

start /B %PYTHON_BIN% -m streamlit run src\dashboard.py --server.port=8501
echo Dashboard running on http://localhost:8501
