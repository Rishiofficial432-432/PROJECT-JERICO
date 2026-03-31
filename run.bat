@echo off
echo Starting CCTV Security System...
call venv\Scripts\activate
start /B streamlit run src\dashboard.py
echo Dashboard running on http://localhost:8501
