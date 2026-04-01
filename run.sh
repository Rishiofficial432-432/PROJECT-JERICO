#!/bin/bash
# One-click startup script for CCTV Security System
echo "Starting CCTV Security System..."
streamlit run src/dashboard.py &
# Uncomment below when ingest, detect, and threat_logic are implemented
# python src/ingest.py | python src/detect.py | python src/threat_logic.py
echo "Dashboard running on http://localhost:8501"
wait
