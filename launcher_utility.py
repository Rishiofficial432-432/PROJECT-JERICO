import subprocess
import os
import sys

# DO NOT import streamlit in this file!
# Naming this file 'streamlit.py' can break Python's import system if we do.
# To bypass that, we directly execute the Streamlit binary via a system call.

if __name__ == "__main__":
    print("🚀 Auto-Launch: CCTV Security System Dashboard")
    # Launch the actual Streamlit executable directly from the virtual environment
    streamlit_exe = os.path.join("venv", "Scripts", "streamlit.exe")
    dashboard_path = os.path.join("src", "dashboard.py")
    
    if not os.path.exists(streamlit_exe):
        print(f"Error: Could not find Streamlit at {streamlit_exe}. Did you create the venv?")
        sys.exit(1)
        
    # Launch Streamlit and pass any extra arguments you provided to it
    subprocess.run([streamlit_exe, "run", dashboard_path] + sys.argv[1:])
