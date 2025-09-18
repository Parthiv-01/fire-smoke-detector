# app.py
import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("🔧 Installing required packages...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "--index-url", "https://download.pytorch.org/whl/cu118",
        "torch", "torchvision", "torchaudio"
    ])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✅ Packages installed successfully!")

def run_streamlit():
    """Run the Streamlit app"""
    print("🚀 Starting Fire Detection Streamlit App...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_fire_detection.py"])

if __name__ == "__main__":
    # Check if requirements are installed
    try:
        import torch
        import av
        import streamlit
        from transformers import AutoProcessor
        print("✅ All packages available")
    except ImportError:
        install_requirements()
    
    # Run the app
    run_streamlit()
