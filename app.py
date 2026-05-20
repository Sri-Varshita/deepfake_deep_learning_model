"""
DeepShield AI - Deepfake Detection System
Entry Point for Streamlit Application

This file serves as the entry point for Streamlit deployment.
All application logic is in the src/ directory.

Developed by: Emin Cem Koyluoglu
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run the main application
from app import main

if __name__ == "__main__":
    main()
