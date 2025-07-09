#!/usr/bin/env python3
"""
Startup script for RAG ChatBot
This script starts both the FastAPI backend and Streamlit frontend
"""

import subprocess
import sys
import os
import time
import signal
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import streamlit
        import sentence_transformers
        import chromadb
        import openai
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def setup_environment():
    """Setup environment variables and directories"""
    # Check for Grok API key
    if not os.getenv("GROK_API_KEY"):
        print("⚠️  Warning: GROK_API_KEY environment variable not set")
        print("Please set it with: set GROK_API_KEY=")
        
        # Ask user if they want to continue with placeholder
        response = input("Continue with placeholder API key? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Create necessary directories
    directories = ["data", "data/documents", "data/chat_history", "data/embeddings"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Environment setup complete")

def start_backend():
    """Start the FastAPI backend server"""
    print("🚀 Starting FastAPI backend...")
    backend_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", 
        "backend.main:app", 
        "--host", "0.0.0.0", 
        "--port", "8000",
        "--reload"
    ])
    return backend_process

def start_frontend():
    """Start the Streamlit frontend"""
    print("🚀 Starting Streamlit frontend...")
    frontend_process = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", 
        "frontend/streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])
    return frontend_process

def main():
    """Main startup function"""
    print("🤖 RAG ChatBot Startup Script")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    # Start backend
    backend_process = start_backend()
    
    # Wait for backend to start
    print("⏳ Waiting for backend to start...")
    time.sleep(5)
    
    # Start frontend
    frontend_process = start_frontend()
    
    print("\n" + "=" * 40)
    print("🎉 RAG ChatBot is running!")
    print("📱 Frontend (Streamlit): http://localhost:8501")
    print("🔧 Backend (FastAPI): http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("=" * 40)
    print("Press Ctrl+C to stop both servers")
    
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print("\n🛑 Shutting down servers...")
        backend_process.terminate()
        frontend_process.terminate()
        
        # Wait for processes to terminate
        backend_process.wait()
        frontend_process.wait()
        
        print("✅ Servers stopped successfully")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    main()