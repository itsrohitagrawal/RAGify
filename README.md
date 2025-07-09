# RAG ChatBot with Grok AI

A complete Retrieval-Augmented Generation (RAG) chatbot system that allows users to upload documents and chat with an AI assistant. Built with FastAPI backend, Streamlit frontend, and powered by Grok AI's low-cost model.

## Features
- 📄 Upload PDF, TXT, DOCX documents
- 🤖 Chat with AI using document context
- 💾 Persistent chat history in JSON
- 🔍 Vector-based document search
- 🌐 Clean web interface
- 📊 Document management system

## Tech Stack
- **Backend**: FastAPI, ChromaDB, Sentence Transformers
- **Frontend**: Streamlit
- **AI**: Grok AI (low-cost model)
- **Storage**: Local files + JSON chat history

## Quick Start
```bash
pip install -r requirements.txt
export GROK_API_KEY="your-key"
python start_app.py
```

Access at: http://localhost:8501
