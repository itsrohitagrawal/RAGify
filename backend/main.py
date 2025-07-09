from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
import uuid
from datetime import datetime
import asyncio
from pathlib import Path

from backend.models.chat_model import ChatMessage, ChatResponse, DocumentInfo
from backend.services.document_service import DocumentService
from backend.services.chat_service import ChatService
from backend.services.embedding_service import EmbeddingService

app = FastAPI(title="RAG ChatBot API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
document_service = DocumentService()
embedding_service = EmbeddingService()
chat_service = ChatService()

# Ensure directories exist
Path("data/documents").mkdir(parents=True, exist_ok=True)
Path("data/chat_history").mkdir(parents=True, exist_ok=True)
Path("data/embeddings").mkdir(parents=True, exist_ok=True)

class ChatRequest(BaseModel):
    message: str
    session_id: str

class DocumentUploadResponse(BaseModel):
    message: str
    document_id: str
    filename: str

@app.post("/upload-document", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Upload and process a document"""
    try:
        # Validate file type
        allowed_types = ['.pdf', '.txt', '.docx']
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"File type {file_ext} not supported. Allowed types: {allowed_types}"
            )
        
        # Generate unique document ID
        doc_id = str(uuid.uuid4())
        
        # Save file
        file_path = await document_service.save_document(file, doc_id)
        
        # Process document in background
        background_tasks.add_task(
            process_document_background, 
            file_path, 
            doc_id, 
            file.filename
        )
        
        return DocumentUploadResponse(
            message="Document uploaded successfully. Processing in background.",
            document_id=doc_id,
            filename=file.filename
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def process_document_background(file_path: str, doc_id: str, filename: str):
    """Background task to process document"""
    try:
        # Extract text from document
        text_chunks = await document_service.extract_text(file_path)
        
        # Create embeddings
        await embedding_service.create_embeddings(text_chunks, doc_id, filename)
        
        print(f"Document {filename} processed successfully")
    except Exception as e:
        print(f"Error processing document {filename}: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the RAG system"""
    try:
        # Get relevant documents
        relevant_docs = await embedding_service.search_similar_documents(
            request.message, 
            top_k=3
        )
        
        # Generate response using Grok
        response = await chat_service.generate_response(
            request.message,
            relevant_docs,
            request.session_id
        )
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat-history/{session_id}", response_model=List[ChatMessage])
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    try:
        history = await chat_service.get_chat_history(session_id)
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chat-history/{session_id}")
async def clear_chat_history(session_id: str):
    """Clear chat history for a session"""
    try:
        await chat_service.clear_chat_history(session_id)
        return {"message": "Chat history cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", response_model=List[DocumentInfo])
async def get_documents():
    """Get list of uploaded documents"""
    try:
        documents = await document_service.get_document_list()
        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and its embeddings"""
    try:
        await document_service.delete_document(doc_id)
        await embedding_service.delete_document_embeddings(doc_id)
        return {"message": "Document deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)