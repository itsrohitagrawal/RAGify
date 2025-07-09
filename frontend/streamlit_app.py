import streamlit as st
import requests
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any
import time
import logging

# Configure logging to reduce noise
logging.getLogger("streamlit").setLevel(logging.ERROR)

# Configure Streamlit page
st.set_page_config(
    page_title="RAG ChatBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend API URL
API_BASE_URL = "http://localhost:8000"
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'uploaded_documents' not in st.session_state:
    st.session_state.uploaded_documents = []

if 'backend_available' not in st.session_state:
    st.session_state.backend_available = True
# Initialize session state with proper error handling
def initialize_session():
    """Initialize session state variables"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'uploaded_documents' not in st.session_state:
        st.session_state.uploaded_documents = []
    
    if 'backend_available' not in st.session_state:
        st.session_state.backend_available = True

# Check if backend is available
def check_backend():
    """Check if backend is available"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        st.session_state.backend_available = response.status_code == 200
        return st.session_state.backend_available
    except Exception:
        st.session_state.backend_available = False
        return False

def load_chat_history():
    """Load chat history from backend"""
    try:
        response = requests.get(f"{API_BASE_URL}/chat-history/{st.session_state.session_id}")
        if response.status_code == 200:
            messages = response.json()
            st.session_state.messages = []
            for msg in messages:
                st.session_state.messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": msg["timestamp"]
                })
    except Exception as e:
        st.error(f"Error loading chat history: {str(e)}")

def upload_document(file):
    """Upload document to backend"""
    try:
        files = {"file": (file.name, file, file.type)}
        response = requests.post(f"{API_BASE_URL}/upload-document", files=files)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error uploading document: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error uploading document: {str(e)}")
        return None

def send_message(message: str):
    """Send message to backend and get response"""
    try:
        payload = {
            "message": message,
            "session_id": st.session_state.session_id
        }
        
        response = requests.post(f"{API_BASE_URL}/chat", json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error sending message: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error sending message: {str(e)}")
        return None

def get_documents():
    """Get list of uploaded documents"""
    try:
        response = requests.get(f"{API_BASE_URL}/documents")
        if response.status_code == 200:
            return response.json()
        else:
            return []
    except Exception as e:
        st.error(f"Error getting documents: {str(e)}")
        return []

def delete_document(doc_id: str):
    """Delete a document"""
    try:
        response = requests.delete(f"{API_BASE_URL}/documents/{doc_id}")
        if response.status_code == 200:
            return True
        else:
            st.error(f"Error deleting document: {response.text}")
            return False
    except Exception as e:
        st.error(f"Error deleting document: {str(e)}")
        return False

def clear_chat_history():
    """Clear chat history"""
    try:
        response = requests.delete(f"{API_BASE_URL}/chat-history/{st.session_state.session_id}")
        if response.status_code == 200:
            st.session_state.messages = []
            st.success("Chat history cleared!")
        else:
            st.error(f"Error clearing chat history: {response.text}")
    except Exception as e:
        st.error(f"Error clearing chat history: {str(e)}")

# Main app layout
def main():
    """Main application function"""
    # Initialize session
    initialize_session()
    
    # Check backend availability
    if not check_backend():
        st.error("🔴 Backend server is not available. Please make sure the FastAPI server is running on port 8000.")
        st.info("Run: `python start_app.py` or `uvicorn backend.main:app --port 8000`")
        return
    
    st.title("🤖 RAG ChatBot")
    st.markdown("Upload documents and chat with your AI assistant!")
    
    # Sidebar for document management
    render_sidebar()
    
    # Main chat interface
    render_chat_interface()
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit, FastAPI, and Grok AI")

def render_sidebar():
    """Render the sidebar with document management"""
    with st.sidebar:
        st.header("📁 Document Management")
        
        # File upload section
        render_file_upload()
        
        # Document list section
        render_document_list()
        
        # Chat controls section
        render_chat_controls()
        
        # Session info section
        # render_session_info()

def render_file_upload():
    """Render file upload section"""
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'txt', 'docx'],
        help="Supported formats: PDF, TXT, DOCX"
    )
    
    if uploaded_file is not None:
        if st.button("Upload Document"):
            with st.spinner("Uploading document..."):
                result = upload_document(uploaded_file)
                if result:
                    st.success(f"Document '{result['filename']}' uploaded successfully!")
                    st.info("Document is being processed in the background.")
                    time.sleep(1)
                    st.rerun()

def render_document_list():
    """Render document list section"""
    st.subheader("Uploaded Documents")
    
    try:
        documents = get_documents()
        
        if documents:
            for doc in documents:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    status = "✅ Processed" if doc["processed"] else "⏳ Processing"
                    st.write(f"**{doc['filename']}**")
                    st.write(f"Size: {doc['file_size']} bytes")
                    st.write(f"Status: {status}")
                    if doc.get("chunk_count"):
                        st.write(f"Chunks: {doc['chunk_count']}")
                
                with col2:
                    if st.button("🗑️", key=f"delete_{doc['id']}", help="Delete document"):
                        if delete_document(doc["id"]):
                            st.success("Document deleted!")
                            st.rerun()
                
                st.divider()
            
            # Show refresh button if any documents are processing
            if any(not doc["processed"] for doc in documents):
                if st.button("🔄 Refresh Status", key="refresh_status"):
                    st.rerun()
                st.info("Some documents are still processing.")
        else:
            st.info("No documents uploaded yet.")
    
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")

def render_chat_controls():
    """Render chat controls section"""
    st.subheader("Chat Controls")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Load History"):
            load_chat_history()
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear History"):
            clear_chat_history()
            st.rerun()

# def render_session_info():
#     """Render session info section"""
#     st.subheader("Session Info")
#     st.write(f"Session ID: `{st.session_state.session_id[:8]}...`")
#     st.write(f"Messages: {len(st.session_state.messages)}")
    
#     # Backend status
#     status_color = "🟢" if st.session_state.backend_available else "🔴"
#     st.write(f"Backend: {status_color}")

def render_chat_interface():
    """Render the main chat interface"""
    st.header("💬 Chat")
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and message.get("sources"):
                    if message["sources"]:
                        st.markdown("**Sources:**")
                        for source in message["sources"]:
                            st.markdown(f"- {source}")
    
    # Chat input
    handle_chat_input()

def handle_chat_input():
    """Handle chat input and responses"""
    if prompt := st.chat_input("Ask me anything about your documents..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from backend
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = send_message(prompt)
                
                if response:
                    st.markdown(response["message"])
                    
                    # Show sources if available
                    if response.get("sources"):
                        st.markdown("**Sources:**")
                        for source in response["sources"]:
                            st.markdown(f"- {source}")
                    
                    # Show response metadata
                    if response.get("response_time"):
                        st.caption(f"Response time: {response['response_time']:.2f}s")
                    
                    # Add assistant response to chat
                    assistant_message = {
                        "role": "assistant",
                        "content": response["message"],
                        "sources": response.get("sources", []),
                        "timestamp": datetime.now().isoformat()
                    }
                    st.session_state.messages.append(assistant_message)
                else:
                    error_message = "Sorry, I couldn't process your request. Please try again."
                    st.error(error_message)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message,
                        "timestamp": datetime.now().isoformat()
                    })

# Main app layout
st.title("🤖 RAG ChatBot")
st.markdown("Upload documents and chat with your AI assistant!")

# Sidebar for document management
with st.sidebar:
    st.header("📁 Document Management")
    
    # File upload
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'txt', 'docx'],
        help="Supported formats: PDF, TXT, DOCX"
    )
    
    if uploaded_file is not None:
        if st.button("Upload Document"):
            with st.spinner("Uploading document..."):
                result = upload_document(uploaded_file)
                if result:
                    st.success(f"Document '{result['filename']}' uploaded successfully!")
                    st.info("Document is being processed in the background.")
                    time.sleep(1)
                    st.rerun()
    
    # Document list
    st.subheader("Uploaded Documents")
    documents = get_documents()
    
    if documents:
        for doc in documents:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                status = "✅ Processed" if doc["processed"] else "⏳ Processing"
                st.write(f"**{doc['filename']}**")
                st.write(f"Size: {doc['file_size']} bytes")
                st.write(f"Status: {status}")
                if doc.get("chunk_count"):
                    st.write(f"Chunks: {doc['chunk_count']}")
            
            with col2:
                if st.button("🗑️", key=f"delete_{doc['id']}", help="Delete document"):
                    if delete_document(doc["id"]):
                        st.success("Document deleted!")
                        st.rerun()
            
            st.divider()
    else:
        st.info("No documents uploaded yet.")
    
    # Chat controls
    st.subheader("Chat Controls")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Load History"):
            load_chat_history()
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear History"):
            clear_chat_history()
            st.rerun()
    
    # Session info
    st.subheader("Session Info")
    st.write(f"Session ID: `{st.session_state.session_id[:8]}...`")
    st.write(f"Messages: {len(st.session_state.messages)}")

# Main chat interface
st.header("💬 Chat")

# Display chat messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                if message["sources"]:
                    st.markdown("**Sources:**")
                    for source in message["sources"]:
                        st.markdown(f"- {source}")

# Chat input
if prompt := st.chat_input("Ask me anything about your documents..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response from backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = send_message(prompt)
            
            if response:
                st.markdown(response["message"])
                
                # Show sources if available
                if response.get("sources"):
                    st.markdown("**Sources:**")
                    for source in response["sources"]:
                        st.markdown(f"- {source}")
                
                # Show response metadata
                if response.get("response_time"):
                    st.caption(f"Response time: {response['response_time']:.2f}s")
                
                # Add assistant response to chat
                assistant_message = {
                    "role": "assistant",
                    "content": response["message"],
                    "sources": response.get("sources", []),
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.messages.append(assistant_message)
            else:
                error_message = "Sorry, I couldn't process your request. Please try again."
                st.error(error_message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "timestamp": datetime.now().isoformat()
                })

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, FastAPI, and Grok AI")

# Auto-refresh for processing status (only if in Streamlit context)
try:
    if documents and any(not doc["processed"] for doc in documents):
        # Use a more controlled refresh mechanism
        if st.button("🔄 Refresh Status", key="refresh_status"):
            st.rerun()
        st.info("Some documents are still processing. Click 'Refresh Status' to update.")
except Exception:
    pass