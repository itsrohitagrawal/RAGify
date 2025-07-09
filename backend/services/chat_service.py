import os
import json
import uuid
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from backend.models.chat_model import ChatHistory, ChatMessage, ChatResponse, DocumentInfo, MessageRole
class ChatService:
    def __init__(self):
        # Initialize Grok client
        self.grok_client = OpenAI(
            api_key=os.getenv("GROK_API_KEY", "your-grok-api-key-here"),
            base_url="https://api.x.ai/v1"
        )
        
        # Chat history storage
        self.chat_history_dir = Path("data/chat_history")
        self.chat_history_dir.mkdir(parents=True, exist_ok=True)
    
    async def generate_response(
        self, 
        user_message: str, 
        relevant_docs: List[Dict[str, Any]], 
        session_id: str
    ) -> ChatResponse:
        """Generate response using Grok AI with RAG context"""
        start_time = time.time()
        
        # Build context from relevant documents
        context = self._build_context(relevant_docs)
        
        # Build system prompt
        system_prompt = self._build_system_prompt(context)
        
        # Get conversation history
        history = await self.get_chat_history(session_id)
        
        # Build messages for Grok
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add recent conversation history (last 10 messages)
        for msg in history[-10:]:
            messages.append({
                "role": msg.role.value,
                "content": msg.content
            })
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        try:
            # Call Grok API
            response = self.grok_client.chat.completions.create(
                model="grok-beta",  # Use the lowest/cheapest model
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            assistant_response = response.choices[0].message.content
            
            # Save messages to history
            await self._save_message_to_history(
                session_id, 
                user_message, 
                MessageRole.USER
            )
            await self._save_message_to_history(
                session_id, 
                assistant_response, 
                MessageRole.ASSISTANT
            )
            
            # Extract sources
            sources = [doc["metadata"]["filename"] for doc in relevant_docs]
            sources = list(set(sources))  # Remove duplicates
            
            response_time = time.time() - start_time
            
            return ChatResponse(
                message=assistant_response,
                sources=sources,
                session_id=session_id,
                response_time=response_time,
                metadata={
                    "relevant_docs_count": len(relevant_docs),
                    "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else 0
                }
            )
        
        except Exception as e:
            # Fallback response
            fallback_response = self._generate_fallback_response(user_message, relevant_docs)
            
            await self._save_message_to_history(
                session_id, 
                user_message, 
                MessageRole.USER
            )
            await self._save_message_to_history(
                session_id, 
                fallback_response, 
                MessageRole.ASSISTANT
            )
            
            response_time = time.time() - start_time
            
            return ChatResponse(
                message=fallback_response,
                sources=[doc["metadata"]["filename"] for doc in relevant_docs],
                session_id=session_id,
                response_time=response_time,
                metadata={"error": str(e)}
            )
    
    def _build_context(self, relevant_docs: List[Dict[str, Any]]) -> str:
        """Build context string from relevant documents"""
        if not relevant_docs:
            return "No relevant documents found."
        
        context = "Based on the following document excerpts:\n\n"
        
        for i, doc in enumerate(relevant_docs, 1):
            context += f"Document {i} (from {doc['metadata']['filename']}):\n"
            context += f"{doc['content']}\n\n"
        
        return context
    
    def _build_system_prompt(self, context: str) -> str:
        """Build system prompt for Grok"""
        return f"""You are a helpful AI assistant that answers questions based on provided documents. 

{context}

Instructions:
1. Answer questions based primarily on the provided document excerpts
2. If the documents don't contain relevant information, clearly state that
3. Be concise and accurate in your responses
4. Always cite which document your information comes from when possible
5. If you're unsure about something, say so
6. Maintain a helpful and professional tone

Remember: Your knowledge comes from the documents provided above. Focus on providing accurate information based on these sources."""
    
    def _generate_fallback_response(self, user_message: str, relevant_docs: List[Dict[str, Any]]) -> str:
        """Generate a fallback response when Grok API fails"""
        if not relevant_docs:
            return "I apologize, but I couldn't find relevant information in the uploaded documents to answer your question. Could you please try rephrasing your question or upload more relevant documents?"
        
        # Simple keyword-based response
        response = "Based on the uploaded documents, I found some relevant information:\n\n"
        
        for doc in relevant_docs[:2]:  # Show top 2 relevant documents
            response += f"From {doc['metadata']['filename']}:\n"
            response += f"{doc['content'][:300]}...\n\n"
        
        response += "Note: I'm currently unable to provide a more detailed analysis. Please try your question again."
        
        return response
    
    async def _save_message_to_history(self, session_id: str, content: str, role: MessageRole):
        """Save a message to chat history"""
        history_file = self.chat_history_dir / f"{session_id}.json"
        
        # Create new message
        message = ChatMessage(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=datetime.now(),
            session_id=session_id
        )
        
        # Load existing history or create new
        if history_file.exists():
            with open(history_file, 'r') as f:
                history_data = json.load(f)
                history = ChatHistory(**history_data)
        else:
            history = ChatHistory(
                session_id=session_id,
                messages=[],
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        
        # Add message and update timestamp
        history.messages.append(message)
        history.updated_at = datetime.now()
        
        # Save to file
        with open(history_file, 'w') as f:
            json.dump(history.model_dump(), f, indent=2, default=str)
    
    async def get_chat_history(self, session_id: str) -> List[ChatMessage]:
        """Get chat history for a session"""
        history_file = self.chat_history_dir / f"{session_id}.json"
        
        if not history_file.exists():
            return []
        
        try:
            with open(history_file, 'r') as f:
                history_data = json.load(f)
                # Convert timestamp strings back to datetime objects
                for msg in history_data.get('messages', []):
                    if isinstance(msg['timestamp'], str):
                        msg['timestamp'] = datetime.fromisoformat(msg['timestamp'])
                
                history = ChatHistory(**history_data)
                return history.messages
        except Exception as e:
            print(f"Error loading chat history: {str(e)}")
            return []
    
    async def clear_chat_history(self, session_id: str):
        """Clear chat history for a session"""
        history_file = self.chat_history_dir / f"{session_id}.json"
        
        if history_file.exists():
            history_file.unlink()
    
    async def get_all_sessions(self) -> List[str]:
        """Get all session IDs"""
        sessions = []
        for file in self.chat_history_dir.glob("*.json"):
            sessions.append(file.stem)
        return sessions