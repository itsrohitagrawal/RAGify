import os
import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from pathlib import Path
import uuid

class EmbeddingService:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path="data/embeddings",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
    
    async def create_embeddings(self, text_chunks: List[str], doc_id: str, filename: str):
        """Create embeddings for text chunks and store in ChromaDB"""
        if not text_chunks:
            return
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(text_chunks, convert_to_tensor=False)
        
        # Prepare data for ChromaDB
        ids = [f"{doc_id}_{i}" for i in range(len(text_chunks))]
        metadatas = [
            {
                "document_id": doc_id,
                "filename": filename,
                "chunk_index": i,
                "text_length": len(chunk)
            }
            for i, chunk in enumerate(text_chunks)
        ]
        
        # Store in ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=text_chunks,
            metadatas=metadatas
        )
        
        print(f"Created {len(text_chunks)} embeddings for document {filename}")
    
    async def search_similar_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar documents using embeddings"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k
            )
            
            # Format results
            similar_docs = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    similar_docs.append({
                        "content": doc,
                        "metadata": results['metadatas'][0][i],
                        "distance": results['distances'][0][i] if results['distances'] else 0
                    })
            
            return similar_docs
        
        except Exception as e:
            print(f"Error searching similar documents: {str(e)}")
            return []
    
    async def delete_document_embeddings(self, doc_id: str):
        """Delete all embeddings for a specific document"""
        try:
            # Get all IDs for this document
            results = self.collection.get(
                where={"document_id": doc_id}
            )
            
            if results['ids']:
                # Delete the embeddings
                self.collection.delete(ids=results['ids'])
                print(f"Deleted embeddings for document {doc_id}")
        
        except Exception as e:
            print(f"Error deleting embeddings for document {doc_id}: {str(e)}")
    
    async def get_document_count(self) -> int:
        """Get total number of document chunks in the collection"""
        try:
            count = self.collection.count()
            return count
        except:
            return 0
    
    async def get_documents_by_id(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document"""
        try:
            results = self.collection.get(
                where={"document_id": doc_id}
            )
            
            documents = []
            if results['documents']:
                for i, doc in enumerate(results['documents']):
                    documents.append({
                        "content": doc,
                        "metadata": results['metadatas'][i],
                        "id": results['ids'][i]
                    })
            
            return documents
        
        except Exception as e:
            print(f"Error getting documents for {doc_id}: {str(e)}")
            return []