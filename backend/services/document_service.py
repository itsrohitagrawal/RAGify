import os
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from fastapi import UploadFile
import PyPDF2
import docx

from backend.models.chat_model import DocumentInfo

class DocumentService:
    def __init__(self):
        self.document_dir = Path("data/documents")
        self.metadata_file = Path("data/document_metadata.json")
        self.document_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing metadata
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load document metadata from JSON file"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_metadata(self):
        """Save document metadata to JSON file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    async def save_document(self, file: UploadFile, doc_id: str) -> str:
        """Save uploaded document to disk"""
        file_extension = Path(file.filename).suffix
        file_path = self.document_dir / f"{doc_id}{file_extension}"
        
        # Save file
        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Update metadata
        self.metadata[doc_id] = {
            "id": doc_id,
            "filename": file.filename,
            "file_size": len(content),
            "upload_date": datetime.now().isoformat(),
            "processed": False,
            "file_path": str(file_path)
        }
        self._save_metadata()
        
        return str(file_path)
    
    async def extract_text(self, file_path: str) -> List[str]:
        """Extract text from document and split into chunks"""
        file_path = Path(file_path)
        text = ""
        
        if file_path.suffix.lower() == '.pdf':
            text = await self._extract_pdf_text(file_path)
        elif file_path.suffix.lower() == '.txt':
            text = await self._extract_txt_text(file_path)
        elif file_path.suffix.lower() == '.docx':
            text = await self._extract_docx_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        # Split text into chunks
        chunks = self._split_text_into_chunks(text)
        
        # Update metadata
        doc_id = file_path.stem
        if doc_id in self.metadata:
            self.metadata[doc_id]["processed"] = True
            self.metadata[doc_id]["chunk_count"] = len(chunks)
            self._save_metadata()
        
        return chunks
    
    async def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            raise ValueError(f"Error extracting PDF text: {str(e)}")
        return text
    
    async def _extract_txt_text(self, file_path: Path) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try different encodings
            encodings = ['latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        return file.read()
                except:
                    continue
            raise ValueError("Unable to decode text file")
    
    async def _extract_docx_text(self, file_path: Path) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise ValueError(f"Error extracting DOCX text: {str(e)}")
    
    def _split_text_into_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks"""
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # If we're not at the end, try to find a good breaking point
            if end < len(text):
                # Look for sentence endings or paragraph breaks
                for i in range(end, max(start + chunk_size // 2, end - 200), -1):
                    if text[i] in '.!?\n':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks
    
    async def get_document_list(self) -> List[DocumentInfo]:
        """Get list of all uploaded documents"""
        documents = []
        for doc_id, metadata in self.metadata.items():
            doc_info = DocumentInfo(
                id=doc_id,
                filename=metadata["filename"],
                file_size=metadata["file_size"],
                upload_date=datetime.fromisoformat(metadata["upload_date"]),
                processed=metadata["processed"],
                chunk_count=metadata.get("chunk_count")
            )
            documents.append(doc_info)
        
        return sorted(documents, key=lambda x: x.upload_date, reverse=True)
    
    async def delete_document(self, doc_id: str):
        """Delete a document and its metadata"""
        if doc_id in self.metadata:
            # Delete file
            file_path = Path(self.metadata[doc_id]["file_path"])
            if file_path.exists():
                file_path.unlink()
            
            # Remove from metadata
            del self.metadata[doc_id]
            self._save_metadata()
        else:
            raise ValueError(f"Document {doc_id} not found")