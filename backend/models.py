"""
Pydantic models for API request and response validation
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class QueryRequest(BaseModel):
    """Request model for querying the RAG system"""
    query: str = Field(..., description="User's question or query", min_length=1)
    top_k: int = Field(default=3, description="Number of relevant chunks to retrieve", ge=1, le=10)
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the main topic of the uploaded documents?",
                "top_k": 3
            }
        }

class QueryResponse(BaseModel):
    """Response model for RAG queries"""
    answer: str = Field(..., description="AI-generated answer based on context")
    sources: List[str] = Field(default_factory=list, description="Source chunks used for answer")
    metadata: dict = Field(default_factory=dict, description="Additional metadata about the query")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Based on the documents, the main topic is...",
                "sources": ["Chunk 1 text...", "Chunk 2 text..."],
                "metadata": {
                    "model": "llama3.2:3b",
                    "num_sources": 2
                }
            }
        }

class DocumentInfo(BaseModel):
    """Information about an uploaded document"""
    filename: str
    size: int
    upload_time: str
    num_chunks: Optional[int] = None
    
class UploadResponse(BaseModel):
    """Response after document upload"""
    success: bool
    message: str
    document: Optional[DocumentInfo] = None
    
class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    ollama_connected: bool
    chroma_connected: bool
    model: str
    
class DeleteResponse(BaseModel):
    """Response after deleting documents"""
    success: bool
    message: str
    deleted_count: int

class ChatMessage(BaseModel):
    """Single chat message"""
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class ChatRequest(BaseModel):
    """Chat request with conversation history"""
    query: str
    history: List[ChatMessage] = Field(default_factory=list)
    top_k: int = Field(default=3, ge=1, le=10)
