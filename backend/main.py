"""
Main FastAPI Application for AI RAG Agent
Provides REST API endpoints for document upload, querying, and management
"""
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List
import aiofiles

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from models import (
    QueryRequest, 
    QueryResponse, 
    UploadResponse, 
    DocumentInfo,
    HealthResponse,
    DeleteResponse,
    ChatRequest
)
from rag_engine_faiss import RAGEngine

# Initialize FastAPI app
app = FastAPI(
    title="AI RAG Agent API",
    description="Retrieval-Augmented Generation API with Ollama, LlamaIndex, and ChromaDB",
    version="1.0.0"
)

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize directories
settings.initialize_directories()

# Initialize RAG engine (singleton)
rag_engine: RAGEngine = None

@app.on_event("startup")
async def startup_event():
    """Initialize RAG engine on startup"""
    global rag_engine
    try:
        rag_engine = RAGEngine()
        print("✅ FastAPI server started successfully!")
    except Exception as e:
        print(f"❌ Error starting RAG engine: {e}")
        raise

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "AI RAG Agent API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "upload": "/upload",
            "query": "/query",
            "chat": "/chat",
            "documents": "/documents",
            "clear": "/clear"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Check health status of all components
    """
    if rag_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG engine not initialized"
        )
    
    health = rag_engine.health_check()
    stats = rag_engine.get_stats()
    
    return HealthResponse(
        status="healthy" if all([
            health["ollama_connected"],
            health["chroma_connected"]
        ]) else "degraded",
        ollama_connected=health["ollama_connected"],
        chroma_connected=health["chroma_connected"],
        model=stats.get("model", "unknown")
    )

@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and ingest a document into the RAG system
    
    Supports: PDF, TXT, MD, DOCX
    """
    if rag_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG engine not initialized"
        )
    
    # Validate file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    if file_size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum allowed size of {settings.MAX_FILE_SIZE} bytes"
        )
    
    # Validate file extension
    allowed_extensions = {".pdf", ".txt", ".md", ".docx", ".doc"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {file_ext} not supported. Allowed types: {allowed_extensions}"
        )
    
    try:
        # Save file to upload directory
        file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Ingest into RAG system
        result = rag_engine.ingest_documents([file_path])
        
        if not result["success"]:
            # Clean up file if ingestion failed
            os.remove(file_path)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["message"]
            )
        
        return UploadResponse(
            success=True,
            message=f"Document '{file.filename}' uploaded and processed successfully",
            document=DocumentInfo(
                filename=file.filename,
                size=file_size,
                upload_time=datetime.now().isoformat(),
                num_chunks=result["num_chunks"]
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up file if it exists
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )

@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_documents(request: QueryRequest):
    """
    Query the RAG system with a question
    
    Returns AI-generated answer based on uploaded documents
    """
    if rag_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG engine not initialized"
        )
    
    try:
        result = rag_engine.query(
            query_text=request.query,
            top_k=request.top_k
        )
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            metadata=result["metadata"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )

@app.post("/chat", response_model=QueryResponse, tags=["Query"])
async def chat_with_context(request: ChatRequest):
    """
    Chat with the RAG system including conversation history
    
    Maintains context across multiple turns
    """
    if rag_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG engine not initialized"
        )
    
    try:
        # Convert chat history to dict format
        history = [
            {"role": msg.role, "content": msg.content}
            for msg in request.history
        ]
        
        result = rag_engine.query(
            query_text=request.query,
            top_k=request.top_k,
            chat_history=history
        )
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            metadata=result["metadata"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat: {str(e)}"
        )

@app.get("/documents", tags=["Documents"])
async def list_documents():
    """
    List all uploaded documents
    """
    try:
        upload_dir = Path(settings.UPLOAD_DIR)
        documents = []
        
        for file_path in upload_dir.iterdir():
            if file_path.is_file():
                stat = file_path.stat()
                documents.append({
                    "filename": file_path.name,
                    "size": stat.st_size,
                    "upload_time": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        stats = rag_engine.get_stats() if rag_engine else {}
        
        return {
            "documents": documents,
            "total_documents": len(documents),
            "total_chunks": stats.get("num_chunks", 0)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing documents: {str(e)}"
        )

@app.delete("/clear", response_model=DeleteResponse, tags=["Documents"])
async def clear_all_documents():
    """
    Clear all documents and reset the index
    
    WARNING: This will delete all uploaded documents and clear the vector database
    """
    if rag_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG engine not initialized"
        )
    
    try:
        # Clear ChromaDB index
        success = rag_engine.clear_index()
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to clear index"
            )
        
        # Delete all uploaded files
        upload_dir = Path(settings.UPLOAD_DIR)
        deleted_count = 0
        
        for file_path in upload_dir.iterdir():
            if file_path.is_file():
                file_path.unlink()
                deleted_count += 1
        
        return DeleteResponse(
            success=True,
            message="All documents cleared successfully",
            deleted_count=deleted_count
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error clearing documents: {str(e)}"
        )

@app.get("/stats", tags=["Info"])
async def get_stats():
    """
    Get statistics about the RAG system
    """
    if rag_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG engine not initialized"
        )
    
    return rag_engine.get_stats()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )
