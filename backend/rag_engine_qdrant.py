"""
RAG Engine - Windows-Compatible Version using Qdrant
Integrates LlamaIndex, Ollama, and Qdrant (in-memory mode)
"""
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import qdrant_client
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    Document
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceSplitter

from config import settings

class RAGEngine:
    """
    Retrieval-Augmented Generation Engine (Windows Compatible)
    
    Handles:
    - Document ingestion and chunking
    - Embedding generation with Ollama
    - Vector storage with Qdrant (in-memory)
    - Semantic search and retrieval
    - Context-aware response generation
    """
    
    def __init__(self):
        """Initialize RAG components"""
        print("🚀 Initializing RAG Engine (Windows Version with Qdrant)...")
        
        # Initialize Ollama LLM
        self.llm = Ollama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.7,
            request_timeout=120.0
        )
        
        # Initialize Ollama Embeddings
        self.embed_model = OllamaEmbedding(
            model_name=settings.OLLAMA_EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
        )
        
        # Configure LlamaIndex global settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50
        
        # Initialize Qdrant client (in-memory mode for simplicity)
        # For persistence, you can use: client = QdrantClient(path="./qdrant_db")
        self.qdrant_client = qdrant_client.QdrantClient(location=":memory:")
        
        # Collection name
        self.collection_name = settings.COLLECTION_NAME
        
        # Create vector store
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
        )
        
        # Create storage context
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        # Initialize index (will be None until documents are loaded)
        self.index = None
        
        # Try to load existing index
        self._load_existing_index()
        
        print("✅ RAG Engine initialized successfully!")
    
    def _load_existing_index(self):
        """Load existing index from Qdrant if documents exist"""
        try:
            collections = self.qdrant_client.get_collections().collections
            if any(c.name == self.collection_name for c in collections):
                collection_info = self.qdrant_client.get_collection(self.collection_name)
                if collection_info.points_count > 0:
                    print(f"📚 Found {collection_info.points_count} existing chunks in Qdrant")
                    self.index = VectorStoreIndex.from_vector_store(
                        vector_store=self.vector_store,
                        storage_context=self.storage_context
                    )
                    print("✅ Existing index loaded successfully")
        except Exception as e:
            print(f"⚠️ Could not load existing index: {e}")
            self.index = None
    
    def ingest_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Ingest documents into the RAG system
        
        Args:
            file_paths: List of file paths to ingest
            
        Returns:
            Dictionary with ingestion statistics
        """
        try:
            print(f"📥 Ingesting {len(file_paths)} document(s)...")
            
            # Load documents
            documents = []
            for file_path in file_paths:
                if os.path.exists(file_path):
                    reader = SimpleDirectoryReader(
                        input_files=[file_path]
                    )
                    docs = reader.load_data()
                    documents.extend(docs)
            
            if not documents:
                return {
                    "success": False,
                    "message": "No valid documents found",
                    "num_documents": 0,
                    "num_chunks": 0
                }
            
            # Parse into chunks
            parser = SentenceSplitter(
                chunk_size=Settings.chunk_size,
                chunk_overlap=Settings.chunk_overlap
            )
            nodes = parser.get_nodes_from_documents(documents)
            
            # Create or update index
            if self.index is None:
                self.index = VectorStoreIndex(
                    nodes=nodes,
                    storage_context=self.storage_context
                )
            else:
                # Add to existing index
                for node in nodes:
                    self.index.insert_nodes([node])
            
            print(f"✅ Successfully ingested {len(documents)} document(s) into {len(nodes)} chunks")
            
            return {
                "success": True,
                "message": "Documents ingested successfully",
                "num_documents": len(documents),
                "num_chunks": len(nodes)
            }
            
        except Exception as e:
            print(f"❌ Error ingesting documents: {e}")
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "num_documents": 0,
                "num_chunks": 0
            }
    
    def query(
        self, 
        query_text: str, 
        top_k: int = 3,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            query_text: User's question
            top_k: Number of relevant chunks to retrieve
            chat_history: Optional conversation history
            
        Returns:
            Dictionary with answer and sources
        """
        try:
            if self.index is None:
                return {
                    "answer": "No documents have been uploaded yet. Please upload documents first.",
                    "sources": [],
                    "metadata": {"error": "No index available"}
                }
            
            print(f"🔍 Querying: {query_text}")
            
            # Build context from chat history if provided
            context_str = ""
            if chat_history:
                context_str = "\n".join([
                    f"{msg['role']}: {msg['content']}" 
                    for msg in chat_history[-5:]  # Last 5 messages
                ])
            
            # Create query engine
            query_engine = self.index.as_query_engine(
                similarity_top_k=top_k,
                response_mode="compact"
            )
            
            # Add context if available
            full_query = query_text
            if context_str:
                full_query = f"Previous conversation:\n{context_str}\n\nCurrent question: {query_text}"
            
            # Execute query
            response = query_engine.query(full_query)
            
            # Extract sources
            sources = []
            if hasattr(response, 'source_nodes'):
                sources = [
                    node.node.get_content()[:300] + "..."  # First 300 chars
                    for node in response.source_nodes
                ]
            
            print(f"✅ Query completed successfully")
            
            return {
                "answer": str(response),
                "sources": sources,
                "metadata": {
                    "model": settings.OLLAMA_MODEL,
                    "num_sources": len(sources),
                    "top_k": top_k
                }
            }
            
        except Exception as e:
            print(f"❌ Error during query: {e}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "metadata": {"error": str(e)}
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index"""
        try:
            num_chunks = 0
            collections = self.qdrant_client.get_collections().collections
            if any(c.name == self.collection_name for c in collections):
                collection_info = self.qdrant_client.get_collection(self.collection_name)
                num_chunks = collection_info.points_count
            
            return {
                "num_chunks": num_chunks,
                "has_index": self.index is not None,
                "model": settings.OLLAMA_MODEL,
                "embedding_model": settings.OLLAMA_EMBEDDING_MODEL
            }
        except Exception as e:
            return {
                "error": str(e),
                "num_chunks": 0,
                "has_index": False
            }
    
    def clear_index(self) -> bool:
        """Clear all documents from the index"""
        try:
            # Delete collection if exists
            collections = self.qdrant_client.get_collections().collections
            if any(c.name == self.collection_name for c in collections):
                self.qdrant_client.delete_collection(self.collection_name)
            
            # Recreate vector store
            self.vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.collection_name,
            )
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            self.index = None
            print("✅ Index cleared successfully")
            return True
        except Exception as e:
            print(f"❌ Error clearing index: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of all components"""
        health = {
            "ollama_connected": False,
            "qdrant_connected": False,
            "index_ready": False
        }
        
        try:
            # Check Ollama connection
            test_response = self.llm.complete("test")
            health["ollama_connected"] = True
        except:
            pass
        
        try:
            # Check Qdrant connection
            self.qdrant_client.get_collections()
            health["qdrant_connected"] = True
        except:
            pass
        
        health["index_ready"] = self.index is not None
        
        return health
