/**
 * API Client for AI RAG Agent Backend
 * Handles all HTTP requests to the FastAPI backend
 */
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Check backend health status
 */
export const checkHealth = async () => {
  const response = await api.get('/health');
  return response.data;
};

/**
 * Upload a document to the RAG system
 */
export const uploadDocument = async (file, onProgress) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress: (progressEvent) => {
      if (onProgress) {
        const percentCompleted = Math.round(
          (progressEvent.loaded * 100) / progressEvent.total
        );
        onProgress(percentCompleted);
      }
    },
  });

  return response.data;
};

/**
 * Query the RAG system
 */
export const queryDocuments = async (query, topK = 3) => {
  const response = await api.post('/query', {
    query,
    top_k: topK,
  });
  return response.data;
};

/**
 * Chat with context (conversation history)
 */
export const chatWithContext = async (query, history = [], topK = 3) => {
  const response = await api.post('/chat', {
    query,
    history,
    top_k: topK,
  });
  return response.data;
};

/**
 * List all uploaded documents
 */
export const listDocuments = async () => {
  const response = await api.get('/documents');
  return response.data;
};

/**
 * Clear all documents
 */
export const clearAllDocuments = async () => {
  const response = await api.delete('/clear');
  return response.data;
};

/**
 * Get system statistics
 */
export const getStats = async () => {
  const response = await api.get('/stats');
  return response.data;
};

export default api;
