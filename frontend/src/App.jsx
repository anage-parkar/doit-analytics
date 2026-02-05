/**
 * Main App Component
 * Orchestrates the entire AI RAG Agent UI
 */
import { useState, useEffect, useRef } from 'react';
import './App.css';
import {
  checkHealth,
  uploadDocument,
  chatWithContext,
  listDocuments,
  clearAllDocuments,
  getStats,
} from './api';

function App() {
  // State management
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [health, setHealth] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [stats, setStats] = useState(null);
  const [activeTab, setActiveTab] = useState('chat'); // chat, documents, settings

  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // Auto-scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Initialize: Check health and load documents
  useEffect(() => {
    initializeApp();
  }, []);

  const initializeApp = async () => {
    try {
      const healthData = await checkHealth();
      setHealth(healthData);

      const docsData = await listDocuments();
      setDocuments(docsData.documents || []);

      const statsData = await getStats();
      setStats(statsData);
    } catch (error) {
      console.error('Error initializing app:', error);
      setMessages([
        {
          role: 'system',
          content: '⚠️ Backend connection failed. Please ensure the backend server is running.',
        },
      ]);
    }
  };

  // Handle file upload
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setUploading(true);
    setUploadProgress(0);

    try {
      const result = await uploadDocument(file, setUploadProgress);

      setMessages((prev) => [
        ...prev,
        {
          role: 'system',
          content: `✅ Document "${file.name}" uploaded successfully! (${result.document.num_chunks} chunks)`,
        },
      ]);

      // Refresh documents list and stats
      const docsData = await listDocuments();
      setDocuments(docsData.documents || []);

      const statsData = await getStats();
      setStats(statsData);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'system',
          content: `❌ Upload failed: ${error.response?.data?.detail || error.message}`,
        },
      ]);
    } finally {
      setUploading(false);
      setUploadProgress(0);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  // Handle sending a message
  const handleSendMessage = async () => {
    if (!inputValue.trim() || loading) return;

    const userMessage = {
      role: 'user',
      content: inputValue,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue('');
    setLoading(true);

    try {
      // Get conversation history (exclude system messages)
      const history = messages
        .filter((msg) => msg.role !== 'system')
        .map((msg) => ({
          role: msg.role,
          content: msg.content,
          timestamp: msg.timestamp,
        }));

      const response = await chatWithContext(inputValue, history, 3);

      const assistantMessage = {
        role: 'assistant',
        content: response.answer,
        sources: response.sources,
        metadata: response.metadata,
        timestamp: new Date().toISOString(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage = {
        role: 'system',
        content: `❌ Error: ${error.response?.data?.detail || error.message}`,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  // Handle clearing all documents
  const handleClearDocuments = async () => {
    if (!confirm('Are you sure you want to delete all documents? This cannot be undone.')) {
      return;
    }

    try {
      await clearAllDocuments();
      setMessages([
        {
          role: 'system',
          content: '✅ All documents cleared successfully',
        },
      ]);

      setDocuments([]);
      const statsData = await getStats();
      setStats(statsData);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'system',
          content: `❌ Error clearing documents: ${error.message}`,
        },
      ]);
    }
  };

  // Format file size
  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <h1>🤖 AI RAG Agent</h1>
        <div className="header-info">
          <span className={`status ${health?.ollama_connected ? 'online' : 'offline'}`}>
            {health?.ollama_connected ? '🟢 Online' : '🔴 Offline'}
          </span>
          <span className="model-info">
            {health?.model || 'Loading...'}
          </span>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="tabs">
        <button
          className={`tab ${activeTab === 'chat' ? 'active' : ''}`}
          onClick={() => setActiveTab('chat')}
        >
          💬 Chat
        </button>
        <button
          className={`tab ${activeTab === 'documents' ? 'active' : ''}`}
          onClick={() => setActiveTab('documents')}
        >
          📚 Documents ({documents.length})
        </button>
        <button
          className={`tab ${activeTab === 'settings' ? 'active' : ''}`}
          onClick={() => setActiveTab('settings')}
        >
          ⚙️ Settings
        </button>
      </nav>

      {/* Main Content */}
      <main className="main-content">
        {/* Chat Tab */}
        {activeTab === 'chat' && (
          <div className="chat-container">
            <div className="messages">
              {messages.length === 0 && (
                <div className="welcome-message">
                  <h2>👋 Welcome to AI RAG Agent!</h2>
                  <p>Upload documents and start asking questions.</p>
                  <p>I'll use RAG (Retrieval-Augmented Generation) to provide accurate answers based on your documents.</p>
                </div>
              )}

              {messages.map((msg, idx) => (
                <div key={idx} className={`message ${msg.role}`}>
                  <div className="message-header">
                    <span className="role">
                      {msg.role === 'user' ? '👤 You' : msg.role === 'assistant' ? '🤖 AI' : '⚙️ System'}
                    </span>
                    {msg.timestamp && (
                      <span className="timestamp">
                        {new Date(msg.timestamp).toLocaleTimeString()}
                      </span>
                    )}
                  </div>
                  <div className="message-content">{msg.content}</div>

                  {msg.sources && msg.sources.length > 0 && (
                    <details className="sources">
                      <summary>📄 Sources ({msg.sources.length})</summary>
                      <div className="sources-list">
                        {msg.sources.map((source, i) => (
                          <div key={i} className="source-item">
                            <strong>Source {i + 1}:</strong>
                            <p>{source}</p>
                          </div>
                        ))}
                      </div>
                    </details>
                  )}
                </div>
              ))}

              {loading && (
                <div className="message assistant loading">
                  <div className="message-header">
                    <span className="role">🤖 AI</span>
                  </div>
                  <div className="message-content">
                    <div className="typing-indicator">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="input-area">
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileUpload}
                style={{ display: 'none' }}
                accept=".pdf,.txt,.md,.docx,.doc"
              />

              <button
                className="upload-btn"
                onClick={() => fileInputRef.current?.click()}
                disabled={uploading}
                title="Upload document"
              >
                {uploading ? `${uploadProgress}%` : '📎'}
              </button>

              <input
                type="text"
                className="message-input"
                placeholder="Ask a question about your documents..."
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                disabled={loading}
              />

              <button
                className="send-btn"
                onClick={handleSendMessage}
                disabled={loading || !inputValue.trim()}
              >
                {loading ? '⏳' : '📤'}
              </button>
            </div>
          </div>
        )}

        {/* Documents Tab */}
        {activeTab === 'documents' && (
          <div className="documents-tab">
            <div className="tab-header">
              <h2>📚 Uploaded Documents</h2>
              <button className="danger-btn" onClick={handleClearDocuments}>
                🗑️ Clear All
              </button>
            </div>

            {stats && (
              <div className="stats-card">
                <div className="stat">
                  <span className="stat-label">Total Documents:</span>
                  <span className="stat-value">{documents.length}</span>
                </div>
                <div className="stat">
                  <span className="stat-label">Total Chunks:</span>
                  <span className="stat-value">{stats.num_chunks}</span>
                </div>
              </div>
            )}

            <div className="documents-list">
              {documents.length === 0 ? (
                <div className="empty-state">
                  <p>📭 No documents uploaded yet</p>
                  <p>Click the 📎 button to upload your first document</p>
                </div>
              ) : (
                documents.map((doc, idx) => (
                  <div key={idx} className="document-card">
                    <div className="doc-icon">📄</div>
                    <div className="doc-info">
                      <div className="doc-name">{doc.filename}</div>
                      <div className="doc-meta">
                        <span>{formatFileSize(doc.size)}</span>
                        <span>•</span>
                        <span>{new Date(doc.upload_time).toLocaleString()}</span>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        )}

        {/* Settings Tab */}
        {activeTab === 'settings' && (
          <div className="settings-tab">
            <h2>⚙️ System Settings</h2>

            <div className="settings-section">
              <h3>🔌 Connection Status</h3>
              <div className="status-grid">
                <div className="status-item">
                  <span>Ollama:</span>
                  <span className={health?.ollama_connected ? 'success' : 'error'}>
                    {health?.ollama_connected ? '✅ Connected' : '❌ Disconnected'}
                  </span>
                </div>
                <div className="status-item">
                  <span>ChromaDB:</span>
                  <span className={health?.chroma_connected ? 'success' : 'error'}>
                    {health?.chroma_connected ? '✅ Connected' : '❌ Disconnected'}
                  </span>
                </div>
              </div>
            </div>

            <div className="settings-section">
              <h3>🤖 Model Information</h3>
              <div className="info-grid">
                <div className="info-item">
                  <span>LLM Model:</span>
                  <span>{stats?.model || 'Unknown'}</span>
                </div>
                <div className="info-item">
                  <span>Embedding Model:</span>
                  <span>{stats?.embedding_model || 'Unknown'}</span>
                </div>
              </div>
            </div>

            <div className="settings-section">
              <h3>📊 Statistics</h3>
              <div className="info-grid">
                <div className="info-item">
                  <span>Indexed Chunks:</span>
                  <span>{stats?.num_chunks || 0}</span>
                </div>
                <div className="info-item">
                  <span>Index Status:</span>
                  <span className={stats?.has_index ? 'success' : 'warning'}>
                    {stats?.has_index ? '✅ Ready' : '⚠️ Empty'}
                  </span>
                </div>
              </div>
            </div>

            <button className="refresh-btn" onClick={initializeApp}>
              🔄 Refresh Status
            </button>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
