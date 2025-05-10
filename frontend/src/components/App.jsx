import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import '../styles/App.css';
import folderIcon from '../assets/folder.png';


const ASSISTANT_COLOR = '#c9ecfd';   // Response bubbles & loading
const USER_COLOR      = '#e1e2e6';   // user bubbles
const ACCENT_COLOR    = '#3aa2da';   // buttons, outline, glow

const glow = `0 0 0 .18rem rgba(58,162,218,.25)`; // box‑shadow

const App = () => {
  const [messages, setMessages]   = useState([]);
  const [inputText, setInputText] = useState('');
  const [files, setFiles]         = useState([]);   // File[]
  const [isLoading, setIsLoading] = useState(false);

  // hover & focus flags
  const [folderHover, setFolderHover] = useState(false);
  const [sendHover,   setSendHover]   = useState(false);
  const [inputFocus,  setInputFocus]  = useState(false);
  const [inputHover,  setInputHover]  = useState(false);

  const bottomRef    = useRef(null);
  const fileInputRef = useRef(null);

  /* ---------- scroll to newest ---------- */
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  /* ---------- file select ---------- */
  const handleFileChange = (e) => {
    const selected = Array.from(e.target.files || []);
    if (selected.length) setFiles(selected);
  };

  /* ---------- submit ---------- */
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputText && files.length === 0) return;

    // optimistic user bubble
    setMessages((prev) => [
      ...prev,
      { sender: 'user', text: inputText, fileNames: files.map(f => f.name) }
    ]);
    setIsLoading(true);

    const formData = new FormData();
    formData.append('question', inputText || '');
    files.forEach(f => formData.append('files', f));

    setInputText('');
    setFiles([]);

    try {
      const res = await axios.post('http://localhost:8000/api/upload-tax-return', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        responseType: 'blob',
      });
      // Create a download link and click it
      const blob = new Blob([res.data], { type: 'text/markdown' });
      const url  = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;

      // Pull filename from Content‑Disposition or fallback
      const disposition = res.headers['content-disposition'];
      const match = disposition && disposition.match(/filename="?(.+)"?/);
      link.download = match?.[1] || 'report.md';

      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      const errorText = err.response?.data?.detail || err.message || JSON.stringify(err);
      setMessages((p) => [...p, { sender: 'assistant', text: 'Error: ' + errorText }]);
    } finally {
      setIsLoading(false);
    }
  };

  /* ---------- helpers ---------- */
  const inputBoxShadow = (inputFocus || inputHover) ? glow : undefined;
  const folderShadow   = folderHover ? glow : undefined;
  const sendShadow     = sendHover   ? glow : undefined;

  /* ---------- render ---------- */
  return (
    <div className="d-flex flex-column vh-100">
      {/* Chat window */}
      <div className="flex-grow-1 overflow-auto px-3 py-4" style={{ backgroundColor: '#f8f9fa' }}>
        <div className="container">
          {messages.map((msg, idx) => (
            <div key={idx} className={`mb-3 d-flex ${msg.sender === 'user' ? 'justify-content-end' : 'justify-content-start'}`}>
              <div className="p-3 rounded shadow-sm" style={{ backgroundColor: msg.sender === 'assistant' ? ASSISTANT_COLOR : USER_COLOR, maxWidth: '75%' }}>
                {msg.fileNames?.length > 0 && (
                  <div className="small text-muted mb-1">
                    {msg.fileNames.map((n, i) => (<span key={i}>📎 {n}{i < msg.fileNames.length - 1 ? ', ' : ''}</span>))}
                  </div>
                )}
                {msg.sender === 'assistant'
                  ? <div className="markdown-content">
                    <ReactMarkdown 
                      remarkPlugins={[remarkGfm]}
                      rehypePlugins={[rehypeHighlight]}
                      className="markdown-content"
                  >{msg.text}</ReactMarkdown></div>
                  : <div>{msg.text}</div>}
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="mb-3 d-flex justify-content-start">
              <div className="p-3 rounded shadow-sm" style={{ backgroundColor: ASSISTANT_COLOR, maxWidth: '75%' }}>
                <div className="typing-indicator"><span></span><span></span><span></span></div>
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      </div>

      {/* Input bar */}
      <form className="border-top bg-white p-3" style={{ boxShadow: '0 -2px 5px rgba(0,0,0,0.05)' }} onSubmit={handleSubmit}>
        <div className="container">
          <div className="row g-2 align-items-center">
            {/* Folder button + label */}
            <div className="col-auto d-flex align-items-center">
              <button
                type="button"
                className="btn d-flex align-items-center justify-content-center"
                style={{
                  padding: '6px 10px',
                  border: `1px solid ${ACCENT_COLOR}`,
                  backgroundColor: '#fff',
                  boxShadow: folderShadow,
                }}
                onMouseEnter={() => setFolderHover(true)}
                onMouseLeave={() => setFolderHover(false)}
                onClick={() => fileInputRef.current?.click()}
                disabled={isLoading}
              >
                <img src={folderIcon} alt="Attach" style={{ width: '28px', height: '28px' }} />
              </button>
              {files.length > 0 && (
                <span className="ms-2 small text-muted">
                  {files.length === 1 ? files[0].name : `${files.length} files`}
                </span>
              )}
              <input ref={fileInputRef} type="file" multiple style={{ display: 'none' }} onChange={handleFileChange} />
            </div>

            {/* Text input */}
            <div className="col">
              <input
                type="text"
                className="form-control"
                placeholder="Type a message or attach files"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onFocus={() => setInputFocus(true)}
                onBlur={() => setInputFocus(false)}
                onMouseEnter={() => setInputHover(true)}
                onMouseLeave={() => setInputHover(false)}
                style={{ borderColor: (inputFocus || inputHover) ? ACCENT_COLOR : undefined, boxShadow: inputBoxShadow }}
                disabled={isLoading}
              />
            </div>

            {/* Send button */}
            <div className="col-auto">
              <button
                type="submit"
                className="btn"
                style={{
                  backgroundColor: ACCENT_COLOR,
                  borderColor: ACCENT_COLOR,
                  color: '#fff',
                  boxShadow: sendShadow,
                }}
                onMouseEnter={() => setSendHover(true)}
                onMouseLeave={() => setSendHover(false)}
                disabled={isLoading}
              >
                {isLoading ? <span className="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> : 'Send'}
              </button>
            </div>
          </div>
        </div>
      </form>
    </div>
  );
};

export default App;
