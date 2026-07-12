import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { Activity, FileUp, Search, ShieldCheck } from "lucide-react";
import "./styles.css";

const API = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

type DocumentRecord = {
  id: string;
  filename: string;
  status: string;
  chunk_count: number;
  error?: string | null;
};

type Hit = {
  chunk_id: string;
  document_id: string;
  text: string;
  dense_score: number;
  bm25_score: number;
  rerank_score: number;
  combined_score: number;
  metadata: Record<string, unknown>;
};

type AskResponse = {
  answer: string;
  claims: { text: string; supported: boolean }[];
  hits: Hit[];
  latency_ms: number;
  insufficient_evidence: boolean;
};

function App() {
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState<AskResponse | null>(null);
  const [busy, setBusy] = useState(false);

  async function refreshDocuments() {
    const res = await fetch(`${API}/documents`);
    setDocuments(await res.json());
  }

  useEffect(() => {
    refreshDocuments();
    const timer = window.setInterval(refreshDocuments, 2500);
    return () => window.clearInterval(timer);
  }, []);

  async function upload(file: File) {
    const form = new FormData();
    form.append("file", file);
    await fetch(`${API}/documents`, { method: "POST", body: form });
    await refreshDocuments();
  }

  async function ask() {
    if (!question.trim()) return;
    setBusy(true);
    const res = await fetch(`${API}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: 8 }),
    });
    setResponse(await res.json());
    setBusy(false);
  }

  return (
    <main>
      <header className="topbar">
        <div>
          <h1>Hybrid Search RAG</h1>
          <p>Upload documents, retrieve with dense plus keyword search, rerank evidence, and verify citations.</p>
        </div>
        <div className="status"><Activity size={18} /> Project 1 foundation</div>
      </header>

      <section className="workspace">
        <aside className="panel">
          <h2><FileUp size={18} /> Documents</h2>
          <label className="upload">
            <input
              type="file"
              accept=".pdf,.md,.markdown,.html,.htm,.txt"
              onChange={(event) => event.target.files?.[0] && upload(event.target.files[0])}
            />
            Upload PDF, Markdown, HTML, or text
          </label>
          <div className="docList">
            {documents.map((doc) => (
              <div className="doc" key={doc.id}>
                <strong>{doc.filename}</strong>
                <span>{doc.status} · {doc.chunk_count} chunks</span>
                {doc.error && <small>{doc.error}</small>}
              </div>
            ))}
            {!documents.length && <p className="muted">No documents indexed yet.</p>}
          </div>
        </aside>

        <section className="panel mainPanel">
          <h2><Search size={18} /> Ask with citations</h2>
          <div className="askRow">
            <input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask a question about your indexed documents" />
            <button onClick={ask} disabled={busy}>{busy ? "Searching..." : "Ask"}</button>
          </div>

          {response && (
            <div className="answer">
              <div className={response.insufficient_evidence ? "badge warning" : "badge ok"}>
                <ShieldCheck size={16} /> {response.insufficient_evidence ? "Needs more evidence" : "Verified"}
              </div>
              <p>{response.answer}</p>
              <span className="metric">{Math.round(response.latency_ms)} ms total latency</span>
            </div>
          )}

          {response && (
            <div className="debugGrid">
              <div>
                <h3>Claims</h3>
                {response.claims.map((claim, index) => (
                  <div className="claim" key={index}>
                    <span className={claim.supported ? "dot green" : "dot red"} />
                    {claim.text}
                  </div>
                ))}
              </div>
              <div>
                <h3>Retrieval Debugging</h3>
                {response.hits.map((hit) => (
                  <div className="hit" key={hit.chunk_id}>
                    <strong>{String(hit.metadata.filename ?? "document")}</strong>
                    <p>{hit.text.slice(0, 260)}...</p>
                    <code>
                      dense {hit.dense_score.toFixed(3)} · bm25 {hit.bm25_score.toFixed(3)} · rerank {hit.rerank_score.toFixed(3)}
                    </code>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>
      </section>

      <section className="roadmap">
        <div><strong>Project 2 ready next:</strong> OCR, layout detection, table extraction, previews, human review, audit logs.</div>
        <div><strong>Project 3 ready next:</strong> LangGraph query rewriting, iterative retrieval, tool routing, traces, streaming.</div>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
