# Azure Horizon Resort & Spa — AI Hotel Assistant

A RAG (Retrieval-Augmented Generation) assistant that answers guest questions about a hotel
from its own knowledge base — room prices, check-in rules, spa services, restaurant menu, FAQs —
and shows which documents each answer came from.

> "Azure Horizon Resort & Spa" is a fictional hotel; all names and data are placeholders.

## Architecture

```
 Hotel website + chat widget (web/index.html)
        │  POST /api/chat
        ▼
 FastAPI (api.py) ── per-session conversation memory
        │
        ▼
 RAG pipeline (rag_engine.py)
   1. Hybrid retrieval: vector search (multilingual embeddings) + BM25 keyword search
   2. Cross-encoder reranker picks the 3 most relevant chunks
   3. LLM (DeepSeek, OpenAI-compatible API) answers only from that context
        │
        ▼
 Answer + source documents
```

- **Hybrid search** — semantic matches ("room size" ≈ "m²") and exact terms ("check-in") are both covered.
- **Reranking** — the multilingual cross-encoder `mmarco-mMiniLMv2-L12-H384-v1` narrows ~10 candidates
  to the best 3. (The English-only `ms-marco` reranker pushed correct Turkish passages out of the top 3.)
- **Grounded answers** — the system prompt forbids answering outside the knowledge base and
  redirects the guest to reception instead of guessing.
- **Session memory** — follow-up questions ("and how much is that?") keep their context;
  the index, retriever and reranker are shared across sessions, memory is not.
- **Local embeddings** — `paraphrase-multilingual-MiniLM-L12-v2` runs on CPU, no embedding API cost.

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Service status and number of indexed documents |
| `GET` | `/api/knowledge` | Documents in the knowledge base |
| `POST` | `/api/chat` | `{ "message": "...", "session_id": "optional" }` → answer + sources |
| `POST` | `/api/chat/reset` | Clears a session's conversation memory |
| `GET` | `/docs` | Interactive OpenAPI documentation |

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # macOS/Linux: cp .env.example .env  — then add DEEPSEEK_API_KEY
uvicorn api:app --port 8000
```

Open `http://localhost:8000` for the chat UI and `http://localhost:8000/docs` for the API docs.
The first start downloads the embedding and reranker models into `.cache/`.

## Knowledge base

Markdown files in `data/` are indexed automatically at startup. Add a new `.md` file and restart
the server to extend what the assistant knows.

## Tech stack

Python · FastAPI · LlamaIndex · DeepSeek LLM · HuggingFace sentence-transformers · BM25 · HTML/CSS/JS
