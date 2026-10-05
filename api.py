"""
Azure Horizon Resort & Spa — REST API
FastAPI backend: RAG pipeline'ını HTTP üzerinden sunar ve web arayüzünü servis eder.

Uçlar:
  GET  /api/health        - servis durumu ve indekslenen belge sayısı
  GET  /api/knowledge     - bilgi bankasındaki belgeler
  POST /api/chat          - soru gönder, yanıt + kaynakları al
  POST /api/chat/reset    - oturumun konuşma belleğini sıfırla
  GET  /                  - web sohbet arayüzü (web/index.html)

Çalıştırma:
  uvicorn api:app --port 8000
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from llama_index.core import Settings
from llama_index.core.chat_engine.context import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.postprocessor import SentenceTransformerRerank

from rag_engine import (
    DATA_DIR,
    SYSTEM_PROMPT,
    build_hybrid_retriever,
    build_index,
    get_source_files,
    initialize_embedding,
    initialize_llm,
    load_documents,
)


# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
WEB_DIR = Path(__file__).parent / "web"
MAX_SESSIONS = 200          # bellekte tutulan en fazla oturum (en eskisi atılır)
# Çok dilli (mMARCO) cross-encoder: yalnızca İngilizce eğitilmiş ms-marco modeli
# Türkçe sorularda doğru parçayı ilk 3'ün dışına itiyordu.
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
MEMORY_TOKEN_LIMIT = 4096   # oturum başına konuşma belleği
# Sistem prompt'undaki "bilgi yok" yanıtı: bu durumda kaynak gösterilmez.
NO_ANSWER_MARKER = "bilgi bankamda yeterli bilgi bulunamadı"

DOC_LABELS = {
    "genel_bilgiler.md": "Genel Bilgiler",
    "oda_tipleri_ve_fiyatlar.md": "Oda Tipleri & Fiyatlar",
    "otel_kurallari.md": "Otel Kuralları",
    "restoran_ve_aktiviteler.md": "Restoran & Aktiviteler",
    "sss.md": "Sıkça Sorulan Sorular",
    "genisletilmis_sss.md": "Genişletilmiş SSS",
}


# ---------------------------------------------------------------------------
# RAG servisi: pipeline bir kez kurulur, oturumlar belleği paylaşmaz
# ---------------------------------------------------------------------------
class RagService:
    """Index, hybrid retriever ve reranker tüm oturumlar arasında paylaşılır;
    her oturumun yalnızca kendi konuşma belleği vardır."""

    def __init__(self) -> None:
        Settings.llm = initialize_llm()
        Settings.embed_model = initialize_embedding()

        documents = load_documents()
        self.document_count = len(documents)
        index = build_index(documents)
        self._retriever = build_hybrid_retriever(index)
        self._reranker = SentenceTransformerRerank(model=RERANK_MODEL, top_n=3)

        self._sessions: OrderedDict[str, ContextChatEngine] = OrderedDict()
        self._lock = threading.Lock()

    def _engine_for(self, session_id: str) -> ContextChatEngine:
        with self._lock:
            engine = self._sessions.get(session_id)
            if engine is None:
                engine = ContextChatEngine.from_defaults(
                    retriever=self._retriever,
                    memory=ChatMemoryBuffer.from_defaults(token_limit=MEMORY_TOKEN_LIMIT),
                    system_prompt=SYSTEM_PROMPT,
                    node_postprocessors=[self._reranker],
                )
                self._sessions[session_id] = engine
                if len(self._sessions) > MAX_SESSIONS:
                    self._sessions.popitem(last=False)
            else:
                self._sessions.move_to_end(session_id)
            return engine

    def chat(self, session_id: str, message: str) -> tuple[str, list[str]]:
        response = self._engine_for(session_id).chat(message)
        answer = str(response)
        # Getirilen parçalar yanıtı desteklemiyorsa kaynak göstermek yanıltıcı olur.
        sources = [] if NO_ANSWER_MARKER in answer.lower() else get_source_files(response)
        return answer, sources

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


# ---------------------------------------------------------------------------
# Şemalar
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=64)


class Source(BaseModel):
    file: str
    label: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[Source]


class ResetRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)


def _source(file_name: str) -> Source:
    return Source(file=file_name, label=DOC_LABELS.get(file_name, file_name))


# ---------------------------------------------------------------------------
# Uygulama
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Modeller ve index sunucu açılırken bir kez yüklenir (ilk istek beklemez).
    app.state.rag = RagService()
    yield


app = FastAPI(
    title="Azure Horizon Resort & Spa — Chatbot API",
    description="Hybrid search (vektör + BM25) ve cross-encoder reranker kullanan RAG otel asistanı.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/api/health", tags=["system"])
def health() -> dict:
    return {"status": "ok", "documents": app.state.rag.document_count}


@app.get("/api/knowledge", response_model=list[Source], tags=["system"])
def knowledge() -> list[Source]:
    return [_source(p.name) for p in sorted(DATA_DIR.glob("*.md"))]


@app.post("/api/chat", response_model=ChatResponse, tags=["chat"])
def chat(body: ChatRequest) -> ChatResponse:
    """Misafir sorusunu yanıtlar. `session_id` gönderilmezse yeni oturum açılır;
    aynı `session_id` ile gelen sorular önceki konuşmayı hatırlar."""
    # Senkron uç: FastAPI bunu thread havuzunda çalıştırır, event loop bloklanmaz.
    session_id = body.session_id or uuid.uuid4().hex
    try:
        answer, files = app.state.rag.chat(session_id, body.message.strip())
    except Exception as exc:  # LLM/API hataları istemciye anlaşılır dönsün
        raise HTTPException(status_code=502, detail=f"Yanıt üretilemedi: {exc}") from exc
    return ChatResponse(
        session_id=session_id,
        answer=answer,
        sources=[_source(f) for f in files],
    )


@app.post("/api/chat/reset", status_code=204, tags=["chat"])
def reset(body: ResetRequest) -> None:
    app.state.rag.reset(body.session_id)


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")
