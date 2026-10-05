"""
Azure Horizon Resort & Spa — RAG Engine
LlamaIndex tabanlı RAG pipeline: DeepSeek LLM + HuggingFace Embeddings
Hybrid Search: Vektör + BM25 + Cross-Encoder Reranker
"""

import os
from typing import List, Any
from pathlib import Path
from dotenv import load_dotenv

# NLTK verisini proje içindeki yazılabilir klasörde tut (container/sunucu uyumluluğu)
import nltk
_NLTK_DIR = str(Path(__file__).parent / ".cache" / "nltk_data")
nltk.data.path.insert(0, _NLTK_DIR)
os.environ["NLTK_DATA"] = _NLTK_DIR

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.core.chat_engine.context import ContextChatEngine
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever


# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"

SYSTEM_PROMPT = """\
Sen "Azure Horizon Resort & Spa" otelinin sanal asistanısın.
Görevin, misafirlere otel hakkında doğru ve yardımcı bilgiler sunmaktır.

KURALLAR:
1. Yalnızca sana verilen bilgi bankasındaki (context) bilgilere dayanarak cevap ver.
2. Bilgi bankasında bulunmayan konularda kesinlikle tahmin yürütme veya uydurma.
   Bunun yerine şunu söyle: "Bu konuda bilgi bankamda yeterli bilgi bulunamadı. \
Detaylı bilgi için resepsiyonumuzu 0264 511 42 00 numarasından arayabilirsiniz."
3. Her zaman Türkçe cevap ver.
4. Nazik, profesyonel ve yardımsever bir üslup kullan.
5. Fiyat, saat veya kural gibi somut bilgileri mümkün olduğunca net ve doğru aktar.
"""


# ---------------------------------------------------------------------------
# Hybrid Retriever: Vektör + BM25
# ---------------------------------------------------------------------------
class HybridRetriever(BaseRetriever):
    """Vektör araması ve BM25 keyword aramasını birleştiren hybrid retriever.

    Aşama 1a: Vektör benzerliği ile aday getir (anlamsal eşleşme)
    Aşama 1b: BM25 keyword araması ile aday getir (kelime eşleşmesi)
    Sonuçlar birleştirilir, tekrarlar çıkarılır.
    """

    def __init__(self, vector_retriever, bm25_retriever):
        self._vector_retriever = vector_retriever
        self._bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """İki arama stratejisinin sonuçlarını birleştir."""
        # Vektör araması (anlamsal)
        vector_results = self._vector_retriever.retrieve(query_bundle)

        # BM25 araması (keyword)
        bm25_results = self._bm25_retriever.retrieve(query_bundle)

        # Sonuçları birleştir, tekrarları çıkar
        seen_ids = set()
        merged = []

        for node in vector_results + bm25_results:
            node_id = node.node.node_id
            if node_id not in seen_ids:
                seen_ids.add(node_id)
                merged.append(node)

        return merged


# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------
def _load_api_key() -> str:
    """API anahtarını ortam değişkeninden veya .env dosyasından yükle."""
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY bulunamadı! "
            "Lütfen .env dosyasına ekleyin veya ortam değişkeni olarak tanımlayın."
        )
    return api_key


def initialize_llm() -> OpenAILike:
    """DeepSeek API ile OpenAI-uyumlu LLM oluştur."""
    api_key = _load_api_key()
    return OpenAILike(
        model="deepseek-chat",
        api_base="https://api.deepseek.com/v1",
        api_key=api_key,
        is_chat_model=True,
        temperature=0.3,
        max_tokens=1024,
    )


def initialize_embedding() -> HuggingFaceEmbedding:
    """Çok dilli, lokal çalışan HuggingFace embedding modeli (Türkçe dahil 50+ dil)."""
    return HuggingFaceEmbedding(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        cache_folder=str(Path(__file__).parent / ".cache"),
    )


def load_documents(data_dir: Path = DATA_DIR):
    """data/ klasöründeki Markdown dosyalarını yükle."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Veri klasörü bulunamadı: {data_dir}")

    reader = SimpleDirectoryReader(
        input_dir=str(data_dir),
        required_exts=[".md"],
        recursive=False,
    )
    documents = reader.load_data()
    if not documents:
        raise ValueError("Hiçbir belge yüklenemedi. data/ klasörünü kontrol edin.")
    return documents


def build_index(documents) -> VectorStoreIndex:
    """Belgelerden VectorStoreIndex oluştur."""
    return VectorStoreIndex.from_documents(documents, show_progress=True)


def build_hybrid_retriever(index: VectorStoreIndex) -> HybridRetriever:
    """Hybrid retriever oluştur: Vektör + BM25.

    Vektör: Anlamsal benzerlik ile 5 aday getirir
    BM25: Kelime eşleşmesi ile 5 aday getirir
    Toplam: En fazla 10 benzersiz aday (reranker'a gider)
    """
    # Vektör retriever
    vector_retriever = index.as_retriever(similarity_top_k=5)

    # BM25 retriever — aynı node'lardan keyword index oluşturur
    nodes = list(index.docstore.docs.values())
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=5,
    )

    return HybridRetriever(vector_retriever, bm25_retriever)


def create_chat_engine(hybrid_retriever: HybridRetriever):
    """Hybrid retriever + Reranker ile ContextChatEngine oluştur.

    Aşama 1: Hybrid retriever (Vektör + BM25) → ~10 aday
    Aşama 2: Cross-encoder reranker → en alakalı 3'ünü seç
    """
    memory = ChatMemoryBuffer.from_defaults(token_limit=4096)

    # Cross-encoder reranker: adayları yeniden sıralar
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n=3,
    )

    return ContextChatEngine.from_defaults(
        retriever=hybrid_retriever,
        memory=memory,
        system_prompt=SYSTEM_PROMPT,
        node_postprocessors=[reranker],
    )


def get_source_files(response) -> List[str]:
    """Yanıttan kaynak dosya adlarını çıkar."""
    sources = set()
    if hasattr(response, "source_nodes"):
        for node in response.source_nodes:
            metadata = node.node.metadata
            if "file_name" in metadata:
                sources.add(metadata["file_name"])
            elif "file_path" in metadata:
                sources.add(Path(metadata["file_path"]).name)
    return sorted(sources)


# ---------------------------------------------------------------------------
# Tek seferde tüm pipeline'ı başlat
# ---------------------------------------------------------------------------
def initialize_rag_pipeline():
    """
    Tüm RAG pipeline'ını başlatır.
    Hybrid Search: Vektör + BM25 + Cross-Encoder Reranker
    Returns: (chat_engine, document_count)
    """
    # Global LlamaIndex ayarları
    llm = initialize_llm()
    embed_model = initialize_embedding()
    Settings.llm = llm
    Settings.embed_model = embed_model

    # Belgeleri yükle ve index oluştur
    documents = load_documents()
    index = build_index(documents)

    # Hybrid retriever oluştur (Vektör + BM25)
    hybrid_retriever = build_hybrid_retriever(index)

    # Chat engine oluştur (Hybrid + Reranker)
    chat_engine = create_chat_engine(hybrid_retriever)

    return chat_engine, len(documents)
