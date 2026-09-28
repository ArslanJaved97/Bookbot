from expander import expand
import os

os.environ["HF_HOME"] = r"D:\hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import CrossEncoder
from config import RRF_K, HYBRID_ALPHA

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class HybridRetriever:
    def __init__(self, store, embedder, use_reranker=True):
        self.store = store
        self.embedder = embedder
        self.use_reranker = use_reranker

        if use_reranker:
            print(f"[rerank] loading {RERANK_MODEL}...")
            self.reranker = CrossEncoder(RERANK_MODEL, max_length=512)
            print("[rerank] ready")
        else:
            self.reranker = None

    def search(self, query, top_k=5, alpha=HYBRID_ALPHA, book_id=None):
        query = expand(query)
        if self.store.count() == 0:
            return []

        # --- Stage 1: hybrid retrieval (fast, approximate) ---
        pool = max(top_k * 6, 30)
        vec = self.store.vector_search(self.embedder.embed_one(query), pool)
        bm = self.store.bm25_search(query, pool)

        fused = {}
        docs = {}
        for rank, r in enumerate(vec):
            key = r["id"]
            fused[key] = fused.get(key, 0) + (1 - alpha) * (1 / (RRF_K + rank + 1))
            docs[key] = r
        for rank, r in enumerate(bm):
            key = r["id"]
            fused[key] = fused.get(key, 0) + alpha * (1 / (RRF_K + rank + 1))
            docs.setdefault(key, r)

        if book_id:
            fused = {k: v for k, v in fused.items() if docs[k]["book"] == book_id}

        candidates = [{**docs[k], "rrf_score": round(v, 5)} for k, v in fused.items()]
        if not candidates:
            return []

        # --- Stage 2: cross-encoder re-ranking (slow, precise) ---
        if self.use_reranker and len(candidates) > 1:
            pairs = [(query, c["text"]) for c in candidates]
            scores = self.reranker.predict(pairs, show_progress_bar=False)
            for c, s in zip(candidates, scores):
                c["rerank_score"] = float(s)
            candidates.sort(key=lambda c: -c["rerank_score"])
        else:
            candidates.sort(key=lambda c: -c["rrf_score"])

        return candidates[:top_k]