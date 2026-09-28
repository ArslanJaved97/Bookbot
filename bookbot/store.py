import json
import pickle
from pathlib import Path

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

from config import CHROMA_PATH, BM25_PATH, REGISTRY_PATH


class BookStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="books",
            metadata={"hnsw:space": "cosine"},
        )
        self.bm25 = None
        self.bm25_ids = []
        self.bm25_docs = []
        self._load_bm25()

    # ---------- registry (books.json) ----------
    def _load_registry(self):
        if Path(REGISTRY_PATH).exists():
            return json.loads(Path(REGISTRY_PATH).read_text())
        return {}

    def _save_registry(self, reg):
        Path(REGISTRY_PATH).write_text(json.dumps(reg, indent=2))

    def has_book(self, book_id):
        return book_id in self._load_registry()

    def list_books(self):
        return list(self._load_registry().values())

    # ---------- add ----------
    def add_book(self, book_meta, chunks, embeddings):
        ids = [f"{book_meta['id']}_c{i}" for i in range(len(chunks))]
        metadatas = [
            {"book_id": book_meta["id"], "book": book_meta["title"], "page": c["page"]}
            for c in chunks
        ]
        documents = [c["text"] for c in chunks]

        self.collection.add(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

        # update registry
        reg = self._load_registry()
        reg[book_meta["id"]] = book_meta
        self._save_registry(reg)

        # rebuild BM25
        self._rebuild_bm25()

    # ---------- BM25 ----------
    def _rebuild_bm25(self):
        data = self.collection.get(include=["documents"])
        self.bm25_ids = data["ids"]
        self.bm25_docs = data["documents"]
        if self.bm25_docs:
            tokenized = [d.lower().split() for d in self.bm25_docs]
            self.bm25 = BM25Okapi(tokenized)
            with open(BM25_PATH, "wb") as f:
                pickle.dump({"ids": self.bm25_ids, "docs": self.bm25_docs}, f)

    def _load_bm25(self):
        if Path(BM25_PATH).exists():
            with open(BM25_PATH, "rb") as f:
                data = pickle.load(f)
            self.bm25_ids = data["ids"]
            self.bm25_docs = data["docs"]
            if self.bm25_docs:
                tokenized = [d.lower().split() for d in self.bm25_docs]
                self.bm25 = BM25Okapi(tokenized)

    # ---------- search ----------
    def vector_search(self, query_embedding, top_k):
        res = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            out.append(
                {
                    "id": res["ids"][0][len(out)],
                    "text": doc,
                    "book": meta["book"],
                    "page": meta["page"],
                    "score": 1 - dist,
                }
            )
        return out

    def bm25_search(self, query, top_k):
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        out = []
        for i in ranked:
            doc_id = self.bm25_ids[i]
            meta = self.collection.get(ids=[doc_id], include=["metadatas", "documents"])
            out.append(
                {
                    "id": doc_id,
                    "text": meta["documents"][0],
                    "book": meta["metadatas"][0]["book"],
                    "page": meta["metadatas"][0]["page"],
                    "score": float(scores[i]),
                }
            )
        return out

    def count(self):
        return self.collection.count()