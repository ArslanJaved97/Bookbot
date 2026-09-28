from pathlib import Path

from config import CHUNK_SIZE, CHUNK_OVERLAP
from embeddings import Embedder
from ingest import extract_pages, chunk_pages
from retriever import HybridRetriever
from store import BookStore


class BookBot:
    def __init__(self):
        self.store = BookStore()
        self.embedder = Embedder()
        self.retriever = HybridRetriever(self.store, self.embedder)

    def add_pdf(self, pdf_path, title=None):
        pdf_path = Path(pdf_path)
        book_id = pdf_path.stem.lower().replace(" ", "_")

        if self.store.has_book(book_id):
            print(f"[skip] {pdf_path.name} already ingested")
            return

        pages = extract_pages(pdf_path)
        chunks = chunk_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            print(f"[warn] no text extracted from {pdf_path.name}")
            return

        print(f"[embed] embedding {len(chunks)} chunks...")
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed(texts, show_progress=True)

        meta = {
            "id": book_id,
            "title": title or pdf_path.stem,
            "path": str(pdf_path),
            "pages": len(pages),
            "chunks": len(chunks),
        }
        self.store.add_book(meta, chunks, embeddings)
        print(f"[ok] {meta['title']} — {len(chunks)} chunks / {len(pages)} pages")

    def ask(self, query, top_k=5, book_id=None):
        results = self.retriever.search(query, top_k=top_k, book_id=book_id)
        if not results:
            return "No matching content found in the library."

        lines = [f"Q: {query}\n"]
        for i, r in enumerate(results, 1):
            snippet = r["text"].strip().replace("\n", " ")
            if len(snippet) > 1200:
                snippet = snippet[:1200].rsplit(" ", 1)[0] + "..."
            score = r.get("rerank_score", r.get("rrf_score", 0.0))
            lines.append(f"[{i}] {r['book']} — page {r['page']}  (score {score:.3f})")
            lines.append(f"    {snippet}\n")
        return "\n".join(lines)