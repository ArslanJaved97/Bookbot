# BookBot

Offline, unfiltered, semantic PDF search. Ask questions across all your books at once. No cloud, no LLM, no content filters.

## What it does

- Ingest PDFs (one or many)
- Search across all books simultaneously with hybrid BM25 + vector retrieval
- Re-rank results with a cross-encoder for precision
- Return verbatim passages with page citations
- Never refuses a query — there is no generative model in the loop

## Stack

- **PyMuPDF** — PDF text extraction
- **sentence-transformers** — local embeddings (`all-MiniLM-L6-v2`)
- **ChromaDB** — persistent vector store
- **rank-bm25** — keyword search
- **cross-encoder/ms-marco-MiniLM-L-6-v2** — re-ranking
- **Streamlit** — web UI

## Install

```bash
conda create -n bookbot python=3.11 -y
conda activate bookbot
pip install -r requirements.txt