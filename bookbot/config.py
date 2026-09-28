from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "bookbot_data"
CHROMA_PATH = str(DATA_DIR / "chroma")
BM25_PATH = DATA_DIR / "bm25.pkl"
REGISTRY_PATH = DATA_DIR / "books.json"

DATA_DIR.mkdir(exist_ok=True)

# Embedding model — runs locally, no API
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking
CHUNK_SIZE = 220        # words per chunk
CHUNK_OVERLAP = 40      # words overlap between chunks

# Retrieval
DEFAULT_TOP_K = 5
RRF_K = 60              # reciprocal rank fusion constant
HYBRID_ALPHA = 0.5      # 0 = pure BM25, 1 = pure vector