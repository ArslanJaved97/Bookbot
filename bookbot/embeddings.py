
import os
os.environ["HF_HOME"] = r"D:\hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


os.environ["HF_TOKEN"] = "hf_dummy"
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL


class Embedder:
    def __init__(self, model_name=EMBEDDING_MODEL):
        print(f"[embed] loading {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("[embed] model ready")

    def embed(self, texts, show_progress=False):
        return self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        ).tolist()

    def embed_one(self, text):
        return self.embed([text])[0]