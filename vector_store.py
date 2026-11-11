"""FAISS vector store with local sentence-transformer embeddings and disk persistence."""
import json
import shutil
from dataclasses import asdict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import config
from document_loader import Chunk

INDEX_FILE = config.INDEX_DIR / "index.faiss"
META_FILE = config.INDEX_DIR / "chunks.json"


class VectorStore:
    def __init__(self, model_name: str = config.EMBED_MODEL):
        self.model = SentenceTransformer(model_name)
        self.index: faiss.Index | None = None
        self.chunks: list[dict] = []

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        return vecs.astype("float32")

    def sources(self) -> list[str]:
        return sorted({c["source"] for c in self.chunks})

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vecs = self._embed([c.text for c in chunks])
        if self.index is None:
            self.index = faiss.IndexFlatIP(vecs.shape[1])  # cosine similarity (vectors are normalised)
        self.index.add(vecs)
        self.chunks.extend(asdict(c) for c in chunks)

    def search(self, query: str, k: int = config.TOP_K) -> list[dict]:
        if self.index is None or self.index.ntotal == 0:
            return []
        scores, ids = self.index.search(self._embed([query]), min(k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            item = dict(self.chunks[idx])
            item["score"] = float(score)
            results.append(item)
        return results

    def save(self) -> None:
        config.INDEX_DIR.mkdir(exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(INDEX_FILE))
            META_FILE.write_text(json.dumps(self.chunks), encoding="utf-8")

    @classmethod
    def load_or_new(cls) -> "VectorStore":
        store = cls()
        if INDEX_FILE.exists() and META_FILE.exists():
            store.index = faiss.read_index(str(INDEX_FILE))
            store.chunks = json.loads(META_FILE.read_text(encoding="utf-8"))
        return store

    def reset(self) -> None:
        self.index = None
        self.chunks = []
        if config.INDEX_DIR.exists():
            shutil.rmtree(config.INDEX_DIR)
