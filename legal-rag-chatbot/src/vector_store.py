# src/vector_store.py
# Creates embeddings with SentenceTransformers and stores/searches them in FAISS
# Supports save/load so the index persists between app restarts.

import os
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Default paths for persisting the index to disk
DEFAULT_INDEX_PATH  = "data/faiss.index"
DEFAULT_CHUNKS_PATH = "data/chunks.pkl"
DEFAULT_META_PATH   = "data/index_meta.json"


class VectorStore:
    """
    Manages the FAISS index and chunk metadata for semantic search.
    Supports building, searching, saving, and loading the index.
    """

    def __init__(self):
        print("🔍 Loading embedding model (downloads once, ~90MB)...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index  = None   # FAISS index
        self.chunks = []     # Original chunk dicts
        self.meta   = {}     # Info: doc names, chunk count, build timestamp

    # ── Build ────────────────────────────────────────────────────────────
    def build_index(self, chunks: list[dict]) -> None:
        """Encode all chunks and build a FAISS cosine-similarity index."""
        if not chunks:
            raise ValueError("No chunks provided to build the index.")

        self.chunks = chunks
        texts = [c["text"] for c in chunks]

        print("🧠 Generating embeddings... (this may take a minute)")
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)          # normalise → cosine via inner-product

        dimension  = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        import datetime
        doc_names = sorted({c["source"] for c in chunks})
        self.meta = {
            "doc_names":   doc_names,
            "chunk_count": len(chunks),
            "dimension":   dimension,
            "built_at":    datetime.datetime.now().isoformat(timespec="seconds"),
        }
        print(f"✅ FAISS index built — {self.index.ntotal} vectors (dim={dimension})")

    # ── Search ───────────────────────────────────────────────────────────
    def search(self, query: str, top_k: int = 4,
               min_score: float = 0.0) -> list[dict]:
        """
        Return the top_k most relevant chunks for a query.

        Args:
            query:     The user's question.
            top_k:     Maximum number of results.
            min_score: Minimum cosine similarity (0-1). Chunks below this
                       threshold are silently dropped — useful to avoid
                       returning irrelevant content.

        Returns:
            List of chunk dicts, each with an added "score" key.
        """
        if self.index is None:
            raise RuntimeError("Index not built yet. Call build_index() first.")

        query_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)

        scores, indices = self.index.search(query_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if float(score) < min_score:
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)

        return results

    # ── Persist ──────────────────────────────────────────────────────────
    def save(self,
             index_path:  str = DEFAULT_INDEX_PATH,
             chunks_path: str = DEFAULT_CHUNKS_PATH,
             meta_path:   str = DEFAULT_META_PATH) -> None:
        """Save the FAISS index + chunks + metadata to disk."""
        if self.index is None:
            raise RuntimeError("Nothing to save — index is empty.")

        os.makedirs(os.path.dirname(index_path),  exist_ok=True)
        os.makedirs(os.path.dirname(chunks_path), exist_ok=True)

        faiss.write_index(self.index, index_path)
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)
        with open(meta_path, "w") as f:
            json.dump(self.meta, f, indent=2)

        print(f"💾 Index saved → {index_path}")

    def load(self,
             index_path:  str = DEFAULT_INDEX_PATH,
             chunks_path: str = DEFAULT_CHUNKS_PATH,
             meta_path:   str = DEFAULT_META_PATH) -> bool:
        """
        Load a previously saved index from disk.

        Returns:
            True if loaded successfully, False if files don't exist.
        """
        if not (os.path.exists(index_path) and os.path.exists(chunks_path)):
            return False

        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.meta = json.load(f)

        print(f"📂 Index loaded — {self.index.ntotal} vectors")
        return True

    @property
    def is_ready(self) -> bool:
        return self.index is not None and self.index.ntotal > 0
