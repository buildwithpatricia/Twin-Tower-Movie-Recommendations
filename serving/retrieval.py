"""
FAISS-based item retrieval engine.

In production recommendation systems, generating scores for all items
on every request is too slow (O(n_items) per query).  Instead we use
Approximate Nearest Neighbour (ANN) search:

  1. Offline: encode all items into a dense embedding space → build FAISS index
  2. Online:  encode the query user → search FAISS for top-K similar items in O(log n)

We expose two index types:
  - IndexFlatIP  : exact inner-product search (accurate, slower)
  - IndexIVFFlat : inverted-file approximate search (faster, slight recall loss)

This mirrors the retrieval stage of a two-stage recommendation pipeline:
  [FAISS retrieval] → [lightweight re-ranker] → [final list]
"""

import pickle
from pathlib import Path
from typing import Tuple

import numpy as np


class FAISSRetriever:
    """
    Wraps a FAISS index for fast item retrieval given a user query vector.

    Workflow:
      retriever = FAISSRetriever()
      retriever.build(item_embeddings)    # one-time offline step
      top_ids, scores = retriever.search(user_vector, k=50)
    """

    def __init__(self, index_type: str = "Flat"):
        """
        Args:
            index_type: "Flat" for exact search, "IVF" for approximate.
        """
        self.index_type = index_type
        self.index = None
        self.embedding_dim: int = 0
        self.n_items: int = 0

    def build(self, item_embeddings: np.ndarray) -> "FAISSRetriever":
        """
        Build the FAISS index from item embeddings.

        Args:
            item_embeddings: np.ndarray of shape (n_items, embedding_dim),
                             already L2-normalised (for cosine similarity via IP).
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("Please install faiss-cpu: pip install faiss-cpu")

        n, d = item_embeddings.shape
        self.embedding_dim = d
        self.n_items = n

        embeddings = item_embeddings.astype(np.float32)

        if self.index_type == "Flat":
            # Exact inner-product search (equivalent to cosine sim for L2-normed vecs)
            self.index = faiss.IndexFlatIP(d)
        elif self.index_type == "IVF":
            # Approximate: cluster into n_lists Voronoi cells, search n_probe cells
            n_lists = max(int(np.sqrt(n)), 10)
            quantiser = faiss.IndexFlatIP(d)
            self.index = faiss.IndexIVFFlat(quantiser, d, n_lists, faiss.METRIC_INNER_PRODUCT)
            self.index.train(embeddings)
            self.index.nprobe = max(n_lists // 10, 1)   # probe 10% of cells
        else:
            raise ValueError(f"Unknown index_type: {self.index_type}")

        self.index.add(embeddings)
        print(f"[FAISS] Built {self.index_type} index with {self.index.ntotal:,} items "
              f"(dim={d})")
        return self

    def search(
        self,
        query: np.ndarray,
        k: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve top-k items for a query vector.

        Args:
            query: np.ndarray of shape (embedding_dim,) — the user embedding.
            k:     Number of items to retrieve.

        Returns:
            item_ids: np.ndarray of shape (k,) — item indices
            scores:   np.ndarray of shape (k,) — inner-product scores
        """
        assert self.index is not None, "Call build() before search()."
        query = query.astype(np.float32).reshape(1, -1)
        scores, item_ids = self.index.search(query, k)
        return item_ids[0], scores[0]

    def search_batch(
        self,
        queries: np.ndarray,
        k: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch retrieval for multiple user queries.

        Args:
            queries: np.ndarray of shape (batch_size, embedding_dim).
            k:       Number of items to retrieve per user.

        Returns:
            item_ids: (batch_size, k)
            scores:   (batch_size, k)
        """
        assert self.index is not None
        queries = queries.astype(np.float32)
        scores, item_ids = self.index.search(queries, k)
        return item_ids, scores

    def save(self, path: str) -> None:
        """Save index to disk."""
        import faiss
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        meta_path = path.with_suffix(".meta.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump({
                "index_type": self.index_type,
                "embedding_dim": self.embedding_dim,
                "n_items": self.n_items,
            }, f)
        print(f"[FAISS] Index saved to {path}")

    def load(self, path: str) -> "FAISSRetriever":
        """Load index from disk."""
        import faiss
        self.index = faiss.read_index(str(path))
        meta_path = Path(path).with_suffix(".meta.pkl")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.index_type    = meta["index_type"]
        self.embedding_dim = meta["embedding_dim"]
        self.n_items       = meta["n_items"]
        print(f"[FAISS] Loaded index with {self.index.ntotal:,} items")
        return self
