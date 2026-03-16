"""
Popularity-based recommender (baseline).

Recommends the globally most popular items that the user hasn't already seen.
Despite its simplicity this baseline is surprisingly hard to beat in cold-start
scenarios — a useful sanity check for all other models.
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from models.base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """
    Ranks items by total interaction count in the training set.

    No personalisation: every user receives the same ranked list minus
    any items they've already watched.
    """

    def __init__(self):
        super().__init__(name="Popularity")
        self.popular_items: Optional[np.ndarray] = None  # sorted by popularity

    def fit(self, train: pd.DataFrame, n_items: int, **kwargs) -> "PopularityRecommender":
        """
        Count interactions per item and sort descending.

        Args:
            train:   Training DataFrame with column 'item_idx'.
            n_items: Total number of unique items.
        """
        counts = np.zeros(n_items, dtype=np.int64)
        for item_idx, count in train["item_idx"].value_counts().items():
            counts[item_idx] = count

        self.popular_items = np.argsort(-counts)   # descending
        self.is_fitted = True
        print(f"[{self.name}] Fitted. Top-5 popular items: {self.popular_items[:5]}")
        return self

    def recommend(
        self,
        user_idx: int,
        top_k: int = 10,
        exclude_seen: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        assert self.is_fitted, "Model must be fitted before calling recommend()."
        exclude_set = set(exclude_seen.tolist()) if exclude_seen is not None else set()
        recs = [it for it in self.popular_items if it not in exclude_set]
        return np.array(recs[:top_k])

    def recommend_batch(
        self,
        user_indices: np.ndarray,
        top_k: int = 10,
        train_matrix: Optional[csr_matrix] = None,
    ) -> np.ndarray:
        results = []
        for u in user_indices:
            if train_matrix is not None:
                seen = train_matrix[u].indices
            else:
                seen = np.array([], dtype=np.int64)
            results.append(self.recommend(u, top_k, seen))
        return np.array(results)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"popular_items": self.popular_items}, f)

    def load(self, path: str) -> "PopularityRecommender":
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.popular_items = data["popular_items"]
        self.is_fitted = True
        return self
