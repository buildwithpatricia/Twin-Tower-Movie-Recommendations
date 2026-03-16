"""Abstract base class shared by all recommender models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import numpy as np


class BaseRecommender(ABC):
    """
    Common interface for all recommenders.

    Every model must implement:
      - fit(train_data, ...)
      - recommend(user_idx, top_k, exclude_seen)
      - save(path) / load(path)
    """

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, **kwargs) -> "BaseRecommender":
        """Train the model on the provided data."""

    @abstractmethod
    def recommend(
        self,
        user_idx: int,
        top_k: int = 10,
        exclude_seen: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Return the top-k item indices recommended for a user.

        Args:
            user_idx:      0-indexed user identifier.
            top_k:         Number of recommendations to return.
            exclude_seen:  Array of item indices to exclude (already seen items).

        Returns:
            Array of item indices, shape (top_k,), sorted by descending score.
        """

    @abstractmethod
    def recommend_batch(
        self,
        user_indices: np.ndarray,
        top_k: int = 10,
        train_matrix=None,
    ) -> np.ndarray:
        """
        Generate recommendations for a batch of users.

        Args:
            user_indices: Array of user indices.
            top_k:        Number of recommendations per user.
            train_matrix: CSR matrix of training interactions (for masking).

        Returns:
            Array of shape (n_users, top_k) with item indices.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist model to disk."""

    @abstractmethod
    def load(self, path: str) -> "BaseRecommender":
        """Load model from disk."""

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.__class__.__name__}(name={self.name!r}, status={status})"
