"""
Alternating Least Squares (ALS) collaborative filtering.

Uses the `implicit` library's GPU-optional ALS implementation, which is the
industry-standard approach for implicit feedback matrix factorisation.

ALS optimises:
    min_{U,V} Σ_{(u,i)} c_ui (r_ui - u_u · v_i)² + λ(‖U‖² + ‖V‖²)

where c_ui = 1 + α * r_ui is a confidence weight (higher for more interactions),
and r_ui is 1 for any observed interaction (implicit feedback).

References:
  - Hu et al. (2008). Collaborative Filtering for Implicit Feedback Datasets.
  - https://github.com/benfred/implicit
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix

from models.base import BaseRecommender


class ALSRecommender(BaseRecommender):
    """
    Matrix factorisation via ALS for implicit feedback.

    After training, user and item latent vectors are stored as:
      self.user_factors: (n_users, factors)
      self.item_factors: (n_items, factors)

    Recommendations are computed as inner products u · V, then ranked.
    """

    def __init__(
        self,
        factors: int = 128,
        iterations: int = 30,
        regularization: float = 0.01,
        alpha: float = 40.0,
        random_state: int = 42,
    ):
        """
        Args:
            factors:        Dimensionality of latent user/item embeddings.
            iterations:     Number of ALS alternation steps.
            regularization: L2 regularisation coefficient (λ).
            alpha:          Confidence scaling factor for implicit feedback.
            random_state:   Reproducibility seed.
        """
        super().__init__(name="ALS")
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.alpha = alpha
        self.random_state = random_state

        self._model = None
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None

    def fit(self, train_matrix: csr_matrix, **kwargs) -> "ALSRecommender":
        """
        Train the ALS model.

        Args:
            train_matrix: User × item sparse CSR matrix of implicit interactions.
                          Values should be raw interaction counts or 1s.
        """
        try:
            import implicit
        except ImportError:
            raise ImportError("Please install `implicit`: pip install implicit")

        # implicit expects items × users
        item_user_matrix = (train_matrix.T * self.alpha).tocsr()

        self._model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.regularization,
            random_state=self.random_state,
            use_gpu=False,
        )

        print(f"[{self.name}] Training ALS (factors={self.factors}, "
              f"iterations={self.iterations})...")
        self._model.fit(item_user_matrix)

        # implicit is fit on (n_items, n_users) matrix, so its naming is inverted:
        #   _model.user_factors → shape (n_users, factors) — axis-0 of transpose
        #   _model.item_factors → shape (n_items, factors)
        # We passed item_user (items × users), so implicit's "users" = our items.
        self.user_factors = self._model.item_factors   # (n_users, factors)
        self.item_factors = self._model.user_factors   # (n_items, factors)
        self.train_matrix = train_matrix
        self.is_fitted = True
        print(f"[{self.name}] Training complete.")
        return self

    def recommend(
        self,
        user_idx: int,
        top_k: int = 10,
        exclude_seen: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        assert self.is_fitted

        # Score all items: (n_items,)
        scores = self.item_factors @ self.user_factors[user_idx]

        if exclude_seen is not None and len(exclude_seen) > 0:
            scores[exclude_seen] = -np.inf

        return np.argsort(-scores)[:top_k]

    def recommend_batch(
        self,
        user_indices: np.ndarray,
        top_k: int = 10,
        train_matrix: Optional[csr_matrix] = None,
    ) -> np.ndarray:
        """
        Vectorised batch recommendation.

        Computes U × Vᵀ in a single matrix multiply — fast for large batches.
        """
        assert self.is_fitted

        # (batch, factors) @ (factors, n_items) → (batch, n_items)
        scores = self.user_factors[user_indices] @ self.item_factors.T

        # Mask already-seen items
        if train_matrix is not None:
            for i, u in enumerate(user_indices):
                seen = train_matrix[u].indices
                if len(seen) > 0:
                    scores[i, seen] = -np.inf

        # Top-k per user
        top_k_indices = np.argsort(-scores, axis=1)[:, :top_k]
        return top_k_indices

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "user_factors": self.user_factors,
                "item_factors": self.item_factors,
                "factors": self.factors,
            }, f)

    def load(self, path: str) -> "ALSRecommender":
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.user_factors = data["user_factors"]
        self.item_factors = data["item_factors"]
        self.factors = data["factors"]
        self.is_fitted = True
        return self
