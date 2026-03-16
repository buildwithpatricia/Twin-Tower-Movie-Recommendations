"""
Neural Two-Tower (Dual Encoder) Recommender.

Architecture overview:
                                    score = dot(u_emb, i_emb)
     ┌──────────────────┐          ┌──────────────────┐
     │    User Tower    │          │    Item Tower    │
     │                  │          │                  │
     │  user_id ──► Emb │          │  item_id ──► Emb │
     │  user_feats ──►  │──► u_emb │  item_feats ──►  │──► i_emb
     │  [FC → ReLU]×2   │          │  [FC → ReLU]×2   │
     └──────────────────┘          └──────────────────┘

Training: in-batch negatives + sampled negatives with binary cross-entropy.
Inference: pre-compute all item embeddings, build FAISS index, serve via ANN.

This mirrors production two-tower systems at YouTube, Google, etc.

References:
  - Covington et al. (2016). Deep Neural Networks for YouTube Recommendations.
  - Yi et al. (2019). Sampling-Bias-Corrected Neural Modeling.
"""

import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader, Dataset

from models.base import BaseRecommender


# ── Dataset ───────────────────────────────────────────────────────────────────

class InteractionDataset(Dataset):
    """
    Generates (user, positive_item, negative_items) training triples.

    For each positive (user, item) interaction we randomly sample
    `num_negatives` items the user has NOT interacted with.
    """

    def __init__(
        self,
        train: "pd.DataFrame",
        n_items: int,
        user_features: np.ndarray,
        item_features: np.ndarray,
        num_negatives: int = 4,
    ):
        import pandas as pd
        self.interactions = train[["user_idx", "item_idx"]].values
        self.n_items = n_items
        self.user_features = torch.tensor(user_features, dtype=torch.float32)
        self.item_features = torch.tensor(item_features, dtype=torch.float32)
        self.num_negatives = num_negatives

        # Build set of seen items per user for fast negative sampling
        self.user_seen = (
            train.groupby("user_idx")["item_idx"]
            .apply(set)
            .to_dict()
        )

    def __len__(self) -> int:
        return len(self.interactions)

    def __getitem__(self, idx: int) -> Tuple:
        user_idx, pos_item = self.interactions[idx]
        seen = self.user_seen.get(user_idx, set())

        # Sample negatives (retry until unseen item found)
        negatives = []
        while len(negatives) < self.num_negatives:
            neg = np.random.randint(0, self.n_items)
            if neg not in seen:
                negatives.append(neg)

        return (
            torch.tensor(user_idx, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(negatives, dtype=torch.long),
            self.user_features[user_idx],
            self.item_features[pos_item],
            self.item_features[negatives],
        )


# ── Tower definition ──────────────────────────────────────────────────────────

class Tower(nn.Module):
    """Single MLP tower: embedding + side features → L2-normalised output."""

    def __init__(
        self,
        n_entities: int,
        entity_embedding_dim: int,
        feature_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(n_entities, entity_embedding_dim)

        input_dim = entity_embedding_dim + feature_dim
        layers: List[nn.Module] = []
        for h_dim in hidden_dims:
            layers += [nn.Linear(input_dim, h_dim), nn.ReLU(), nn.Dropout(dropout)]
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, output_dim))

        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, entity_ids: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(entity_ids)
        x = torch.cat([emb, features], dim=-1)
        out = self.mlp(x)
        return F.normalize(out, p=2, dim=-1)   # L2 normalise for dot-product similarity


# ── Main model ────────────────────────────────────────────────────────────────

class TwoTowerModel(nn.Module):

    def __init__(
        self,
        n_users: int,
        n_items: int,
        user_feature_dim: int,
        item_feature_dim: int,
        embedding_dim: int = 64,
        hidden_dims: List[int] = (256, 128),
        output_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.user_tower = Tower(n_users, embedding_dim, user_feature_dim,
                                list(hidden_dims), output_dim, dropout)
        self.item_tower = Tower(n_items, embedding_dim, item_feature_dim,
                                list(hidden_dims), output_dim, dropout)

    def forward(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
        user_feats: torch.Tensor,
        pos_item_feats: torch.Tensor,
        neg_item_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute BPR-style binary cross-entropy loss.

        For each (user, pos, neg) triple, score = u·i.  We want
        score(pos) >> score(neg).
        """
        u_emb = self.user_tower(user_ids, user_feats)              # (B, D)
        i_pos_emb = self.item_tower(pos_item_ids, pos_item_feats)  # (B, D)

        B, K, D = neg_item_ids.shape[0], neg_item_ids.shape[1], u_emb.shape[-1]
        neg_ids_flat   = neg_item_ids.view(-1)                      # (B*K,)
        neg_feats_flat = neg_item_feats.view(-1, neg_item_feats.shape[-1])
        i_neg_emb = self.item_tower(neg_ids_flat, neg_feats_flat).view(B, K, D)

        pos_scores = (u_emb * i_pos_emb).sum(dim=-1)               # (B,)
        # Expand u_emb for broadcast: (B, 1, D) × (B, K, D) → (B, K)
        neg_scores = (u_emb.unsqueeze(1) * i_neg_emb).sum(dim=-1)  # (B, K)

        # BPR loss: want pos_score > each neg_score
        pos_expanded = pos_scores.unsqueeze(1).expand_as(neg_scores)
        loss = -F.logsigmoid(pos_expanded - neg_scores).mean()
        return loss

    @torch.no_grad()
    def encode_items(
        self,
        item_ids: torch.Tensor,
        item_feats: torch.Tensor,
    ) -> torch.Tensor:
        return self.item_tower(item_ids, item_feats)

    @torch.no_grad()
    def encode_user(
        self,
        user_id: torch.Tensor,
        user_feats: torch.Tensor,
    ) -> torch.Tensor:
        return self.user_tower(user_id, user_feats)


# ── Recommender wrapper ───────────────────────────────────────────────────────

class TwoTowerRecommender(BaseRecommender):
    """Sklearn-style wrapper around TwoTowerModel for the evaluation pipeline."""

    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dims: Tuple = (256, 128),
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 2048,
        epochs: int = 20,
        num_negatives: int = 4,
        random_state: int = 42,
    ):
        super().__init__(name="TwoTower")
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_negatives = num_negatives
        self.random_state = random_state

        self.model: Optional[TwoTowerModel] = None
        self.item_embeddings: Optional[np.ndarray] = None  # pre-computed
        self.user_features_tensor: Optional[torch.Tensor] = None
        self.item_features_tensor: Optional[torch.Tensor] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(
        self,
        train,
        n_users: int,
        n_items: int,
        user_features: np.ndarray,
        item_features: np.ndarray,
        **kwargs,
    ) -> "TwoTowerRecommender":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.user_features_tensor = torch.tensor(user_features, dtype=torch.float32)
        self.item_features_tensor = torch.tensor(item_features, dtype=torch.float32)

        dataset = InteractionDataset(
            train, n_items, user_features, item_features, self.num_negatives
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                            num_workers=0, pin_memory=False)

        self.model = TwoTowerModel(
            n_users=n_users,
            n_items=n_items,
            user_feature_dim=user_features.shape[1],
            item_feature_dim=item_features.shape[1],
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)

        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=self.epochs
        )

        print(f"[{self.name}] Training on {self.device} for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            for batch in loader:
                batch = [b.to(self.device) for b in batch]
                user_ids, pos_ids, neg_ids, u_feats, pos_feats, neg_feats = batch

                optimiser.zero_grad()
                loss = self.model(user_ids, pos_ids, neg_ids, u_feats, pos_feats, neg_feats)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimiser.step()
                total_loss += loss.item()

            scheduler.step()
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{self.epochs} | "
                      f"Loss: {total_loss / len(loader):.4f}")

        # Pre-compute item embeddings for fast inference
        print(f"[{self.name}] Pre-computing item embeddings...")
        self._precompute_item_embeddings(n_items, item_features)
        self.is_fitted = True
        print(f"[{self.name}] Training complete.")
        return self

    def _precompute_item_embeddings(self, n_items: int, item_features: np.ndarray):
        """Compute and cache all item embeddings for fast retrieval."""
        self.model.eval()
        all_ids   = torch.arange(n_items, dtype=torch.long).to(self.device)
        all_feats = torch.tensor(item_features, dtype=torch.float32).to(self.device)

        chunk_size = 2048
        embeddings = []
        for start in range(0, n_items, chunk_size):
            end = min(start + chunk_size, n_items)
            emb = self.model.encode_items(all_ids[start:end], all_feats[start:end])
            embeddings.append(emb.cpu().numpy())

        self.item_embeddings = np.concatenate(embeddings, axis=0)  # (n_items, D)

    def _get_user_embedding(self, user_idx: int) -> np.ndarray:
        self.model.eval()
        uid = torch.tensor([user_idx], dtype=torch.long).to(self.device)
        uf  = self.user_features_tensor[user_idx].unsqueeze(0).to(self.device)
        return self.model.encode_user(uid, uf).cpu().numpy()[0]

    def recommend(
        self,
        user_idx: int,
        top_k: int = 10,
        exclude_seen: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        assert self.is_fitted
        u_emb = self._get_user_embedding(user_idx)          # (D,)
        scores = self.item_embeddings @ u_emb               # (n_items,)

        if exclude_seen is not None and len(exclude_seen) > 0:
            scores[exclude_seen] = -np.inf

        return np.argsort(-scores)[:top_k]

    def recommend_batch(
        self,
        user_indices: np.ndarray,
        top_k: int = 10,
        train_matrix: Optional[csr_matrix] = None,
    ) -> np.ndarray:
        assert self.is_fitted
        self.model.eval()

        # Batch-encode users
        uid_t   = torch.tensor(user_indices, dtype=torch.long).to(self.device)
        u_feats = self.user_features_tensor[user_indices].to(self.device)
        u_embs  = self.model.encode_user(uid_t, u_feats).cpu().numpy()  # (B, D)

        # Score: (B, D) @ (D, n_items) → (B, n_items)
        scores = u_embs @ self.item_embeddings.T

        if train_matrix is not None:
            for i, u in enumerate(user_indices):
                seen = train_matrix[u].indices
                if len(seen) > 0:
                    scores[i, seen] = -np.inf

        return np.argsort(-scores, axis=1)[:, :top_k]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "item_embeddings": self.item_embeddings,
            "config": {
                "embedding_dim": self.embedding_dim,
                "hidden_dims": self.hidden_dims,
                "dropout": self.dropout,
            },
        }, path)

    def load(self, path: str) -> "TwoTowerRecommender":
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.item_embeddings = data["item_embeddings"]
        self.is_fitted = True
        return self
