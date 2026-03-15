"""
SASRec: Self-Attentive Sequential Recommendation.

SASRec uses a causal (left-to-right) Transformer to model sequential user
behaviour.  Given a user's watch history [i1, i2, …, iT], it predicts which
item iT+1 they will interact with next.

Key design choices:
  - Causal attention mask: position t only attends to positions ≤ t
  - Shared item embedding matrix for input and output projection
  - Layer normalisation before attention (Pre-LN for stability)
  - Point-wise feed-forward network with GELU activation

Training:
  For each user sequence [i1, …, iT] we predict all suffix positions using
  binary cross-entropy with one sampled negative per positive.

  Positive at position t: item i_{t+1}
  Negative at position t: randomly sampled item NOT in user's history

References:
  - Wang et al. (2018). Self-Attentive Sequential Recommendation. ICDM 2018.
    https://arxiv.org/abs/1808.09781
"""

import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader, Dataset

from models.base import BaseRecommender


# ── Dataset ───────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    """
    Builds padded interaction sequences for each user.

    For a user with history [i1, i2, i3, i4]:
      - Input  sequence: [PAD, i1, i2, i3]  (shifted right)
      - Target sequence: [i1,  i2, i3, i4]  (next item at each position)
      - Negatives: one random unseen item per target position
    """

    def __init__(
        self,
        train: "pd.DataFrame",
        n_items: int,
        max_seq_len: int = 50,
    ):
        self.n_items = n_items
        self.max_seq_len = max_seq_len
        self.PAD = 0   # item index 0 is reserved for padding

        # Build sorted sequences per user (item indices are 1-indexed here)
        # We shift item_idx by 1 so that 0 can serve as PAD
        user_sequences = {}
        for _, group in train.sort_values("timestamp").groupby("user_idx"):
            seq = (group["item_idx"].values + 1).tolist()  # 1-indexed
            user_sequences[group["user_idx"].iloc[0]] = seq

        self.samples = []  # list of (input_seq, target_seq, neg_seq)
        for user_idx, seq in user_sequences.items():
            if len(seq) < 2:
                continue  # need at least 2 items to form input/target pair
            seen = set(seq)
            self.samples.append((user_idx, seq, seen))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user_idx, seq, seen = self.samples[idx]
        L = self.max_seq_len

        # Truncate to last L items and build shifted input/target
        seq = seq[-L:]
        input_seq  = [self.PAD] + seq[:-1]   # shifted right, PAD prepended
        target_seq = seq

        # Pad to max_seq_len
        pad_len = L - len(input_seq)
        input_seq  = [self.PAD] * pad_len + input_seq
        target_seq = [self.PAD] * pad_len + target_seq

        # Sample one negative per position
        neg_seq = []
        for _ in target_seq:
            neg = np.random.randint(1, self.n_items + 1)  # 1-indexed
            while neg in seen:
                neg = np.random.randint(1, self.n_items + 1)
            neg_seq.append(neg)

        return (
            torch.tensor(input_seq,  dtype=torch.long),
            torch.tensor(target_seq, dtype=torch.long),
            torch.tensor(neg_seq,    dtype=torch.long),
        )


# ── Model components ──────────────────────────────────────────────────────────

class PointWiseFeedForward(nn.Module):
    """Position-wise FFN: two linear layers with GELU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SASRecBlock(nn.Module):
    """
    One SASRec transformer block:
      Pre-LN Multi-Head Self-Attention → residual
      Pre-LN Feed-Forward              → residual
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = PointWiseFeedForward(d_model, d_model * 4, dropout)
        self.drop  = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        # Pre-LN attention with causal mask
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.drop(x) + residual

        # Pre-LN feed-forward
        residual = x
        x = self.norm2(x)
        x = self.ffn(x) + residual
        return x


class SASRecModel(nn.Module):
    """
    Self-Attentive Sequential Recommendation model.

    Item IDs are 1-indexed; 0 is the PAD token.
    """

    def __init__(
        self,
        n_items: int,
        max_seq_len: int = 50,
        embedding_dim: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_items = n_items
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim

        # +1 for PAD token (index 0)
        self.item_embedding = nn.Embedding(n_items + 1, embedding_dim, padding_idx=0)
        self.pos_embedding  = nn.Embedding(max_seq_len, embedding_dim)
        self.emb_dropout    = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            SASRecBlock(embedding_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(embedding_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        for block in self.blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask to prevent attending to future positions."""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()
        return mask

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_seq: (batch, seq_len) item IDs (0 = PAD)

        Returns:
            hidden: (batch, seq_len, embedding_dim) contextual representations
        """
        B, L = input_seq.shape
        positions = torch.arange(L, device=input_seq.device).unsqueeze(0)  # (1, L)

        x = self.item_embedding(input_seq) + self.pos_embedding(positions)
        x = self.emb_dropout(x)

        causal_mask = self._causal_mask(L, input_seq.device)
        for block in self.blocks:
            x = block(x, causal_mask)

        return self.final_norm(x)   # (B, L, D)

    def compute_loss(
        self,
        input_seq: torch.Tensor,
        pos_seq:   torch.Tensor,
        neg_seq:   torch.Tensor,
    ) -> torch.Tensor:
        """
        Binary cross-entropy loss over all non-PAD positions.

        Score for item i at position t = hidden[t] · item_embedding[i]
        """
        hidden = self.forward(input_seq)                         # (B, L, D)
        pos_emb = self.item_embedding(pos_seq)                   # (B, L, D)
        neg_emb = self.item_embedding(neg_seq)                   # (B, L, D)

        pos_scores = (hidden * pos_emb).sum(dim=-1)              # (B, L)
        neg_scores = (hidden * neg_emb).sum(dim=-1)              # (B, L)

        # Mask PAD positions
        mask = (pos_seq != 0).float()
        loss = (-F.logsigmoid(pos_scores) - F.logsigmoid(-neg_scores)) * mask
        return loss.sum() / mask.sum()

    @torch.no_grad()
    def score_all_items(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        Score all items for the last (most recent) position.

        Returns:
            scores: (batch, n_items+1) — item 0 is PAD, ignored in ranking
        """
        hidden = self.forward(input_seq)[:, -1, :]   # (B, D) — use last position
        all_item_embs = self.item_embedding.weight    # (n_items+1, D)
        return hidden @ all_item_embs.T              # (B, n_items+1)


# ── Recommender wrapper ───────────────────────────────────────────────────────

class SASRecRecommender(BaseRecommender):

    def __init__(
        self,
        embedding_dim: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        dropout: float = 0.2,
        max_seq_len: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        epochs: int = 20,
        random_state: int = 42,
    ):
        super().__init__(name="SASRec")
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state

        self.model: Optional[SASRecModel] = None
        self.user_sequences: Optional[dict] = None  # user_idx → list of 0-indexed item IDs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(
        self,
        train: "pd.DataFrame",
        n_items: int,
        **kwargs,
    ) -> "SASRecRecommender":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Store user sequences (0-indexed) for inference
        self.user_sequences = {}
        for user_idx, group in train.sort_values("timestamp").groupby("user_idx"):
            self.user_sequences[user_idx] = group["item_idx"].tolist()

        dataset = SequenceDataset(train, n_items, self.max_seq_len)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                             num_workers=0)

        self.model = SASRecModel(
            n_items=n_items,
            max_seq_len=self.max_seq_len,
            embedding_dim=self.embedding_dim,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
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
            for inp, pos, neg in loader:
                inp, pos, neg = inp.to(self.device), pos.to(self.device), neg.to(self.device)
                optimiser.zero_grad()
                loss = self.model.compute_loss(inp, pos, neg)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimiser.step()
                total_loss += loss.item()

            scheduler.step()
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{self.epochs} | "
                      f"Loss: {total_loss / len(loader):.4f}")

        self.n_items = n_items
        self.is_fitted = True
        print(f"[{self.name}] Training complete.")
        return self

    def _build_input_seq(self, user_idx: int) -> torch.Tensor:
        """Build padded input sequence for a user (1-indexed items, 0=PAD)."""
        seq = self.user_sequences.get(user_idx, [])
        # Convert to 1-indexed, take last max_seq_len items
        seq_1idx = [s + 1 for s in seq][-self.max_seq_len:]
        pad_len  = self.max_seq_len - len(seq_1idx)
        padded   = [0] * pad_len + seq_1idx
        return torch.tensor([padded], dtype=torch.long).to(self.device)

    def recommend(
        self,
        user_idx: int,
        top_k: int = 10,
        exclude_seen: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        assert self.is_fitted
        self.model.eval()

        input_seq = self._build_input_seq(user_idx)              # (1, L)
        scores    = self.model.score_all_items(input_seq)[0]     # (n_items+1,)
        scores    = scores[1:].cpu().numpy()                     # drop PAD → (n_items,)

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

        # Build batch of input sequences
        batch_seqs = []
        for u in user_indices:
            seq = self.user_sequences.get(u, [])
            seq_1idx = [s + 1 for s in seq][-self.max_seq_len:]
            pad_len  = self.max_seq_len - len(seq_1idx)
            padded   = [0] * pad_len + seq_1idx
            batch_seqs.append(padded)

        batch_tensor = torch.tensor(batch_seqs, dtype=torch.long).to(self.device)

        with torch.no_grad():
            scores = self.model.score_all_items(batch_tensor)    # (B, n_items+1)
            scores = scores[:, 1:].cpu().numpy()                 # drop PAD → (B, n_items)

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
            "user_sequences": self.user_sequences,
            "n_items": self.n_items,
            "config": {
                "embedding_dim": self.embedding_dim,
                "n_heads": self.n_heads,
                "n_layers": self.n_layers,
                "dropout": self.dropout,
                "max_seq_len": self.max_seq_len,
            },
        }, path)

    def load(self, path: str) -> "SASRecRecommender":
        data = torch.load(path, map_location=self.device, weights_only=False)
        cfg  = data["config"]
        self.n_items       = data["n_items"]
        self.user_sequences = data["user_sequences"]
        self.model = SASRecModel(
            n_items=self.n_items, **cfg
        ).to(self.device)
        self.model.load_state_dict(data["model_state"])
        self.model.eval()
        self.is_fitted = True
        return self
