"""
FastAPI recommendation service.

Endpoints:
  GET  /health                   — liveness check
  GET  /recommend/{user_id}      — get top-K recommendations for a user
  POST /recommend/batch          — batch recommendations for multiple users
  GET  /item/{item_id}           — item metadata lookup
  GET  /similar/{item_id}        — content-similar items (cold-start fallback)

Architecture:
  Request → [Two-Tower user encoder] → user embedding
           → [FAISS ANN retrieval] → top-50 candidate items
           → [Diversity re-ranker] → final top-K

The re-ranking step applies Maximal Marginal Relevance (MMR) to balance
relevance (FAISS score) with diversity (genre distance), mirroring how
streaming platforms avoid recommending 10 identical-genre titles.

Run with:
  uvicorn serving.api:app --reload --host 0.0.0.0 --port 8000
"""

import os
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── Lazy-loaded global state (avoids loading at import time for tests) ─────────
_models = {}


def get_state():
    """Load models into memory on first request (lazy loading)."""
    global _models
    if _models:
        return _models

    model_dir = Path(os.getenv("MODEL_DIR", "results/models"))
    data_dir  = Path(os.getenv("DATA_DIR",  "data/processed"))

    if not (model_dir / "two_tower.pt").exists():
        raise RuntimeError(
            f"No trained models found at {model_dir}. "
            "Run `python scripts/train_all.py` first."
        )

    # Load Two-Tower model
    from models.two_tower import TwoTowerRecommender
    two_tower = TwoTowerRecommender()

    # We need to load user/item features to rebuild the model architecture
    with open(data_dir / "metadata.pkl", "rb") as f:
        meta = pickle.load(f)
    with open(data_dir / "id_maps.pkl", "rb") as f:
        id_maps = pickle.load(f)

    import pandas as pd
    import torch
    from features.item_features import build_item_features
    from features.user_features import build_user_features

    movies = pd.read_parquet(data_dir / "movies.parquet")
    users  = pd.read_parquet(data_dir / "users.parquet")
    train  = pd.read_parquet(data_dir / "train.parquet")

    from features.item_features import build_genre_matrix
    item_genre_matrix = build_genre_matrix(movies)
    item_features = build_item_features(movies, train, meta["n_items"])
    user_features = build_user_features(users, train, item_genre_matrix, meta["n_users"])

    two_tower.user_features_tensor = torch.tensor(user_features, dtype=torch.float32)
    two_tower.item_features_tensor = torch.tensor(item_features, dtype=torch.float32)

    data = torch.load(model_dir / "two_tower.pt", map_location="cpu", weights_only=False)
    from models.two_tower import TwoTowerModel
    two_tower.model = TwoTowerModel(
        n_users=meta["n_users"],
        n_items=meta["n_items"],
        user_feature_dim=user_features.shape[1],
        item_feature_dim=item_features.shape[1],
        embedding_dim=64,
        hidden_dims=[256, 128],
    )
    two_tower.model.load_state_dict(data["model_state"])
    two_tower.model.eval()
    two_tower.item_embeddings = data["item_embeddings"]
    two_tower.is_fitted = True

    # Load FAISS index
    from serving.retrieval import FAISSRetriever
    faiss_retriever = FAISSRetriever()
    faiss_retriever.load(str(model_dir / "faiss_index.bin"))

    # Load item metadata for display
    item_meta = {}
    for _, row in movies.iterrows():
        item_meta[int(row["item_idx"])] = {
            "title": row["title"],
            "genres": row["genres"],
        }

    _models = {
        "two_tower": two_tower,
        "faiss": faiss_retriever,
        "item_meta": item_meta,
        "idx2item": id_maps["idx2item"],
        "metadata": meta,
        "item_genre_matrix": item_genre_matrix,
    }
    return _models


# ── Maximal Marginal Relevance re-ranker ──────────────────────────────────────

def mmr_rerank(
    candidate_ids: np.ndarray,
    candidate_scores: np.ndarray,
    item_genre_matrix: np.ndarray,
    top_k: int = 10,
    lambda_diversity: float = 0.3,
) -> np.ndarray:
    """
    Maximal Marginal Relevance re-ranking.

    Balances relevance (retrieval score) with diversity (genre distance).
    At each step, selects the item that maximises:
        MMR = λ * relevance - (1-λ) * max_similarity_to_selected

    Args:
        candidate_ids:    Item indices from FAISS retrieval.
        candidate_scores: Retrieval scores (higher = more relevant).
        item_genre_matrix: (n_items, n_genres) for diversity computation.
        top_k:            Final list length.
        lambda_diversity: Trade-off. 1.0 = pure relevance, 0.0 = pure diversity.

    Returns:
        Array of top_k item indices after re-ranking.
    """
    if len(candidate_ids) <= top_k:
        return candidate_ids[:top_k]

    # L2-normalise candidate genre vectors for cosine similarity
    genre_vecs = item_genre_matrix[candidate_ids].astype(np.float32)
    norms = np.linalg.norm(genre_vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    genre_vecs_normed = genre_vecs / norms

    # Normalise scores to [0, 1]
    scores = (candidate_scores - candidate_scores.min()) / (
        candidate_scores.max() - candidate_scores.min() + 1e-8
    )

    selected_indices = []   # indices into candidate_ids array
    remaining = list(range(len(candidate_ids)))

    for _ in range(top_k):
        if not remaining:
            break

        if not selected_indices:
            # First pick: highest relevance
            best = max(remaining, key=lambda i: scores[i])
        else:
            # MMR score for each remaining candidate
            selected_vecs = genre_vecs_normed[selected_indices]  # (n_sel, n_genres)
            best_score = -np.inf
            best = remaining[0]
            for i in remaining:
                rel = scores[i]
                # Max cosine similarity to already-selected items
                sims = genre_vecs_normed[i] @ selected_vecs.T   # (n_sel,)
                max_sim = float(sims.max()) if len(sims) > 0 else 0.0
                mmr_score = lambda_diversity * rel - (1 - lambda_diversity) * max_sim
                if mmr_score > best_score:
                    best_score = mmr_score
                    best = i

        selected_indices.append(best)
        remaining.remove(best)

    return candidate_ids[selected_indices]


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Tubi RecSys API",
    description=(
        "Streaming content recommendation API. "
        "Two-Tower neural retrieval + MMR diversity re-ranking."
    ),
    version="1.0.0",
)


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[dict]
    model: str = "TwoTower+MMR"


class BatchRequest(BaseModel):
    user_ids: List[int]
    top_k: int = 10


@app.get("/health")
def health():
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def recommend(
    user_id: int,
    top_k: int = 10,
    diversity: float = 0.3,
):
    """
    Get top-K recommendations for a user.

    - **user_id**: Internal 0-indexed user identifier
    - **top_k**: Number of recommendations to return (default 10)
    - **diversity**: MMR lambda (0=max diversity, 1=max relevance; default 0.3)
    """
    state = get_state()
    meta  = state["metadata"]

    if user_id < 0 or user_id >= meta["n_users"]:
        raise HTTPException(status_code=404, detail=f"user_id {user_id} not found.")

    # Get user embedding from Two-Tower
    user_emb = state["two_tower"]._get_user_embedding(user_id)   # (D,)

    # FAISS retrieval: get top-50 candidates
    cand_ids, cand_scores = state["faiss"].search(user_emb, k=min(50, meta["n_items"]))

    # MMR re-ranking for diversity
    final_ids = mmr_rerank(
        cand_ids, cand_scores, state["item_genre_matrix"],
        top_k=top_k, lambda_diversity=diversity
    )

    # Build response with item metadata
    recs = []
    for rank, item_idx in enumerate(final_ids, start=1):
        item_info = state["item_meta"].get(int(item_idx), {})
        recs.append({
            "rank":     rank,
            "item_idx": int(item_idx),
            "title":    item_info.get("title", "Unknown"),
            "genres":   item_info.get("genres", ""),
        })

    return RecommendationResponse(user_id=user_id, recommendations=recs)


@app.post("/recommend/batch")
def recommend_batch(request: BatchRequest):
    """Batch recommendations for multiple users."""
    state = get_state()
    results = []
    for uid in request.user_ids:
        try:
            result = recommend(uid, top_k=request.top_k)
            results.append(result)
        except HTTPException:
            results.append({"user_id": uid, "error": "User not found"})
    return {"results": results}


@app.get("/similar/{item_id}")
def similar_items(item_id: int, top_k: int = 10):
    """
    Find content-similar items using genre-based cosine similarity.

    Useful as a cold-start fallback: when a user has no history,
    return items similar to whatever they're currently viewing.
    """
    state = get_state()
    meta  = state["metadata"]

    if item_id < 0 or item_id >= meta["n_items"]:
        raise HTTPException(status_code=404, detail=f"item_id {item_id} not found.")

    # Use FAISS on item embeddings (item-to-item similarity)
    item_emb = state["two_tower"].item_embeddings[item_id]   # (D,)
    cand_ids, cand_scores = state["faiss"].search(item_emb, k=top_k + 1)

    results = []
    for rank, idx in enumerate(cand_ids, start=1):
        if int(idx) == item_id:
            continue   # exclude self
        info = state["item_meta"].get(int(idx), {})
        results.append({
            "rank":      rank,
            "item_idx":  int(idx),
            "title":     info.get("title", "Unknown"),
            "genres":    info.get("genres", ""),
            "score":     float(cand_scores[rank - 1]),
        })
        if len(results) >= top_k:
            break

    query_info = state["item_meta"].get(item_id, {})
    return {
        "query_item": {"item_idx": item_id, **query_info},
        "similar_items": results,
    }
