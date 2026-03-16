"""
Ranking evaluation metrics for recommender systems.

Metrics implemented:
  - Recall@K    : fraction of relevant items found in top-K
  - Precision@K : fraction of top-K items that are relevant
  - NDCG@K      : normalised discounted cumulative gain
  - MRR         : mean reciprocal rank of first relevant item
  - HitRate@K   : fraction of users where ≥1 relevant item is in top-K
  - Coverage    : fraction of the item catalogue recommended to ≥1 user
  - Diversity   : mean intra-list genre distance (novelty signal)

All metrics are computed over a batch of users and averaged.

Design note:
  We use a *temporal* train/test split (see splitter.py), so the test set
  contains truly future interactions — this prevents the evaluation from being
  overly optimistic due to temporal leakage.
"""

from typing import Dict, List, Optional

import numpy as np


def recall_at_k(
    recommended: np.ndarray,
    relevant: np.ndarray,
    k: int,
) -> float:
    """
    Recall@K = |Recommended ∩ Relevant| / |Relevant|

    Args:
        recommended: Array of recommended item indices (sorted, best first).
        relevant:    Array of ground-truth item indices.
        k:           Cutoff.

    Returns:
        Recall in [0, 1].
    """
    if len(relevant) == 0:
        return 0.0
    relevant_set = set(relevant)
    hits = sum(1 for item in recommended[:k] if item in relevant_set)
    return hits / min(len(relevant), k)


def precision_at_k(
    recommended: np.ndarray,
    relevant: np.ndarray,
    k: int,
) -> float:
    """Precision@K = |Recommended ∩ Relevant| / K"""
    if k == 0:
        return 0.0
    relevant_set = set(relevant)
    hits = sum(1 for item in recommended[:k] if item in relevant_set)
    return hits / k


def ndcg_at_k(
    recommended: np.ndarray,
    relevant: np.ndarray,
    k: int,
) -> float:
    """
    Normalised Discounted Cumulative Gain at K.

    DCG@K  = Σ_{i=1}^{K} rel_i / log2(i+1)
    IDCG@K = DCG of perfect ranking (all relevant items first)
    NDCG@K = DCG@K / IDCG@K

    Args:
        recommended: Ranked list of recommended item indices.
        relevant:    Set of ground-truth relevant item indices.
        k:           Cutoff.

    Returns:
        NDCG in [0, 1].
    """
    if len(relevant) == 0:
        return 0.0

    relevant_set = set(relevant)
    n_rel = min(len(relevant_set), k)

    # Ideal DCG: top n_rel positions are all relevant
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_rel))
    if idcg == 0:
        return 0.0

    # Actual DCG
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant_set:
            dcg += 1.0 / np.log2(i + 2)

    return dcg / idcg


def hit_rate_at_k(
    recommended: np.ndarray,
    relevant: np.ndarray,
    k: int,
) -> float:
    """HitRate@K = 1 if any relevant item appears in top-K, else 0."""
    if len(relevant) == 0:
        return 0.0
    relevant_set = set(relevant)
    return float(any(item in relevant_set for item in recommended[:k]))


def mrr(
    recommended: np.ndarray,
    relevant: np.ndarray,
) -> float:
    """
    Mean Reciprocal Rank.

    MRR = 1 / rank_of_first_relevant_item
    Returns 0 if no relevant item appears in the list.
    """
    if len(relevant) == 0:
        return 0.0
    relevant_set = set(relevant)
    for rank, item in enumerate(recommended, start=1):
        if item in relevant_set:
            return 1.0 / rank
    return 0.0


def catalog_coverage(
    all_recommendations: List[np.ndarray],
    n_items: int,
) -> float:
    """
    Fraction of the catalogue recommended to at least one user.

    Coverage = |∪_u recommended(u)| / n_items

    Low coverage → model is over-focused on popular items (popularity bias).
    High coverage → model explores the catalogue more broadly.
    """
    recommended_items = set()
    for recs in all_recommendations:
        recommended_items.update(recs.tolist())
    return len(recommended_items) / n_items


def intra_list_diversity(
    recommendations: np.ndarray,
    item_genre_matrix: np.ndarray,
) -> float:
    """
    Mean pairwise cosine *distance* within a recommendation list.

    Higher values → more genre-diverse recommendations.

    ILD = (2 / K(K-1)) * Σ_{i<j} (1 - cosine_similarity(item_i, item_j))

    Args:
        recommendations: Array of recommended item indices.
        item_genre_matrix: (n_items, n_genres) binary genre matrix.

    Returns:
        Mean intra-list distance in [0, 1].
    """
    k = len(recommendations)
    if k < 2:
        return 0.0

    vecs = item_genre_matrix[recommendations]  # (K, n_genres)

    # Normalise rows
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    vecs_normed = vecs / norms

    # Cosine similarity matrix
    sim_matrix = vecs_normed @ vecs_normed.T   # (K, K)

    # Extract upper triangle (excluding diagonal)
    triu_indices = np.triu_indices(k, k=1)
    pairwise_sims = sim_matrix[triu_indices]

    # Distance = 1 - similarity
    return float(np.mean(1.0 - pairwise_sims))


# ── Batch evaluation ──────────────────────────────────────────────────────────

def evaluate_model(
    recommender,
    test: "pd.DataFrame",
    train_matrix,
    k_values: List[int] = (5, 10, 20),
    item_genre_matrix: Optional[np.ndarray] = None,
    n_items: Optional[int] = None,
    max_users: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a recommender on the test set across multiple K values.

    Args:
        recommender:      Fitted BaseRecommender instance.
        test:             Test DataFrame with columns [user_idx, item_idx].
        train_matrix:     CSR matrix of training interactions (for masking).
        k_values:         List of cutoff values to evaluate.
        item_genre_matrix: Optional genre matrix for diversity metric.
        n_items:          Total number of items (for coverage metric).
        max_users:        Limit evaluation to this many users (for speed).
        verbose:          Print progress.

    Returns:
        Dictionary of metric_name → float value.
    """
    import pandas as pd

    # Group test interactions by user
    user_test_items = test.groupby("user_idx")["item_idx"].apply(list).to_dict()
    user_indices = np.array(list(user_test_items.keys()))

    if max_users is not None and len(user_indices) > max_users:
        rng = np.random.default_rng(42)
        user_indices = rng.choice(user_indices, size=max_users, replace=False)

    if verbose:
        print(f"  Evaluating on {len(user_indices):,} users...")

    # Get recommendations for all users at once
    recommendations = recommender.recommend_batch(
        user_indices, top_k=max(k_values), train_matrix=train_matrix
    )  # (n_users, max_k)

    # Accumulate metrics
    results: Dict[str, List[float]] = {f"recall@{k}": [] for k in k_values}
    results.update({f"ndcg@{k}": [] for k in k_values})
    results.update({f"precision@{k}": [] for k in k_values})
    results.update({f"hit_rate@{k}": [] for k in k_values})
    results["mrr"] = []

    all_recs_for_coverage = []

    for i, u in enumerate(user_indices):
        recs    = recommendations[i]
        targets = np.array(user_test_items[u])

        for k in k_values:
            results[f"recall@{k}"].append(recall_at_k(recs, targets, k))
            results[f"ndcg@{k}"].append(ndcg_at_k(recs, targets, k))
            results[f"precision@{k}"].append(precision_at_k(recs, targets, k))
            results[f"hit_rate@{k}"].append(hit_rate_at_k(recs, targets, k))

        results["mrr"].append(mrr(recs, targets))
        all_recs_for_coverage.append(recs[:max(k_values)])

    # Average
    averaged = {metric: float(np.mean(vals)) for metric, vals in results.items()}

    # Coverage
    if n_items is not None:
        primary_k = k_values[1] if len(k_values) > 1 else k_values[0]
        averaged["coverage"] = catalog_coverage(
            [r[:primary_k] for r in all_recs_for_coverage], n_items
        )

    # Diversity
    if item_genre_matrix is not None:
        primary_k = k_values[1] if len(k_values) > 1 else k_values[0]
        div_scores = [
            intra_list_diversity(r[:primary_k], item_genre_matrix)
            for r in all_recs_for_coverage
        ]
        averaged["diversity"] = float(np.mean(div_scores))

    return averaged


def print_results_table(results: Dict[str, Dict[str, float]], k: int = 10) -> None:
    """Pretty-print a comparison table of model results."""
    models = list(results.keys())
    metrics = [f"recall@{k}", f"ndcg@{k}", f"precision@{k}", f"hit_rate@{k}",
               "mrr", "coverage", "diversity"]
    metrics = [m for m in metrics if any(m in results[model] for model in models)]

    col_width = 12
    header = f"{'Model':<15}" + "".join(f"{m:>{col_width}}" for m in metrics)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for model in models:
        row = f"{model:<15}"
        for m in metrics:
            val = results[model].get(m, float("nan"))
            row += f"{val:>{col_width}.4f}"
        print(row)

    print("=" * len(header))
