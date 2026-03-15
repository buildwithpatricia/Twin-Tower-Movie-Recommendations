"""
User feature engineering.

We build a feature vector for each user combining:
  1. Demographic features (gender, age bucket, occupation) — one-hot encoded
  2. Interaction statistics (activity level, average rating, recency)
  3. Genre affinity vector — what proportion of watched items belong to each genre

This gives the Two-Tower model's user tower a rich side-information signal
that's especially valuable for sparse users.
"""

import numpy as np
import pandas as pd


# MovieLens 1M occupation codes (0-20)
N_OCCUPATIONS = 21
N_AGE_BUCKETS = 7   # see AGE_MAP in preprocess.py
N_GENDERS     = 2


def build_user_features(
    users: pd.DataFrame,
    train: pd.DataFrame,
    item_genre_matrix: np.ndarray,
    n_users: int,
) -> np.ndarray:
    """
    Build a dense feature matrix for all users.

    Feature vector layout (per user):
      [0:2]               gender one-hot               (2 dims)
      [2:9]               age bucket one-hot            (7 dims)
      [9:30]              occupation one-hot            (21 dims)
      [30]                log10 interaction count       (1 dim)
      [31]                mean rating (normalised)      (1 dim)
      [32]                recency score                 (1 dim)
      [33:33+N_GENRES]    genre affinity vector         (18 dims)

    Total: 51 dimensions

    Args:
        users:             Processed users DataFrame with user_idx.
        train:             Training interactions DataFrame.
        item_genre_matrix: Item genre matrix (n_items, N_GENRES).
        n_users:           Total number of users.

    Returns:
        feature_matrix: np.ndarray of shape (n_users, feature_dim)
    """
    N_GENRES = item_genre_matrix.shape[1]
    feature_dim = 2 + 7 + 21 + 3 + N_GENRES   # = 51
    feature_matrix = np.zeros((n_users, feature_dim), dtype=np.float32)

    # ── Precompute per-user statistics from training data
    user_stats = (
        train.groupby("user_idx")
        .agg(
            interaction_count=("item_idx", "count"),
            mean_rating=("rating", "mean"),
            max_timestamp=("timestamp", "max"),
        )
        .reset_index()
    )

    global_max_ts = train["timestamp"].max()
    global_min_ts = train["timestamp"].min()
    ts_range = max(global_max_ts - global_min_ts, 1)

    # Normalisation denominators
    max_log_count = np.log10(user_stats["interaction_count"].max() + 1)

    # ── Genre affinity: for each user, average genre vector of watched items
    user_genre_sums = np.zeros((n_users, N_GENRES), dtype=np.float32)
    user_item_counts = np.zeros(n_users, dtype=np.float32)

    for _, row in train.iterrows():
        u = int(row["user_idx"])
        it = int(row["item_idx"])
        if it < len(item_genre_matrix):
            user_genre_sums[u] += item_genre_matrix[it]
            user_item_counts[u] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        user_genre_affinity = np.where(
            user_item_counts[:, None] > 0,
            user_genre_sums / user_item_counts[:, None],
            0.0,
        )

    # ── Demographic features
    gender_map = {"F": 0, "M": 1}
    age_map    = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}

    for _, row in users.iterrows():
        u = int(row["user_idx"])

        # Gender (2-dim one-hot)
        g = gender_map.get(row["gender"], 0)
        feature_matrix[u, g] = 1.0

        # Age bucket (7-dim one-hot)
        a = age_map.get(int(row["age"]), 0)
        feature_matrix[u, 2 + a] = 1.0

        # Occupation (21-dim one-hot)
        occ = int(row["occupation"])
        if 0 <= occ < N_OCCUPATIONS:
            feature_matrix[u, 9 + occ] = 1.0

    # ── Interaction statistics
    for _, row in user_stats.iterrows():
        u = int(row["user_idx"])
        feature_matrix[u, 30] = np.log10(row["interaction_count"] + 1) / (max_log_count + 1e-8)
        feature_matrix[u, 31] = (row["mean_rating"] - 1.0) / 4.0   # normalise [1,5] → [0,1]
        recency = (row["max_timestamp"] - global_min_ts) / ts_range
        feature_matrix[u, 32] = recency

    # ── Genre affinity
    feature_matrix[:, 33:33 + N_GENRES] = user_genre_affinity

    return feature_matrix
