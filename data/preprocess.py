"""
Preprocessing pipeline for MovieLens 1M.

Steps:
1. Parse raw .dat files (ratings, movies, users)
2. Filter cold-start users and items
3. Map IDs to contiguous integers (required by embedding layers)
4. Perform temporal train/val/test split
5. Save processed data to disk
"""

import os
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


# ── Raw file column specs ──────────────────────────────────────────────────────

RATINGS_COLS = ["user_id", "item_id", "rating", "timestamp"]
MOVIES_COLS  = ["item_id", "title", "genres"]
USERS_COLS   = ["user_id", "gender", "age", "occupation", "zip_code"]

# MovieLens age buckets
AGE_MAP = {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44",
           45: "45-49", 50: "50-55", 56: "56+"}


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_ratings(data_dir: Path) -> pd.DataFrame:
    """Load ratings.dat into a DataFrame."""
    path = data_dir / "ratings.dat"
    df = pd.read_csv(path, sep="::", header=None, names=RATINGS_COLS,
                     engine="python", encoding="latin-1")
    df["timestamp"] = df["timestamp"].astype(int)
    return df


def load_movies(data_dir: Path) -> pd.DataFrame:
    """Load movies.dat and expand pipe-separated genres into a list column."""
    path = data_dir / "movies.dat"
    df = pd.read_csv(path, sep="::", header=None, names=MOVIES_COLS,
                     engine="python", encoding="latin-1")
    df["genre_list"] = df["genres"].str.split("|")
    return df


def load_users(data_dir: Path) -> pd.DataFrame:
    """Load users.dat."""
    path = data_dir / "users.dat"
    df = pd.read_csv(path, sep="::", header=None, names=USERS_COLS,
                     engine="python", encoding="latin-1")
    df["age_label"] = df["age"].map(AGE_MAP)
    return df


# ── Filtering ─────────────────────────────────────────────────────────────────

def filter_cold_start(
    ratings: pd.DataFrame,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
) -> pd.DataFrame:
    """
    Iteratively remove users and items with fewer than the threshold
    number of interactions.  Runs until convergence (no more removals).
    """
    n_before = len(ratings)
    for iteration in range(10):  # at most 10 passes
        user_counts = ratings["user_id"].value_counts()
        item_counts = ratings["item_id"].value_counts()

        valid_users = user_counts[user_counts >= min_user_interactions].index
        valid_items = item_counts[item_counts >= min_item_interactions].index

        ratings = ratings[
            ratings["user_id"].isin(valid_users) &
            ratings["item_id"].isin(valid_items)
        ]

        if len(ratings) == n_before:
            break
        n_before = len(ratings)

    return ratings.reset_index(drop=True)


# ── ID remapping ──────────────────────────────────────────────────────────────

def build_id_maps(
    ratings: pd.DataFrame,
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int], Dict[int, int]]:
    """
    Create contiguous 0-indexed integer maps for user and item IDs.

    Returns:
        user2idx: original_user_id → 0-indexed int
        idx2user: 0-indexed int → original_user_id
        item2idx: original_item_id → 0-indexed int
        idx2item: 0-indexed int → original_item_id
    """
    users = sorted(ratings["user_id"].unique())
    items = sorted(ratings["item_id"].unique())

    user2idx = {u: i for i, u in enumerate(users)}
    idx2user = {i: u for u, i in user2idx.items()}
    item2idx = {it: i for i, it in enumerate(items)}
    idx2item = {i: it for it, i in item2idx.items()}

    return user2idx, idx2user, item2idx, idx2item


# ── Temporal split ────────────────────────────────────────────────────────────

def temporal_split(
    ratings: pd.DataFrame,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    For each user, sort interactions by timestamp and put:
      - last test_ratio fraction  → test set
      - next val_ratio fraction   → validation set
      - remainder                 → training set

    This simulates real-world deployment: we train on historical data
    and predict future interactions.

    Args:
        ratings:    DataFrame with columns [user_id, item_id, rating, timestamp]
        val_ratio:  Fraction of each user's history reserved for validation.
        test_ratio: Fraction of each user's history reserved for testing.

    Returns:
        train, val, test DataFrames
    """
    ratings = ratings.sort_values(["user_id", "timestamp"])

    train_rows, val_rows, test_rows = [], [], []

    for user_id, group in ratings.groupby("user_id"):
        n = len(group)
        n_test = max(1, int(n * test_ratio))
        n_val  = max(1, int(n * val_ratio))
        n_train = n - n_val - n_test

        if n_train < 1:
            # Not enough interactions — put everything in train
            train_rows.append(group)
            continue

        train_rows.append(group.iloc[:n_train])
        val_rows.append(group.iloc[n_train : n_train + n_val])
        test_rows.append(group.iloc[n_train + n_val :])

    train = pd.concat(train_rows, ignore_index=True)
    val   = pd.concat(val_rows,   ignore_index=True)
    test  = pd.concat(test_rows,  ignore_index=True)

    return train, val, test


# ── Sparse matrix builder ─────────────────────────────────────────────────────

def build_user_item_matrix(
    ratings: pd.DataFrame,
    n_users: int,
    n_items: int,
    binary: bool = True,
) -> csr_matrix:
    """
    Build a user × item sparse matrix.

    Args:
        ratings: DataFrame with columns [user_idx, item_idx, rating]
        n_users: Total number of users (matrix rows).
        n_items: Total number of items (matrix cols).
        binary:  If True, replace all ratings with 1 (implicit feedback).

    Returns:
        CSR sparse matrix of shape (n_users, n_items).
    """
    values = np.ones(len(ratings)) if binary else ratings["rating"].values
    row = ratings["user_idx"].values
    col = ratings["item_idx"].values
    return csr_matrix((values, (row, col)), shape=(n_users, n_items))


# ── Master pipeline ───────────────────────────────────────────────────────────

def run_preprocessing(
    raw_dir: str = "data/raw/ml-1m",
    processed_dir: str = "data/processed",
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict:
    """
    Full preprocessing pipeline.  Saves all artifacts to processed_dir and
    returns a dict with the processed DataFrames and metadata.
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load raw data
    print("Loading raw data...")
    ratings = load_ratings(raw_dir)
    movies  = load_movies(raw_dir)
    users   = load_users(raw_dir)

    print(f"  Raw ratings : {len(ratings):,}")
    print(f"  Raw users   : {ratings['user_id'].nunique():,}")
    print(f"  Raw items   : {ratings['item_id'].nunique():,}")

    # ── 2. Implicit feedback: convert ratings → binary interactions
    #     We keep all interactions (not just high ratings) to maximise signal,
    #     matching Tubi's implicit watch-history setting.
    ratings = ratings[["user_id", "item_id", "rating", "timestamp"]].copy()

    # ── 3. Filter cold-start
    print("Filtering cold-start users/items...")
    ratings = filter_cold_start(ratings, min_user_interactions, min_item_interactions)
    print(f"  Filtered ratings: {len(ratings):,}")
    print(f"  Filtered users  : {ratings['user_id'].nunique():,}")
    print(f"  Filtered items  : {ratings['item_id'].nunique():,}")

    # ── 4. Build ID maps
    user2idx, idx2user, item2idx, idx2item = build_id_maps(ratings)
    n_users = len(user2idx)
    n_items = len(item2idx)

    ratings["user_idx"] = ratings["user_id"].map(user2idx)
    ratings["item_idx"] = ratings["item_id"].map(item2idx)

    # Map movie/user metadata to new indices
    movies = movies[movies["item_id"].isin(item2idx)].copy()
    movies["item_idx"] = movies["item_id"].map(item2idx)
    users  = users[users["user_id"].isin(user2idx)].copy()
    users["user_idx"]  = users["user_id"].map(user2idx)

    # ── 5. Temporal split
    print("Performing temporal train/val/test split...")
    train, val, test = temporal_split(ratings, val_ratio, test_ratio)
    print(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    # ── 6. Build sparse matrices
    train_matrix = build_user_item_matrix(train, n_users, n_items, binary=True)
    val_matrix   = build_user_item_matrix(val,   n_users, n_items, binary=True)

    # ── 7. Save artifacts
    print(f"Saving processed data to {processed_dir}...")

    train.to_parquet(processed_dir / "train.parquet", index=False)
    val.to_parquet(processed_dir   / "val.parquet",   index=False)
    test.to_parquet(processed_dir  / "test.parquet",  index=False)
    movies.to_parquet(processed_dir / "movies.parquet", index=False)
    users.to_parquet(processed_dir  / "users.parquet",  index=False)

    with open(processed_dir / "id_maps.pkl", "wb") as f:
        pickle.dump({"user2idx": user2idx, "idx2user": idx2user,
                     "item2idx": item2idx, "idx2item": idx2item}, f)

    # Save sparse matrices in scipy format
    from scipy.sparse import save_npz
    save_npz(processed_dir / "train_matrix.npz", train_matrix)
    save_npz(processed_dir / "val_matrix.npz",   val_matrix)

    metadata = {
        "n_users": n_users,
        "n_items": n_items,
        "n_train": len(train),
        "n_val":   len(val),
        "n_test":  len(test),
        "density": train_matrix.nnz / (n_users * n_items),
    }
    with open(processed_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"\nPreprocessing complete.")
    print(f"  Users: {n_users:,}  |  Items: {n_items:,}")
    print(f"  Matrix density: {metadata['density']:.4%}")

    return {
        "train": train, "val": val, "test": test,
        "movies": movies, "users": users,
        "train_matrix": train_matrix, "val_matrix": val_matrix,
        "user2idx": user2idx, "idx2user": idx2user,
        "item2idx": item2idx, "idx2item": idx2item,
        "metadata": metadata,
    }


def load_processed(processed_dir: str = "data/processed") -> dict:
    """Load all preprocessed artifacts from disk."""
    processed_dir = Path(processed_dir)
    from scipy.sparse import load_npz

    with open(processed_dir / "id_maps.pkl", "rb") as f:
        id_maps = pickle.load(f)
    with open(processed_dir / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    return {
        "train":        pd.read_parquet(processed_dir / "train.parquet"),
        "val":          pd.read_parquet(processed_dir / "val.parquet"),
        "test":         pd.read_parquet(processed_dir / "test.parquet"),
        "movies":       pd.read_parquet(processed_dir / "movies.parquet"),
        "users":        pd.read_parquet(processed_dir / "users.parquet"),
        "train_matrix": load_npz(processed_dir / "train_matrix.npz"),
        "val_matrix":   load_npz(processed_dir / "val_matrix.npz"),
        **id_maps,
        "metadata":     metadata,
    }
