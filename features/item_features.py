"""
Item (content) feature engineering.

We build a feature vector for each movie using:
  1. Multi-hot genre encoding  (18 genres in MovieLens 1M)
  2. Release year (extracted from title, normalized)
  3. Popularity features (log interaction count)

These features feed into the Two-Tower model's item tower and
provide a content-based signal for cold-start recommendations.
"""

import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# All genres present in MovieLens 1M (sorted for reproducibility)
MOVIELENS_GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western",
]
GENRE_TO_IDX = {g: i for i, g in enumerate(MOVIELENS_GENRES)}
N_GENRES = len(MOVIELENS_GENRES)


def extract_year(title: str) -> float:
    """Extract 4-digit year from a title like 'Toy Story (1995)'."""
    match = re.search(r"\((\d{4})\)", str(title))
    return float(match.group(1)) if match else np.nan


def build_genre_matrix(movies: pd.DataFrame) -> np.ndarray:
    """
    Build a multi-hot genre matrix of shape (n_items, N_GENRES).

    Each row is a binary vector where 1 indicates the movie belongs to
    that genre.  Multi-hot encoding allows a movie to belong to multiple
    genres simultaneously (e.g., Action + Comedy).
    """
    n_items = movies["item_idx"].max() + 1
    genre_matrix = np.zeros((n_items, N_GENRES), dtype=np.float32)

    for _, row in movies.iterrows():
        idx = row["item_idx"]
        for genre in row["genre_list"]:
            if genre in GENRE_TO_IDX:
                genre_matrix[idx, GENRE_TO_IDX[genre]] = 1.0

    return genre_matrix


def build_item_features(
    movies: pd.DataFrame,
    train: pd.DataFrame,
    n_items: int,
) -> np.ndarray:
    """
    Build a dense feature matrix for all items.

    Feature vector layout (per item):
      [0:N_GENRES]   multi-hot genre encoding     (18 dims)
      [N_GENRES]     release year, normalized      (1 dim)
      [N_GENRES+1]   log10 interaction count       (1 dim)

    Total: 20 dimensions

    Args:
        movies:  Processed movies DataFrame with item_idx and genre_list.
        train:   Training interactions for computing popularity.
        n_items: Total number of items (including those not in movies).

    Returns:
        feature_matrix: np.ndarray of shape (n_items, feature_dim)
    """
    # ── Genre encoding
    genre_matrix = build_genre_matrix(movies)

    # ── Release year
    movies = movies.copy()
    movies["year"] = movies["title"].apply(extract_year)
    year_min = movies["year"].min()
    year_max = movies["year"].max()

    year_vector = np.zeros(n_items, dtype=np.float32)
    for _, row in movies.iterrows():
        if not np.isnan(row["year"]):
            year_vector[int(row["item_idx"])] = (
                (row["year"] - year_min) / (year_max - year_min + 1e-8)
            )

    # ── Popularity (log-normalised interaction count)
    counts = train["item_idx"].value_counts()
    pop_vector = np.zeros(n_items, dtype=np.float32)
    for item_idx, count in counts.items():
        pop_vector[item_idx] = np.log10(count + 1)
    if pop_vector.max() > 0:
        pop_vector /= pop_vector.max()

    # ── Concatenate
    feature_matrix = np.concatenate(
        [genre_matrix,
         year_vector.reshape(-1, 1),
         pop_vector.reshape(-1, 1)],
        axis=1,
    ).astype(np.float32)

    return feature_matrix
