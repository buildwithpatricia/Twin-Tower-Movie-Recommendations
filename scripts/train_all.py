"""
Train all recommender models and save to results/models/.

Usage:
    python scripts/train_all.py [--config configs/config.yaml]

This script:
  1. Loads preprocessed data (run scripts/run_pipeline.py first)
  2. Builds user/item feature matrices
  3. Trains Popularity, ALS, Two-Tower, and SASRec models
  4. Saves each model to results/models/
  5. Builds a FAISS index from Two-Tower item embeddings

Runtime estimate (CPU):
  - Popularity : < 1 second
  - ALS        : ~2-5 minutes
  - Two-Tower  : ~10-20 minutes (20 epochs, ~800K interactions)
  - SASRec     : ~15-25 minutes (20 epochs)
"""

import argparse
import sys
import time
from pathlib import Path

import yaml

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocess import load_processed
from features.item_features import build_item_features, build_genre_matrix
from features.user_features import build_user_features
from models.popularity import PopularityRecommender
from models.als import ALSRecommender
from models.two_tower import TwoTowerRecommender
from models.sasrec import SASRecRecommender
from serving.retrieval import FAISSRetriever


def train_all(config_path: str = "configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_dir = Path(cfg["output"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Load preprocessed data ─────────────────────────────────────────────────
    print("=" * 60)
    print("Loading preprocessed data...")
    print("=" * 60)
    data = load_processed(cfg["data"]["processed_dir"])

    train        = data["train"]
    train_matrix = data["train_matrix"]
    movies       = data["movies"]
    users        = data["users"]
    meta         = data["metadata"]
    n_users      = meta["n_users"]
    n_items      = meta["n_items"]

    print(f"  Users: {n_users:,}  |  Items: {n_items:,}")
    print(f"  Train: {meta['n_train']:,} interactions")

    # ── Build features ─────────────────────────────────────────────────────────
    print("\nBuilding feature matrices...")
    item_genre_matrix = build_genre_matrix(movies)
    item_features     = build_item_features(movies, train, n_items)
    user_features     = build_user_features(users, train, item_genre_matrix, n_users)
    print(f"  Item features: {item_features.shape}")
    print(f"  User features: {user_features.shape}")

    # ── 1. Popularity ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training Popularity baseline...")
    print("=" * 60)
    t0 = time.time()
    pop_model = PopularityRecommender()
    pop_model.fit(train=train, n_items=n_items)
    pop_model.save(str(model_dir / "popularity.pkl"))
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── 2. ALS ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training ALS...")
    print("=" * 60)
    t0 = time.time()
    als_cfg = cfg["models"]["als"]
    als_model = ALSRecommender(
        factors=als_cfg["factors"],
        iterations=als_cfg["iterations"],
        regularization=als_cfg["regularization"],
        alpha=als_cfg["alpha"],
        random_state=als_cfg["random_state"],
    )
    als_model.fit(train_matrix=train_matrix)
    als_model.save(str(model_dir / "als.pkl"))
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── 3. Two-Tower ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training Two-Tower...")
    print("=" * 60)
    t0 = time.time()
    tt_cfg = cfg["models"]["two_tower"]
    tt_model = TwoTowerRecommender(
        embedding_dim=tt_cfg["embedding_dim"],
        hidden_dims=tuple(tt_cfg["hidden_dims"]),
        dropout=tt_cfg["dropout"],
        learning_rate=tt_cfg["learning_rate"],
        batch_size=tt_cfg["batch_size"],
        epochs=tt_cfg["epochs"],
        num_negatives=tt_cfg["num_negatives"],
        random_state=tt_cfg["random_state"],
    )
    tt_model.fit(
        train=train,
        n_users=n_users,
        n_items=n_items,
        user_features=user_features,
        item_features=item_features,
    )
    tt_model.save(str(model_dir / "two_tower.pt"))
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── 4. SASRec ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training SASRec...")
    print("=" * 60)
    t0 = time.time()
    sr_cfg = cfg["models"]["sasrec"]
    sr_model = SASRecRecommender(
        embedding_dim=sr_cfg["embedding_dim"],
        n_heads=sr_cfg["num_heads"],
        n_layers=sr_cfg["num_layers"],
        dropout=sr_cfg["dropout"],
        max_seq_len=sr_cfg["max_seq_len"],
        learning_rate=sr_cfg["learning_rate"],
        batch_size=sr_cfg["batch_size"],
        epochs=sr_cfg["epochs"],
        random_state=sr_cfg["random_state"],
    )
    sr_model.fit(train=train, n_items=n_items)
    sr_model.save(str(model_dir / "sasrec.pt"))
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── 5. FAISS index (Two-Tower item embeddings) ─────────────────────────────
    print("\n" + "=" * 60)
    print("Building FAISS index...")
    print("=" * 60)
    t0 = time.time()
    retriever = FAISSRetriever(index_type=cfg["serving"]["faiss_index_type"])
    retriever.build(tt_model.item_embeddings)
    retriever.save(str(model_dir / "faiss_index.bin"))
    print(f"  Done in {time.time() - t0:.1f}s")

    print("\n" + "=" * 60)
    print("All models trained and saved to:", model_dir)
    print("=" * 60)
    return {
        "popularity": pop_model,
        "als":        als_model,
        "two_tower":  tt_model,
        "sasrec":     sr_model,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all recommender models.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train_all(args.config)
