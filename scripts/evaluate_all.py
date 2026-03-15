"""
Evaluate all trained models and generate results.

Usage:
    python scripts/evaluate_all.py [--config configs/config.yaml]

Outputs:
    results/metrics.csv        — full metric table
    results/plots/             — bar charts and comparison plots
    results/ab_test_report.txt — A/B test results

Metrics reported at K=5, 10, 20:
    Recall@K, NDCG@K, Precision@K, HitRate@K, MRR, Coverage, Diversity
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocess import load_processed
from features.item_features import build_item_features, build_genre_matrix
from features.user_features import build_user_features
from models.popularity import PopularityRecommender
from models.als import ALSRecommender
from models.two_tower import TwoTowerRecommender
from models.sasrec import SASRecRecommender
from evaluation.metrics import evaluate_model, print_results_table
from evaluation.ab_test import run_ab_test, print_ab_report


def load_all_models(model_dir: Path, data, cfg) -> dict:
    """Load all trained models from disk."""
    import torch

    meta         = data["metadata"]
    movies       = data["movies"]
    users        = data["users"]
    train        = data["train"]

    item_genre_matrix = build_genre_matrix(movies)
    item_features     = build_item_features(movies, train, meta["n_items"])
    user_features     = build_user_features(users, train, item_genre_matrix, meta["n_users"])

    models = {}

    # Popularity
    pop = PopularityRecommender()
    pop.load(str(model_dir / "popularity.pkl"))
    models["Popularity"] = pop

    # ALS
    als = ALSRecommender()
    als.load(str(model_dir / "als.pkl"))
    als.train_matrix = data["train_matrix"]
    models["ALS"] = als

    # Two-Tower
    tt_cfg = cfg["models"]["two_tower"]
    tt = TwoTowerRecommender(
        embedding_dim=tt_cfg["embedding_dim"],
        hidden_dims=tuple(tt_cfg["hidden_dims"]),
    )
    tt.user_features_tensor = __import__("torch").tensor(user_features, dtype=__import__("torch").float32)
    tt.item_features_tensor = __import__("torch").tensor(item_features, dtype=__import__("torch").float32)

    checkpoint = __import__("torch").load(model_dir / "two_tower.pt", map_location="cpu", weights_only=False)
    from models.two_tower import TwoTowerModel
    tt.model = TwoTowerModel(
        n_users=meta["n_users"],
        n_items=meta["n_items"],
        user_feature_dim=user_features.shape[1],
        item_feature_dim=item_features.shape[1],
        embedding_dim=tt_cfg["embedding_dim"],
        hidden_dims=tt_cfg["hidden_dims"],
    )
    tt.model.load_state_dict(checkpoint["model_state"])
    tt.model.eval()
    tt.item_embeddings = checkpoint["item_embeddings"]
    tt.is_fitted = True
    models["TwoTower"] = tt

    # SASRec
    sr = SASRecRecommender()
    sr.load(str(model_dir / "sasrec.pt"))
    models["SASRec"] = sr

    return models, item_genre_matrix, item_features


def evaluate_all(config_path: str = "configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_dir  = Path(cfg["output"]["model_dir"])
    plots_dir  = Path(cfg["output"]["plots_dir"])
    results_dir = Path(cfg["output"]["results_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)

    k_values   = cfg["evaluation"]["k_values"]
    primary_k  = cfg["evaluation"]["primary_k"]
    ab_users   = cfg["evaluation"]["ab_test_users"]

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading preprocessed data...")
    data = load_processed(cfg["data"]["processed_dir"])
    test         = data["test"]
    train_matrix = data["train_matrix"]
    meta         = data["metadata"]

    # ── Load models ────────────────────────────────────────────────────────────
    print("Loading trained models...")
    models, item_genre_matrix, item_features = load_all_models(model_dir, data, cfg)

    # ── Evaluate each model ────────────────────────────────────────────────────
    all_results = {}
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}...")
        print(f"{'='*60}")
        t0 = time.time()
        results = evaluate_model(
            recommender=model,
            test=test,
            train_matrix=train_matrix,
            k_values=k_values,
            item_genre_matrix=item_genre_matrix,
            n_items=meta["n_items"],
            verbose=True,
        )
        elapsed = time.time() - t0
        results["eval_time_s"] = round(elapsed, 1)
        all_results[model_name] = results
        print(f"  Eval time: {elapsed:.1f}s")

    # ── Print summary table ────────────────────────────────────────────────────
    print_results_table(all_results, k=primary_k)

    # ── Save metrics CSV ───────────────────────────────────────────────────────
    rows = []
    for model_name, metrics in all_results.items():
        row = {"model": model_name}
        row.update({k: v for k, v in metrics.items() if k != "eval_time_s"})
        rows.append(row)
    metrics_df = pd.DataFrame(rows).set_index("model")
    metrics_df.to_csv(results_dir / "metrics.csv")
    print(f"\nMetrics saved to {results_dir / 'metrics.csv'}")

    # ── A/B Test: ALS (control) vs SASRec (treatment) ─────────────────────────
    print(f"\n{'='*60}")
    print("Running A/B test: ALS vs SASRec...")
    print(f"{'='*60}")
    ab_result = run_ab_test(
        model_a=models["ALS"],
        model_b=models["SASRec"],
        test=test,
        train_matrix=train_matrix,
        k=primary_k,
        n_users=ab_users,
    )
    print_ab_report(ab_result)

    # Save A/B report
    ab_report_path = results_dir / "ab_test_report.json"
    ab_result_serializable = {
        k: v for k, v in ab_result.items()
        if not isinstance(v, np.ndarray)
    }
    with open(ab_report_path, "w") as f:
        json.dump(ab_result_serializable, f, indent=2)
    print(f"\nA/B report saved to {ab_report_path}")

    # ── Generate plots ─────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    generate_plots(all_results, ab_result, plots_dir, k_values, primary_k)
    print(f"Plots saved to {plots_dir}")

    return all_results, ab_result


def generate_plots(all_results, ab_result, plots_dir, k_values, primary_k):
    """Generate bar charts and metric comparison plots."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"figure.dpi": 120, "font.size": 11})

    model_names = list(all_results.keys())
    colors = sns.color_palette("muted", len(model_names))

    # ── Plot 1: Recall@K and NDCG@K across all K values ──────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for metric_base, ax in zip(["recall", "ndcg"], axes):
        for i, model_name in enumerate(model_names):
            vals = [all_results[model_name].get(f"{metric_base}@{k}", 0) for k in k_values]
            ax.plot(k_values, vals, marker="o", label=model_name,
                    color=colors[i], linewidth=2, markersize=7)
        ax.set_xlabel("K (cutoff)")
        ax.set_ylabel(metric_base.upper() + "@K")
        ax.set_title(f"{metric_base.upper()}@K Comparison")
        ax.legend()
        ax.set_xticks(k_values)

    plt.suptitle("Model Comparison: Recall and NDCG at Multiple Cutoffs", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(plots_dir / "recall_ndcg_curves.png", bbox_inches="tight")
    plt.close()

    # ── Plot 2: Bar chart of all metrics at primary K ─────────────────────────
    primary_metrics = [
        f"recall@{primary_k}", f"ndcg@{primary_k}",
        f"precision@{primary_k}", f"hit_rate@{primary_k}",
        "mrr", "coverage", "diversity"
    ]
    present_metrics = [m for m in primary_metrics if any(
        m in all_results[mn] for mn in model_names
    )]

    n_metrics = len(present_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(3 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, present_metrics):
        vals = [all_results[mn].get(metric, 0) for mn in model_names]
        bars = ax.bar(model_names, vals, color=colors, edgecolor="white", linewidth=1.2)
        ax.set_title(metric.replace("@", "@\n"), fontsize=10)
        ax.set_ylim(0, max(vals) * 1.2 + 0.001)
        ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=9)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle(f"All Metrics at K={primary_k}", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(plots_dir / "metrics_bar_chart.png", bbox_inches="tight")
    plt.close()

    # ── Plot 3: A/B Test distribution ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    scores_a = ab_result["per_user_a"]
    scores_b = ab_result["per_user_b"]

    ax.hist(scores_a, bins=20, alpha=0.6, color=colors[1], label=ab_result["model_a_name"])
    ax.hist(scores_b, bins=20, alpha=0.6, color=colors[3], label=ab_result["model_b_name"])
    ax.axvline(scores_a.mean(), color=colors[1], linestyle="--", linewidth=2,
               label=f"Mean A: {scores_a.mean():.4f}")
    ax.axvline(scores_b.mean(), color=colors[3], linestyle="--", linewidth=2,
               label=f"Mean B: {scores_b.mean():.4f}")
    ax.set_xlabel(ab_result["metric_name"])
    ax.set_ylabel("Number of users")
    ax.set_title(
        f"A/B Test: {ab_result['model_a_name']} vs {ab_result['model_b_name']}\n"
        f"p={ab_result['p_value']:.4f}  |  "
        f"Relative lift: {ab_result['relative_lift_pct']:+.1f}%"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "ab_test_distribution.png", bbox_inches="tight")
    plt.close()

    # ── Plot 4: Relevance vs Diversity scatter ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (model_name, res) in enumerate(all_results.items()):
        recall = res.get(f"recall@{primary_k}", 0)
        div    = res.get("diversity", 0)
        ax.scatter(div, recall, s=180, color=colors[i], zorder=3,
                   edgecolors="white", linewidths=1.5)
        ax.annotate(model_name, (div, recall), textcoords="offset points",
                    xytext=(8, 4), fontsize=10)

    ax.set_xlabel("Diversity (Intra-List Distance)")
    ax.set_ylabel(f"Recall@{primary_k}")
    ax.set_title("Relevance vs Diversity Trade-off")
    plt.tight_layout()
    plt.savefig(plots_dir / "relevance_vs_diversity.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all recommender models.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    evaluate_all(args.config)
