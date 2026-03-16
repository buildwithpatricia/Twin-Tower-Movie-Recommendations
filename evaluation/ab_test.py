"""
A/B Test Simulation for Recommender Models.

In production, you'd run a live A/B test routing a percentage of traffic to
each model variant and measuring downstream business metrics (CTR, watch time,
session length).  Here we simulate that process using held-out test interactions
as a proxy for user engagement.

Statistical methodology:
  - Split test users randomly into control (model A) and treatment (model B)
  - Compute per-user Recall@K as the proxy metric (1 = hit, 0 = miss)
  - Run a two-sample t-test (Welch's) to test H₀: μ_A = μ_B
  - Report p-value, confidence interval, and relative lift

This approach mirrors real production A/B testing (e.g., Netflix, Spotify)
where binary engagement signals (clicked / not clicked) are averaged per user.
"""

from typing import Dict, Optional, Tuple
from scipy import stats
import numpy as np


def run_ab_test(
    model_a,
    model_b,
    test: "pd.DataFrame",
    train_matrix,
    k: int = 10,
    n_users: int = 1000,
    random_state: int = 42,
    confidence_level: float = 0.95,
) -> Dict:
    """
    Simulate an A/B test between two recommender models.

    Args:
        model_a:          Control model (e.g., ALS).
        model_b:          Treatment model (e.g., SASRec).
        test:             Test DataFrame with [user_idx, item_idx].
        train_matrix:     CSR matrix of training interactions.
        k:                Recommendation list length.
        n_users:          Number of users to include in the test.
        random_state:     Seed for user sampling.
        confidence_level: e.g. 0.95 for 95% CI.

    Returns:
        Dictionary containing:
          - metric_a, metric_b: mean Recall@K for each model
          - absolute_lift:     metric_b - metric_a
          - relative_lift_pct: (metric_b - metric_a) / metric_a * 100
          - p_value:           two-sample Welch's t-test p-value
          - significant:       whether p < (1 - confidence_level)
          - ci_lower, ci_upper: confidence interval for the difference
          - per_user_a, per_user_b: per-user metric arrays
    """
    import pandas as pd
    from evaluation.metrics import recall_at_k

    rng = np.random.default_rng(random_state)

    # ── Sample users with test interactions
    user_test_items = test.groupby("user_idx")["item_idx"].apply(list).to_dict()
    all_users = np.array(list(user_test_items.keys()))
    n_users = min(n_users, len(all_users))
    sampled_users = rng.choice(all_users, size=n_users, replace=False)

    # 50/50 split
    mid = n_users // 2
    users_a = sampled_users[:mid]
    users_b = sampled_users[mid:]

    # ── Get recommendations from each model
    recs_a = model_a.recommend_batch(users_a, top_k=k, train_matrix=train_matrix)
    recs_b = model_b.recommend_batch(users_b, top_k=k, train_matrix=train_matrix)

    # ── Per-user Recall@K (binary: 1 = at least one hit, 0 = miss)
    def per_user_recall(users, recommendations):
        scores = []
        for i, u in enumerate(users):
            targets = user_test_items.get(u, [])
            scores.append(recall_at_k(recommendations[i], targets, k))
        return np.array(scores)

    scores_a = per_user_recall(users_a, recs_a)
    scores_b = per_user_recall(users_b, recs_b)

    # ── Welch's two-sample t-test
    t_stat, p_value = stats.ttest_ind(scores_b, scores_a, equal_var=False)

    # ── Confidence interval for the difference in means
    alpha = 1 - confidence_level
    mean_diff = scores_b.mean() - scores_a.mean()
    se = np.sqrt(scores_a.var() / len(scores_a) + scores_b.var() / len(scores_b))
    # Approximate degrees of freedom (Welch-Satterthwaite)
    df = (scores_a.var() / len(scores_a) + scores_b.var() / len(scores_b)) ** 2 / (
        (scores_a.var() / len(scores_a)) ** 2 / (len(scores_a) - 1) +
        (scores_b.var() / len(scores_b)) ** 2 / (len(scores_b) - 1)
    )
    t_crit = stats.t.ppf(1 - alpha / 2, df=df)
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se

    metric_a = float(scores_a.mean())
    metric_b = float(scores_b.mean())
    abs_lift  = metric_b - metric_a
    rel_lift  = (abs_lift / metric_a * 100) if metric_a > 0 else 0.0

    return {
        "model_a_name":      model_a.name,
        "model_b_name":      model_b.name,
        "n_users_a":         len(users_a),
        "n_users_b":         len(users_b),
        "metric_name":       f"Recall@{k}",
        "metric_a":          metric_a,
        "metric_b":          metric_b,
        "absolute_lift":     float(abs_lift),
        "relative_lift_pct": float(rel_lift),
        "p_value":           float(p_value),
        "t_statistic":       float(t_stat),
        "significant":       bool(p_value < (1 - confidence_level)),
        "confidence_level":  confidence_level,
        "ci_lower":          float(ci_lower),
        "ci_upper":          float(ci_upper),
        "per_user_a":        scores_a,
        "per_user_b":        scores_b,
    }


def print_ab_report(result: Dict) -> None:
    """Print a formatted A/B test report."""
    a_name = result["model_a_name"]
    b_name = result["model_b_name"]
    metric = result["metric_name"]
    ci_pct = int(result["confidence_level"] * 100)

    print("\n" + "=" * 55)
    print(f"  A/B TEST REPORT: {a_name} vs {b_name}")
    print("=" * 55)
    print(f"  Metric          : {metric}")
    print(f"  Users (control) : {result['n_users_a']:,}")
    print(f"  Users (treat.)  : {result['n_users_b']:,}")
    print("-" * 55)
    print(f"  Control  [{a_name}]  : {result['metric_a']:.4f}")
    print(f"  Treatment[{b_name}] : {result['metric_b']:.4f}")
    print(f"  Absolute lift   : {result['absolute_lift']:+.4f}")
    print(f"  Relative lift   : {result['relative_lift_pct']:+.2f}%")
    print("-" * 55)
    print(f"  p-value         : {result['p_value']:.4f}")
    print(f"  t-statistic     : {result['t_statistic']:.3f}")
    print(f"  {ci_pct}% CI for diff  : "
          f"[{result['ci_lower']:+.4f}, {result['ci_upper']:+.4f}]")
    sig = "YES ✓" if result["significant"] else "NO ✗"
    print(f"  Significant?    : {sig}")
    print("=" * 55)

    if result["significant"] and result["absolute_lift"] > 0:
        print(f"\n  Recommendation: SHIP {b_name} — statistically significant "
              f"improvement of {result['relative_lift_pct']:+.1f}% over {a_name}.")
    elif result["significant"] and result["absolute_lift"] < 0:
        print(f"\n  Recommendation: DO NOT SHIP {b_name} — significant regression.")
    else:
        print(f"\n  Recommendation: Inconclusive — run longer or increase sample size.")
