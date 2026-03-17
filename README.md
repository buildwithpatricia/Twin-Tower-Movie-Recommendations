# Streaming Content Recommendation Engine

A recommendation system built for streaming platforms. Implements four recommender models — from a simple popularity baseline up to a Transformer-based sequential model — with a full evaluation pipeline, A/B testing framework, and a live FastAPI serving layer.

---

## Architecture

```
                        ┌─────────────────────────────────┐
                        │        Request: user_id          │
                        └────────────────┬────────────────┘
                                         │
                        ┌────────────────▼────────────────┐
                        │   Two-Tower User Encoder         │
                        │   user_id + side features        │
                        │   → 64-dim L2-normed embedding   │
                        └────────────────┬────────────────┘
                                         │
                        ┌────────────────▼────────────────┐
                        │   FAISS ANN Retrieval            │
                        │   IndexFlatIP (exact inner prod) │
                        │   → top-50 candidate items       │
                        └────────────────┬────────────────┘
                                         │
                        ┌────────────────▼────────────────┐
                        │   MMR Diversity Re-ranker        │
                        │   λ·relevance − (1-λ)·sim        │
                        │   → final top-10 recommendations │
                        └─────────────────────────────────┘
```

### Models implemented

| Model | Type | Key idea |
|---|---|---|
| **Popularity** | Non-personalised baseline | Rank items by global interaction count |
| **ALS** | Matrix factorisation | Alternating Least Squares on implicit feedback (Hu et al. 2008) |
| **Two-Tower** | Neural dual encoder | User tower + item tower trained with BPR loss; pre-computed item embeddings + FAISS |
| **SASRec** | Sequential Transformer | Causal self-attention over user watch history (Wang et al. 2018) |

---

## Results

Evaluated on the **MovieLens 1M** dataset (1M ratings, 6,040 users, 3,416 movies) with a **temporal train/val/test split** — the last 10% of each user's chronological history is held out for testing. This mirrors real deployment conditions where you predict future interactions.

### Offline Metrics (K = 10)

| Model | Recall@10 | NDCG@10 | Precision@10 | HitRate@10 | MRR | Coverage | Diversity |
|---|---|---|---|---|---|---|---|
| Popularity | 0.0589 | 0.0558 | 0.0470 | 0.2810 | 0.1232 | 0.034 | 0.634 |
| ALS | 0.0585 | 0.0469 | 0.0343 | 0.2672 | 0.1032 | 0.597 | 0.643 |
| TwoTower | 0.0382 | 0.0349 | 0.0304 | 0.2137 | 0.0845 | 0.268 | 0.647 |
| **SASRec** | **0.0608** | **0.0546** | 0.0438 | **0.2940** | 0.1226 | 0.128 | 0.629 |

> **HitRate@10 = 29.4%**: nearly 1 in 3 users finds a relevant title in their top-10 list.

### A/B Test: ALS (control) vs SASRec (treatment)

| Metric | ALS | SASRec | Lift | p-value | Significant? |
|---|---|---|---|---|---|
| Recall@10 | 0.0529 | 0.0700 | **+32.4%** | 0.038 | ✓ YES |

95% CI for the difference: `[+0.0010, +0.0333]`

**Recommendation: SHIP SASRec** — statistically significant improvement at the 95% confidence level.

### Key observations

- **SASRec wins on relevance** — Sequential modelling of watch history outperforms all other models on Recall@10 and HitRate@10, confirming that *order matters* in streaming consumption patterns.
- **ALS wins on coverage** — ALS surfaces items across 60% of the catalogue vs SASRec's 13%, making ALS a better choice for catalogue discovery and new-release exposure.
- **Popularity is a surprisingly strong baseline** — Competitive on MRR and HitRate, validating that popular content is genuinely relevant to many users. Hard to beat cold-start.
- **Two-Tower underperforms** — With only 20 training epochs on CPU, the neural model needs more compute to match ALS. On GPU with more epochs it would close the gap.
- **Production insight** — A real system would use **SASRec for personalised ranking** + **ALS for catalogue exploration** in a blended ensemble, addressing the relevance-diversity trade-off.

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/tubi-ml-project.git
cd tubi-ml-project
pip install -r requirements.txt
```

> **Python 3.9+** required. GPU optional but recommended for Two-Tower / SASRec.

### 2. Run the full pipeline

```bash
# Download data, train all models, evaluate, generate plots
python scripts/run_pipeline.py
```

This will:
1. Download MovieLens 1M (~6 MB) automatically from GroupLens
2. Preprocess and temporally split the dataset
3. Train Popularity, ALS, Two-Tower, and SASRec
4. Evaluate all models and save metrics to `results/metrics.csv`
5. Run a simulated A/B test
6. Generate plots to `results/plots/`

**Estimated runtime on CPU:**
| Step | Time |
|---|---|
| Download + Preprocess | < 1 min |
| Popularity | < 1 sec |
| ALS (30 iterations) | ~10 sec |
| Two-Tower (20 epochs) | ~10 min |
| SASRec (20 epochs) | ~1 min |
| Evaluation | ~10 sec |

### 3. Skip steps you've already run

```bash
# Skip download and training, only evaluate
python scripts/run_pipeline.py --skip-download --skip-train

# Evaluation only
python scripts/run_pipeline.py --eval-only
```

### 4. Start the recommendation API

```bash
uvicorn serving.api:app --host 0.0.0.0 --port 8000 --reload
```

Then visit **http://localhost:8000/docs** for the interactive API explorer.

#### Example API calls

```bash
# Get top-10 recommendations for user 42
curl http://localhost:8000/recommend/42

# Get recommendations with more diversity
curl "http://localhost:8000/recommend/42?top_k=10&diversity=0.5"

# Find content similar to item 100 (cold-start use case)
curl http://localhost:8000/similar/100

# Batch recommendations
curl -X POST http://localhost:8000/recommend/batch \
     -H "Content-Type: application/json" \
     -d '{"user_ids": [1, 2, 3], "top_k": 10}'
```

---

## Project Structure

```
tubi-ml-project/
├── configs/
│   └── config.yaml              # All hyperparameters in one place
│
├── data/
│   ├── download.py              # Downloads MovieLens 1M
│   └── preprocess.py            # Filtering, ID maps, temporal split
│
├── features/
│   ├── item_features.py         # Genre encoding, year, popularity (20 dims)
│   └── user_features.py         # Demographics + genre affinity (51 dims)
│
├── models/
│   ├── base.py                  # Abstract BaseRecommender interface
│   ├── popularity.py            # Global popularity baseline
│   ├── als.py                   # ALS collaborative filtering (implicit lib)
│   ├── two_tower.py             # Neural Two-Tower with BPR loss
│   └── sasrec.py                # SASRec Transformer (causal attention)
│
├── evaluation/
│   ├── metrics.py               # Recall@K, NDCG@K, MRR, Coverage, Diversity
│   └── ab_test.py               # Welch's t-test A/B test simulation
│
├── serving/
│   ├── retrieval.py             # FAISS index (exact or approximate ANN)
│   └── api.py                   # FastAPI with MMR re-ranking
│
├── scripts/
│   ├── run_pipeline.py          # Master entrypoint
│   ├── train_all.py             # Train all models
│   └── evaluate_all.py          # Evaluate + generate plots
│
└── results/
    ├── metrics.csv              # Full metric table
    ├── ab_test_report.json      # A/B test statistics
    └── plots/                   # Charts (recall curves, bar chart, A/B dist)
```

---

## Design Decisions

### Why temporal splitting?
Random train/test splits cause **temporal leakage** — the model sees future interactions during training. A temporal split mimics real deployment: train on history, predict the future.

### Why BPR loss for Two-Tower?
Bayesian Personalised Ranking optimises the *ranking* of positive items over negatives rather than predicting raw ratings. This maps better to implicit feedback (watch/no-watch) and typically outperforms MSE-based objectives.

### Why causal attention in SASRec?
Users consume content sequentially. Causal masking ensures that the prediction at position `t` uses only information from positions `≤ t`, preventing information leakage from future interactions during training.

### Why MMR re-ranking?
Pure relevance maximisation leads to homogeneous lists (e.g., 10 action movies). **Maximal Marginal Relevance** trades a small relevance loss for significant genre diversity, improving the user experience — especially important for Tubi's large free catalogue.

### Coverage vs. Relevance trade-off
ALS covers 60% of the catalogue; SASRec covers only 13%. In production you'd blend both: SASRec for the personalised ranking signal, ALS to ensure catalogue breadth and new-release exposure.

---

## References

- Hu, Y., Koren, Y., & Volinsky, C. (2008). *Collaborative Filtering for Implicit Feedback Datasets*. ICDM 2008.
- Wang, J., et al. (2018). *Self-Attentive Sequential Recommendation*. ICDM 2018. https://arxiv.org/abs/1808.09781
- Covington, P., Adams, J., & Sargin, E. (2016). *Deep Neural Networks for YouTube Recommendations*. RecSys 2016.
- Carbonell, J., & Goldstein, J. (1998). *The Use of MMR, Diversity-Based Reranking for Reordering Documents*. SIGIR 1998.

---

## Dataset

**MovieLens 1M** (GroupLens Research, University of Minnesota)
- 1,000,209 ratings from 6,040 users on 3,706 movies
- Ratings: 1–5 stars; includes user demographics (age, gender, occupation)
- License: for non-commercial research use only

Dataset source: https://files.grouplens.org/datasets/movielens/ml-1m.zip
