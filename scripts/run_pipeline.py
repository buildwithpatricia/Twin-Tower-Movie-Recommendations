"""
Master pipeline script: download → preprocess → train → evaluate → serve.

Usage:
    # Full pipeline (recommended for first run)
    python scripts/run_pipeline.py

    # Skip steps you've already run
    python scripts/run_pipeline.py --skip-download --skip-train

    # Only evaluate
    python scripts/run_pipeline.py --eval-only
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Full RecSys pipeline.")
    parser.add_argument("--config",        default="configs/config.yaml")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip data download if data already exists")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip preprocessing if processed data exists")
    parser.add_argument("--skip-train",    action="store_true",
                        help="Skip model training if models already saved")
    parser.add_argument("--eval-only",     action="store_true",
                        help="Only run evaluation (implies all skips)")
    args = parser.parse_args()

    if args.eval_only:
        args.skip_download   = True
        args.skip_preprocess = True
        args.skip_train      = True

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Step 1: Download ───────────────────────────────────────────────────────
    if not args.skip_download:
        print("\n" + "=" * 60)
        print("STEP 1: Downloading MovieLens 1M dataset")
        print("=" * 60)
        from data.download import download_movielens_1m
        download_movielens_1m(cfg["data"]["data_dir"])
    else:
        print("Skipping download.")

    # ── Step 2: Preprocess ─────────────────────────────────────────────────────
    if not args.skip_preprocess:
        processed_dir = Path(cfg["data"]["processed_dir"])
        if (processed_dir / "train.parquet").exists():
            print("\nProcessed data already exists. Skipping preprocessing.")
            print("(Delete data/processed/ to force re-processing.)")
        else:
            print("\n" + "=" * 60)
            print("STEP 2: Preprocessing")
            print("=" * 60)
            from data.preprocess import run_preprocessing
            run_preprocessing(
                raw_dir=cfg["data"]["data_dir"] + "/ml-1m",
                processed_dir=cfg["data"]["processed_dir"],
                min_user_interactions=cfg["data"]["min_user_interactions"],
                min_item_interactions=cfg["data"]["min_item_interactions"],
                val_ratio=cfg["data"]["val_ratio"],
                test_ratio=cfg["data"]["test_ratio"],
            )
    else:
        print("Skipping preprocessing.")

    # ── Step 3: Train ──────────────────────────────────────────────────────────
    if not args.skip_train:
        print("\n" + "=" * 60)
        print("STEP 3: Training all models")
        print("=" * 60)
        from scripts.train_all import train_all
        train_all(args.config)
    else:
        print("Skipping training.")

    # ── Step 4: Evaluate ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Evaluating all models")
    print("=" * 60)
    from scripts.evaluate_all import evaluate_all
    results, ab_result = evaluate_all(args.config)

    # ── Done ───────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
    print(f"  Metrics CSV  : {cfg['output']['results_dir']}/metrics.csv")
    print(f"  Plots        : {cfg['output']['plots_dir']}/")
    print(f"  A/B Report   : {cfg['output']['results_dir']}/ab_test_report.json")
    print(f"  Models       : {cfg['output']['model_dir']}/")
    print()
    print("To start the recommendation API:")
    print("  uvicorn serving.api:app --host 0.0.0.0 --port 8000 --reload")
    print()
    print("Then visit: http://localhost:8000/docs")


if __name__ == "__main__":
    main()
