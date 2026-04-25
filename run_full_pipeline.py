"""End-to-end pipeline: ETL → feature matrix → ground truth → model training.

Usage
-----
    uv run python -m run_full_pipeline              # full run
    uv run python -m run_full_pipeline --etl-only   # ETL + feature matrix only
    uv run python -m run_full_pipeline --train-only # train from existing parquets
    uv run python -m run_full_pipeline --limit 5000 # cap ETL row count
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("data/models")


def run_etl_stage(limit: int) -> dict[str, pd.DataFrame]:
    from src.data.etl_runner import run_all_etl

    logger.info("=== Stage 1: ETL (limit=%d) ===", limit)
    results, status = run_all_etl(limit=limit)

    ok = sum(1 for v in status.values() if v == "ok")
    failed = [k for k, v in status.items() if v == "failed"]
    logger.info(
        "ETL complete: %d ok, %d failed, %d empty/skipped",
        ok,
        len(failed),
        len(status) - ok - len(failed),
    )
    if failed:
        logger.warning("Failed datasets: %s", failed)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        if not df.empty:
            path = PROCESSED_DIR / f"{name}.parquet"
            df.to_parquet(path, index=False)
            logger.info("  Saved %s → %s (%d rows)", name, path, len(df))

    return results


def build_feature_matrix_stage(etl_outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    from src.features.feature_matrix import build_zone_year_matrix
    from src.features.ground_truth import build_ground_truth

    logger.info("=== Stage 2: Feature matrix ===")
    features = build_zone_year_matrix(etl_outputs)
    logger.info(
        "Zone-year matrix: %d rows × %d cols", len(features), len(features.columns)
    )

    licenses_df = etl_outputs.get("licenses", pd.DataFrame())
    yelp_df = etl_outputs.get("yelp", pd.DataFrame())
    inspections_df = etl_outputs.get("inspections", pd.DataFrame())

    gt = build_ground_truth(licenses_df, yelp_df, inspections_df)
    logger.info("Ground truth: %d rows", len(gt))

    if gt.empty or features.empty:
        logger.warning(
            "Ground truth or features empty — saving features without target"
        )
        matrix = features
    else:
        matrix = features.merge(
            gt[["zone_id", "time_key", "y_composite", "label_quality"]],
            on=["zone_id", "time_key"],
            how="left",
        )
        matrix = matrix.rename(columns={"y_composite": "target"})
        logger.info(
            "Merged matrix: %d rows × %d cols (target coverage: %.1f%%)",
            len(matrix),
            len(matrix.columns),
            100 * matrix["target"].notna().mean()
            if "target" in matrix.columns
            else 0.0,
        )

    out_path = PROCESSED_DIR / "feature_matrix.parquet"
    matrix.to_parquet(out_path, index=False)
    logger.info("Feature matrix saved → %s", out_path)
    return matrix


def load_etl_from_disk() -> dict[str, pd.DataFrame]:
    """Load ETL parquets already on disk (for --train-only mode)."""
    results: dict[str, pd.DataFrame] = {}
    for path in PROCESSED_DIR.glob("*.parquet"):
        name = path.stem
        if name == "feature_matrix":
            continue
        results[name] = pd.read_parquet(path)
        logger.info("Loaded %s from %s (%d rows)", name, path, len(results[name]))
    return results


def train_survival_stage() -> None:
    logger.info("=== Stage 3a: Survival model ===")
    licenses_path = PROCESSED_DIR / "licenses.parquet"
    inspections_path = PROCESSED_DIR / "inspections.parquet"
    if not licenses_path.exists() or not inspections_path.exists():
        logger.warning(
            "Skipping survival training — %s or %s not found",
            licenses_path,
            inspections_path,
        )
        return
    from src.models.train_survival import train_and_evaluate as train_surv

    train_surv()


def train_scoring_stage() -> None:
    logger.info("=== Stage 3b: Scoring model ===")
    fm_path = PROCESSED_DIR / "feature_matrix.parquet"
    if not fm_path.exists():
        logger.warning("Skipping scoring training — %s not found", fm_path)
        return
    matrix = pd.read_parquet(fm_path)
    if "target" not in matrix.columns or matrix["target"].notna().sum() < 10:
        logger.warning(
            "feature_matrix.parquet has <10 labeled rows — skipping scoring training"
        )
        return
    from src.models.train_scoring import train_and_evaluate as train_score

    train_score()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NYC Restaurant Intelligence full pipeline"
    )
    parser.add_argument(
        "--etl-only",
        action="store_true",
        help="Run ETL + feature matrix, skip training",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Train from existing parquets, skip ETL",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50000,
        help="Row limit per ETL module (default: 50000)",
    )
    args = parser.parse_args()

    if args.train_only:
        train_survival_stage()
        train_scoring_stage()
        return

    etl_outputs = run_etl_stage(args.limit)
    matrix = build_feature_matrix_stage(etl_outputs)

    if not args.etl_only:
        train_survival_stage()
        train_scoring_stage()

    summary: dict[str, tuple[int, int]] = {
        f"{name}.parquet": (len(df), len(df.columns))
        for name, df in etl_outputs.items()
        if not df.empty
    }
    summary["feature_matrix.parquet"] = (len(matrix), len(matrix.columns))

    logger.info("=== Pipeline complete ===")
    logger.info("Artifacts in %s/", PROCESSED_DIR)
    for fname in sorted(summary):
        rows, cols = summary[fname]
        logger.info("  %-40s %d rows × %d cols", fname, rows, cols)
    logger.info("Models in %s/", MODEL_DIR)
    for path in sorted(MODEL_DIR.glob("*.joblib")):
        logger.info("  %s", path.name)


if __name__ == "__main__":
    main()
