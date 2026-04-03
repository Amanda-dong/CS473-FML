"""Processed-data preflight checks before embeddings or model training."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from src.config import get_settings
from src.data.quality import (
    prepare_embedding_corpus,
    prepare_survival_history,
    prepare_training_frame,
    validate_dataset_contract,
)
from src.data.registry import DATASET_REGISTRY
from src.models.survival_model import build_real_restaurant_history


@dataclass(frozen=True)
class PreflightCheck:
    """One readiness check for the training and embedding pipeline."""

    name: str
    passed: bool
    message: str
    details: dict[str, int | float | str] = field(default_factory=dict)


@dataclass(frozen=True)
class PreflightReport:
    """Collection of preflight checks with a repo-friendly pass/fail summary."""

    checks: tuple[PreflightCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failed_checks(self) -> tuple[PreflightCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "checks": [asdict(check) for check in self.checks],
        }


def assess_embedding_readiness(
    reviews_df: pd.DataFrame,
    *,
    min_rows: int = 100,
) -> PreflightCheck:
    """Check whether a review corpus is ready for embedding generation."""

    try:
        cleaned, report = prepare_embedding_corpus(reviews_df)
    except ValueError as exc:
        return PreflightCheck(
            name="embedding_corpus",
            passed=False,
            message=str(exc),
        )

    unique_restaurants = int(cleaned["restaurant_id"].nunique()) if "restaurant_id" in cleaned.columns else 0
    passed = len(cleaned) >= min_rows
    return PreflightCheck(
        name="embedding_corpus",
        passed=passed,
        message=(
            "Embedding corpus ready."
            if passed
            else f"Need at least {min_rows} cleaned review rows before generating embeddings."
        ),
        details={
            "input_rows": report.input_rows,
            "clean_rows": report.output_rows,
            "dropped_rows": report.dropped_rows,
            "unique_restaurants": unique_restaurants,
        },
    )


def assess_scoring_training_readiness(
    feature_matrix: pd.DataFrame,
    *,
    min_rows: int = 50,
    min_zones: int = 5,
) -> PreflightCheck:
    """Check whether the scoring feature matrix is cluster-training ready."""

    try:
        cleaned, report = prepare_training_frame(feature_matrix, target_col="target")
    except ValueError as exc:
        return PreflightCheck(
            name="scoring_training",
            passed=False,
            message=str(exc),
        )

    zone_count = int(cleaned["zone_id"].nunique()) if "zone_id" in cleaned.columns else 0
    passed = len(cleaned) >= min_rows and zone_count >= min_zones
    return PreflightCheck(
        name="scoring_training",
        passed=passed,
        message=(
            "Scoring feature matrix ready."
            if passed
            else f"Need at least {min_rows} rows across {min_zones} zones for stable scoring-model training."
        ),
        details={
            "input_rows": report.input_rows,
            "clean_rows": report.output_rows,
            "dropped_rows": report.dropped_rows,
            "zone_count": zone_count,
        },
    )


def assess_survival_training_readiness(
    history: pd.DataFrame,
    *,
    min_rows: int = 50,
    min_events: int = 10,
) -> PreflightCheck:
    """Check whether the survival history is fit for model training."""

    try:
        cleaned, report = prepare_survival_history(history)
    except ValueError as exc:
        return PreflightCheck(
            name="survival_training",
            passed=False,
            message=str(exc),
        )

    event_count = int(cleaned["event_observed"].sum()) if "event_observed" in cleaned.columns else 0
    passed = len(cleaned) >= min_rows and event_count >= min_events
    return PreflightCheck(
        name="survival_training",
        passed=passed,
        message=(
            "Survival history ready."
            if passed
            else f"Need at least {min_rows} restaurant histories and {min_events} observed closures."
        ),
        details={
            "input_rows": report.input_rows,
            "clean_rows": report.output_rows,
            "dropped_rows": report.dropped_rows,
            "event_count": event_count,
        },
    )


def run_processed_data_preflight(
    processed_dir: str | Path | None = None,
    *,
    min_scoring_rows: int = 50,
    min_scoring_zones: int = 5,
    min_embedding_rows: int = 100,
    min_survival_rows: int = 50,
    min_survival_events: int = 10,
) -> PreflightReport:
    """Validate the processed artifacts needed for embeddings and model training."""

    settings = get_settings()
    base_dir = Path(processed_dir) if processed_dir is not None else settings.processed_dir

    checks: list[PreflightCheck] = []

    feature_matrix_path = base_dir / "feature_matrix.parquet"
    if not feature_matrix_path.exists():
        checks.append(
            PreflightCheck(
                name="scoring_training",
                passed=False,
                message=f"Missing required artifact: {feature_matrix_path}",
            )
        )
    else:
        try:
            feature_matrix = pd.read_parquet(feature_matrix_path)
            checks.append(
                assess_scoring_training_readiness(
                    feature_matrix,
                    min_rows=min_scoring_rows,
                    min_zones=min_scoring_zones,
                )
            )
        except Exception as exc:
            checks.append(
                PreflightCheck(
                    name="scoring_training",
                    passed=False,
                    message=f"Could not validate {feature_matrix_path}: {exc}",
                )
            )

    licenses_path = base_dir / "licenses.parquet"
    inspections_path = base_dir / "inspections.parquet"
    if not licenses_path.exists() or not inspections_path.exists():
        checks.append(
            PreflightCheck(
                name="survival_training",
                passed=False,
                message=(
                    "Missing required survival artifacts: "
                    f"{licenses_path.name if not licenses_path.exists() else ''} "
                    f"{inspections_path.name if not inspections_path.exists() else ''}"
                ).strip(),
            )
        )
    else:
        try:
            licenses = pd.read_parquet(licenses_path)
            inspections = pd.read_parquet(inspections_path)
            validate_dataset_contract(licenses, DATASET_REGISTRY["licenses"])
            validate_dataset_contract(inspections, DATASET_REGISTRY["inspections"])
            zone_features_path = base_dir / "zone_features.parquet"
            zone_features = pd.read_parquet(zone_features_path) if zone_features_path.exists() else None
            history = build_real_restaurant_history(licenses, inspections, zone_features)
            checks.append(
                assess_survival_training_readiness(
                    history,
                    min_rows=min_survival_rows,
                    min_events=min_survival_events,
                )
            )
        except Exception as exc:
            checks.append(
                PreflightCheck(
                    name="survival_training",
                    passed=False,
                    message=f"Could not validate survival artifacts under {base_dir}: {exc}",
                )
            )

    review_path = next(
        (
            candidate
            for candidate in (base_dir / "yelp.parquet", base_dir / "reviews.parquet")
            if candidate.exists()
        ),
        None,
    )
    if review_path is None:
        checks.append(
            PreflightCheck(
                name="embedding_corpus",
                passed=False,
                message=f"Missing required embedding artifact under {base_dir}: yelp.parquet or reviews.parquet",
            )
        )
    else:
        try:
            reviews = pd.read_parquet(review_path)
            checks.append(
                assess_embedding_readiness(
                    reviews,
                    min_rows=min_embedding_rows,
                )
            )
        except Exception as exc:
            checks.append(
                PreflightCheck(
                    name="embedding_corpus",
                    passed=False,
                    message=f"Could not validate {review_path}: {exc}",
                )
            )

    return PreflightReport(checks=tuple(checks))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate processed data before embeddings or model training.")
    parser.add_argument("--processed-dir", default=None, help="Directory containing processed parquet artifacts.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for repo-local data preflight checks."""

    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    report = run_processed_data_preflight(processed_dir=args.processed_dir)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        overall = "PASS" if report.passed else "FAIL"
        print(f"Processed-data preflight: {overall}")
        for check in report.checks:
            status = "PASS" if check.passed else "FAIL"
            print(f"- {status} {check.name}: {check.message}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
