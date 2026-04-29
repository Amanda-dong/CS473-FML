"""Command-line runner for causal uplift evaluation with temporal backtesting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.validation.causal import (
    CausalMLConfig,
    export_fold_manifest,
    load_causal_frame,
    run_causal_temporal_backtest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run temporal causal uplift evaluation and optional MLflow logging."
    )
    parser.add_argument("--dataset", required=True, help="Input CSV or parquet file.")
    parser.add_argument("--time-col", required=True, help="Time column for splits.")
    parser.add_argument("--treatment-col", required=True, help="Binary treatment column.")
    parser.add_argument("--outcome-col", required=True, help="Outcome column.")
    parser.add_argument(
        "--feature-cols",
        required=True,
        nargs="+",
        help="Feature columns used by the uplift and propensity models.",
    )
    parser.add_argument("--propensity-col", default=None)
    parser.add_argument("--model-type", default="t_learner_gbr")
    parser.add_argument("--feature-set-version", default="v1")
    parser.add_argument("--treatment-definition", default="model_driven_action")
    parser.add_argument("--outcome-definition", default="outcome")
    parser.add_argument("--min-train-periods", type=int, default=3)
    parser.add_argument("--test-size", type=int, default=1)
    parser.add_argument("--window-type", choices=["expanding", "rolling"], default="expanding")
    parser.add_argument("--top-fraction", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--experiment-name", default="causal_uplift_model_v1")
    parser.add_argument(
        "--output-dir",
        default="data/processed/causal_uplift_model_v1",
        help="Directory for plots, reports, models, and summaries.",
    )
    parser.add_argument("--mlflow-tracking-uri", default=None)
    parser.add_argument(
        "--baseline-policy",
        choices=["no_treatment", "treat_all", "historical"],
        default="no_treatment",
    )
    parser.add_argument(
        "--skip-sensitivity-analysis",
        action="store_true",
        help="Disable optional sensitivity analysis artifact generation.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = CausalMLConfig(
        time_col=args.time_col,
        treatment_col=args.treatment_col,
        outcome_col=args.outcome_col,
        feature_cols=list(args.feature_cols),
        propensity_col=args.propensity_col,
        model_type=args.model_type,
        feature_set_version=args.feature_set_version,
        treatment_definition=args.treatment_definition,
        outcome_definition=args.outcome_definition,
        min_train_periods=args.min_train_periods,
        test_size=args.test_size,
        window_type=args.window_type,
        top_fraction=args.top_fraction,
        random_state=args.random_state,
        experiment_name=args.experiment_name,
        output_dir=args.output_dir,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        baseline_policy=args.baseline_policy,
        perform_sensitivity_analysis=not args.skip_sensitivity_analysis,
    )

    frame = load_causal_frame(args.dataset, time_col=config.time_col)
    summary, _folds = run_causal_temporal_backtest(frame, config)
    manifest_path = export_fold_manifest(config, summary)

    console_summary = {
        "rows": int(len(frame)),
        "splits": int(len(summary)),
        "output_dir": str(Path(config.output_dir).resolve()),
        "manifest": str(manifest_path.resolve()),
    }
    print(json.dumps(console_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
