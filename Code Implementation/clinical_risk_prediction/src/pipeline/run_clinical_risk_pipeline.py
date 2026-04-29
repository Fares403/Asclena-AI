#!/usr/bin/env python3
"""Run the end-to-end Asclena clinical risk pipeline with explicit stages."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data_cleaning.run_cleaning_pipeline import run_pipeline as run_cleaning_pipeline
from src.feature_engineering.run_feature_engineering import run_pipeline as run_feature_pipeline
from src.modeling.train_xgboost_risk_model import train_and_evaluate


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_paths() -> dict[str, Path]:
    root = repo_root()
    return {
        "config": root / "configs" / "db_config.yaml",
        "data_cleaning_sql": root / "sql" / "data_cleaning",
        "feature_sql": root / "sql" / "feature_engineering",
        "modeling_sql": root / "sql" / "modeling",
        "data_cleaning_reports": root / "reports" / "data_cleaning",
        "feature_reports": root / "reports" / "feature_engineering",
        "modeling_reports": root / "reports" / "modeling",
        "model_dir": root / "models",
    }


def run_data_cleaning_stage(args: argparse.Namespace) -> None:
    print("Stage 1/3: data cleaning")
    run_cleaning_pipeline(
        argparse.Namespace(
            schema=args.schema,
            config=args.config,
            sql_dir=args.data_cleaning_sql_dir,
            output_dir=args.data_cleaning_output_dir,
        )
    )


def run_feature_engineering_stage(args: argparse.Namespace) -> None:
    print("Stage 2/3: feature engineering")
    run_feature_pipeline(
        argparse.Namespace(
            schema=args.schema,
            config=args.config,
            sql_dir=args.feature_sql_dir,
            output_dir=args.feature_output_dir,
        )
    )


def run_model_training_stage(args: argparse.Namespace) -> None:
    print("Stage 3/3: model training")
    train_and_evaluate(
        argparse.Namespace(
            schema=args.schema,
            config=args.config,
            output_dir=args.modeling_output_dir,
            model_dir=args.model_dir,
            sql_dir=args.modeling_sql_dir,
            model_name=args.model_name,
            model_version=args.model_version,
            test_size=args.test_size,
            random_state=args.random_state,
            classification_threshold=args.classification_threshold,
            calibration_method=args.calibration_method,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            n_jobs=args.n_jobs,
            save_predictions=args.save_predictions,
        )
    )


def parse_args() -> argparse.Namespace:
    paths = default_paths()

    parser = argparse.ArgumentParser(
        description=(
            "Run the Asclena clinical risk pipeline from raw PostgreSQL data "
            "through feature engineering and XGBoost model training."
        )
    )
    parser.add_argument(
        "--stage",
        choices=["full", "clean", "features", "train"],
        default="full",
        help="Pipeline stage to execute.",
    )
    parser.add_argument("--schema", default="asclena")
    parser.add_argument("--config", type=Path, default=paths["config"])

    parser.add_argument("--data-cleaning-sql-dir", type=Path, default=paths["data_cleaning_sql"])
    parser.add_argument("--feature-sql-dir", type=Path, default=paths["feature_sql"])
    parser.add_argument("--modeling-sql-dir", type=Path, default=paths["modeling_sql"])

    parser.add_argument("--data-cleaning-output-dir", type=Path, default=paths["data_cleaning_reports"])
    parser.add_argument("--feature-output-dir", type=Path, default=paths["feature_reports"])
    parser.add_argument("--modeling-output-dir", type=Path, default=paths["modeling_reports"])
    parser.add_argument("--model-dir", type=Path, default=paths["model_dir"])

    parser.add_argument("--model-name", default="asclena_xgboost_risk")
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--classification-threshold", type=float, default=0.40)
    parser.add_argument(
        "--calibration-method",
        choices=["isotonic", "sigmoid"],
        default="isotonic",
        help="Probability calibration method fit on the validation set.",
    )
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--save-predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist training-set predictions to PostgreSQL after model training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.stage == "clean":
        run_data_cleaning_stage(args)
        return

    if args.stage == "features":
        run_feature_engineering_stage(args)
        return

    if args.stage == "train":
        run_model_training_stage(args)
        return

    run_data_cleaning_stage(args)
    run_feature_engineering_stage(args)
    run_model_training_stage(args)
    print("End-to-end pipeline completed successfully.")


if __name__ == "__main__":
    main()
