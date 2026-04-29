#!/usr/bin/env python3
"""Train XGBoost risk model and save probability predictions to PostgreSQL.

The model learns binary risk_target, then calibrates probabilities on a
validation set. The system output is risk_score:
calibrated_model.predict_proba(X)[:, 1].
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING
from urllib.parse import quote_plus

if TYPE_CHECKING:
    import polars as pl
    from sqlalchemy.engine import Connection, Engine


pl = None
np = None
joblib = None
create_engine = None
text = None
SimpleImputer = None
CalibratedClassifierCV = None
FrozenEstimator = None
train_test_split = None
roc_auc_score = None
average_precision_score = None
accuracy_score = None
precision_recall_fscore_support = None
confusion_matrix = None
brier_score_loss = None
calibration_curve = None
plt = None
XGBClassifier = None


MODEL_FEATURES = [
    "gender_male",
    "gender_female",
    "gender_unknown",
    "triage_temperature",
    "triage_heartrate",
    "triage_resprate",
    "triage_o2sat",
    "triage_sbp",
    "triage_dbp",
    "acuity",
    "triage_shock_index",
    "triage_temperature_missing",
    "triage_heartrate_missing",
    "triage_resprate_missing",
    "triage_o2sat_missing",
    "triage_sbp_missing",
    "triage_dbp_missing",
    "acuity_missing",
    "vital_row_count",
    "temperature_mean",
    "temperature_min",
    "temperature_max",
    "hr_mean",
    "hr_min",
    "hr_max",
    "rr_mean",
    "rr_min",
    "rr_max",
    "spo2_mean",
    "spo2_min",
    "spo2_max",
    "sbp_mean",
    "sbp_min",
    "sbp_max",
    "dbp_mean",
    "dbp_min",
    "dbp_max",
    "shock_index",
    "shock_index_max",
    "hr_slope",
    "bp_slope",
    "tachycardia_count",
    "hypotension_count",
    "hypoxia_count",
    "fever_count",
    "temperature_missing_rate",
    "heartrate_missing_rate",
    "resprate_missing_rate",
    "o2sat_missing_rate",
    "sbp_missing_rate",
    "dbp_missing_rate",
]

LEAKAGE_OR_EXCLUDED_COLUMNS = [
    "stay_id",
    "subject_id",
    "race",
    "arrival_transport",
    "disposition",
    "has_hadm_id",
    "length_of_stay_hours",
    "diagnosis_count",
    "distinct_diagnosis_count",
    "comorbidity_score",
    "pyxis_med_count",
    "pyxis_distinct_med_count",
    "med_recon_count",
    "med_recon_distinct_med_count",
    "medication_intensity_score",
    "risk_target",
]


def quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def table_ref(schema: str, table: str) -> str:
    return f"{quote_ident(schema)}.{quote_ident(table)}"


def parse_simple_yaml(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def load_runtime_dependencies() -> None:
    global CalibratedClassifierCV
    global FrozenEstimator
    global SimpleImputer
    global XGBClassifier
    global accuracy_score
    global average_precision_score
    global brier_score_loss
    global calibration_curve
    global confusion_matrix
    global create_engine
    global joblib
    global np
    global pl
    global plt
    global precision_recall_fscore_support
    global roc_auc_score
    global text
    global train_test_split

    try:
        import joblib as joblib_module
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        import matplotlib as matplotlib_module
        matplotlib_module.use("Agg")
        import matplotlib.pyplot as pyplot_module
        import numpy as numpy_module
        from sklearn.calibration import CalibratedClassifierCV as sklearn_calibrated_classifier_cv
        from sklearn.calibration import calibration_curve as sklearn_calibration_curve
        try:
            from sklearn.frozen import FrozenEstimator as sklearn_frozen_estimator
        except ImportError:
            sklearn_frozen_estimator = None
        import polars as polars_module
        from sklearn.impute import SimpleImputer as sklearn_simple_imputer
        from sklearn.metrics import accuracy_score as sklearn_accuracy_score
        from sklearn.metrics import average_precision_score as sklearn_average_precision_score
        from sklearn.metrics import brier_score_loss as sklearn_brier_score_loss
        from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
        from sklearn.metrics import precision_recall_fscore_support as sklearn_precision_recall_fscore_support
        from sklearn.metrics import roc_auc_score as sklearn_roc_auc_score
        from sklearn.model_selection import train_test_split as sklearn_train_test_split
        from sqlalchemy import create_engine as sqlalchemy_create_engine
        from sqlalchemy import text as sqlalchemy_text
        from xgboost import XGBClassifier as xgboost_classifier
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Python dependency. Install with: "
            "pip install -r requirements-data-cleaning.txt"
        ) from exc

    joblib = joblib_module
    np = numpy_module
    pl = polars_module
    plt = pyplot_module
    CalibratedClassifierCV = sklearn_calibrated_classifier_cv
    FrozenEstimator = sklearn_frozen_estimator
    calibration_curve = sklearn_calibration_curve
    SimpleImputer = sklearn_simple_imputer
    accuracy_score = sklearn_accuracy_score
    average_precision_score = sklearn_average_precision_score
    brier_score_loss = sklearn_brier_score_loss
    confusion_matrix = sklearn_confusion_matrix
    precision_recall_fscore_support = sklearn_precision_recall_fscore_support
    roc_auc_score = sklearn_roc_auc_score
    train_test_split = sklearn_train_test_split
    create_engine = sqlalchemy_create_engine
    text = sqlalchemy_text
    XGBClassifier = xgboost_classifier


def build_engine(config_path: Path) -> "Engine":
    database_url = os.getenv("ASCLENA_DATABASE_URL")
    if database_url:
        return create_engine(database_url)

    config = parse_simple_yaml(config_path)
    host = os.getenv("ASCLENA_DB_HOST", config.get("host", "localhost"))
    port = os.getenv("ASCLENA_DB_PORT", config.get("port", "5432"))
    database = os.getenv("ASCLENA_DB_NAME", config.get("database", "asclena_db"))
    user = os.getenv("ASCLENA_DB_USER", config.get("user", "postgres"))
    password = os.getenv("ASCLENA_DB_PASSWORD", config.get("password", ""))

    url = (
        "postgresql+psycopg2://"
        f"{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{database}"
    )
    return create_engine(url)


def risk_label(score: float) -> str:
    if score >= 0.70:
        return "HIGH"
    if score >= 0.40:
        return "MODERATE"
    return "LOW"


def load_feature_store(conn: "Connection", schema: str) -> "pl.DataFrame":
    selected_columns = ["stay_id", "subject_id", *MODEL_FEATURES, "risk_target"]
    select_sql = ",\n  ".join(
        quote_ident(column) if column in ("stay_id", "subject_id", "risk_target") else f"{quote_ident(column)}::double precision AS {quote_ident(column)}"
        for column in selected_columns
    )
    query = f"""
    SELECT
      {select_sql}
    FROM {table_ref(schema, 'patient_feature_store')}
    WHERE risk_target IS NOT NULL
    """
    return pl.read_database(query=query, connection=conn)


def validate_feature_columns(df: "pl.DataFrame") -> None:
    missing = [column for column in MODEL_FEATURES if column not in df.columns]
    if missing:
        raise SystemExit(f"Missing model features in patient_feature_store: {missing}")


def feature_importance_json(model: Any, limit: int = 20) -> tuple[str, list[dict[str, Any]]]:
    importances = model.feature_importances_
    ranked = sorted(
        zip(MODEL_FEATURES, importances, strict=True),
        key=lambda item: float(item[1]),
        reverse=True,
    )[:limit]
    top_features = [
        {"feature": feature, "importance": round(float(importance), 8)}
        for feature, importance in ranked
    ]
    return json.dumps(top_features), top_features


def build_prefit_calibrated_model(model: Any, method: str, calibration_row_count: int) -> Any:
    if FrozenEstimator is None:
        return CalibratedClassifierCV(
            estimator=model,
            method=method,
            cv="prefit",
        )

    # scikit-learn 1.8 removed cv="prefit"; FrozenEstimator preserves the
    # already-trained XGBoost model and uses the validation rows only for calibration.
    calibration_cv = [(np.array([], dtype=int), np.arange(calibration_row_count))]
    return CalibratedClassifierCV(
        estimator=FrozenEstimator(model),
        method=method,
        cv=calibration_cv,
    )


def validate_training_inputs(target: Any, test_size: float) -> None:
    class_labels, class_counts = np.unique(target, return_counts=True)
    if len(class_labels) < 2:
        raise SystemExit("Training requires both positive and negative risk_target classes.")

    min_class_count = int(class_counts.min())
    if min_class_count < 3:
        raise SystemExit(
            "Training requires at least 3 rows in each class so stratified train/validation/test splits can succeed."
        )

    if not 0 < test_size < 1:
        raise SystemExit("--test-size must be between 0 and 1.")


def create_predictions_table(conn: "Connection", sql_path: Path, schema: str) -> None:
    sql = sql_path.read_text(encoding="utf-8").replace(
        "asclena.risk_predictions",
        table_ref(schema, "risk_predictions"),
    )
    conn.exec_driver_sql(sql)


def save_predictions(
    conn: "Connection",
    schema: str,
    predictions: "pl.DataFrame",
    model_name: str,
    model_version: str,
) -> None:
    raw_conn = conn.connection.driver_connection
    target_table = table_ref(schema, "risk_predictions")
    rows = [
        (
            int(row["stay_id"]),
            int(row["subject_id"]),
            model_name,
            model_version,
            float(row["risk_score"]),
            int(row["predicted_target"]),
            row["risk_label"],
            float(row["threshold_used"]),
            row["top_features"],
        )
        for row in predictions.iter_rows(named=True)
    ]

    with raw_conn.cursor() as cursor:
        cursor.execute(
            f"""
            DELETE FROM {target_table}
            WHERE model_name = %s
              AND model_version = %s
            """,
            (model_name, model_version),
        )
        from psycopg2.extras import execute_values

        execute_values(
            cursor,
            f"""
            INSERT INTO {target_table} (
              stay_id,
              subject_id,
              model_name,
              model_version,
              risk_score,
              predicted_target,
              risk_label,
              threshold_used,
              top_features
            )
            VALUES %s
            """,
            rows,
            template="(%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)",
            page_size=5000,
        )
    raw_conn.commit()


def train_and_evaluate(args: argparse.Namespace) -> None:
    load_runtime_dependencies()

    run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    model_version = args.model_version or run_id
    engine = build_engine(args.config)
    output_dir = args.output_dir / model_version
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    with engine.connect() as conn:
        df = load_feature_store(conn, args.schema)

    validate_feature_columns(df)
    feature_df = df.select(MODEL_FEATURES)
    target = df["risk_target"].to_numpy().astype(int)
    stay_subject = df.select(["stay_id", "subject_id"])
    validate_training_inputs(target, args.test_size)

    X = feature_df.to_numpy()
    indices = np.arange(len(target))
    try:
        # Split into train+validation and test
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=target,
        )

        # Further split train_val into train and validation sets
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=0.2,
            random_state=args.random_state,
            stratify=target[train_val_idx],
        )
    except ValueError as exc:
        raise SystemExit(
            "Unable to create stratified train/validation/test splits. "
            "Increase dataset size, reduce --test-size, or review class balance."
        ) from exc

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = target[train_idx]
    y_test = target[test_idx]
    y_val = target[val_idx]

    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    X_val = X[val_idx]
    X_test_imputed = imputer.transform(X_test)
    X_val_imputed = imputer.transform(X_val)
    X_all_imputed = imputer.transform(X)

    positive_count = int((y_train == 1).sum())
    negative_count = int((y_train == 0).sum())
    scale_pos_weight = negative_count / positive_count if positive_count else 1.0

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        early_stopping_rounds=50,
    )
    model.fit(
        X_train_imputed,
        y_train,
        eval_set=[(X_val_imputed, y_val)],
        verbose=10,
    )

    calibrated_model = build_prefit_calibrated_model(
        model,
        args.calibration_method,
        len(y_val),
    )
    calibrated_model.fit(X_val_imputed, y_val)

    test_risk_scores = calibrated_model.predict_proba(X_test_imputed)[:, 1]
    test_predictions = (test_risk_scores >= args.classification_threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        test_predictions,
        average="binary",
        zero_division=0,
    )
    conf = confusion_matrix(y_test, test_predictions)
    metrics = {
        "model_name": args.model_name,
        "model_version": model_version,
        "feature_count": len(MODEL_FEATURES),
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(val_idx)),
        "test_rows": int(len(test_idx)),
        "positive_train_rows": positive_count,
        "negative_train_rows": negative_count,
        "scale_pos_weight": float(scale_pos_weight),
        "classification_threshold": float(args.classification_threshold),
        "calibration_method": args.calibration_method,
        "best_iteration": int(getattr(model, "best_iteration", args.n_estimators)),
        "roc_auc": float(roc_auc_score(y_test, test_risk_scores)),
        "pr_auc": float(average_precision_score(y_test, test_risk_scores)),
        "brier_score": float(brier_score_loss(y_test, test_risk_scores)),
        "accuracy": float(accuracy_score(y_test, test_predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix_tn": int(conf[0, 0]),
        "confusion_matrix_fp": int(conf[0, 1]),
        "confusion_matrix_fn": int(conf[1, 0]),
        "confusion_matrix_tp": int(conf[1, 1]),
    }

    top_features_json, top_features = feature_importance_json(model)

    prob_true, prob_pred = calibration_curve(
        y_test,
        test_risk_scores,
        n_bins=10,
        strategy="quantile",
    )
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed probability")
    plt.title("Calibration Curve")
    plt.legend()
    plt.savefig(output_dir / "05_calibration_curve.png", dpi=200)
    plt.close()

    all_risk_scores = calibrated_model.predict_proba(X_all_imputed)[:, 1]
    all_predictions = (all_risk_scores >= args.classification_threshold).astype(int)
    prediction_df = stay_subject.with_columns(
        [
            pl.Series("risk_score", all_risk_scores).round(5),
            pl.Series("predicted_target", all_predictions),
            pl.Series("risk_label", [risk_label(float(score)) for score in all_risk_scores]),
            pl.lit(float(args.classification_threshold)).alias("threshold_used"),
            pl.lit(top_features_json).alias("top_features"),
        ]
    )

    artifact = {
        "model": calibrated_model,
        "explanation_model": model,
        "imputer": imputer,
        "feature_names": MODEL_FEATURES,
        "leakage_or_excluded_columns": LEAKAGE_OR_EXCLUDED_COLUMNS,
        "model_name": args.model_name,
        "model_version": model_version,
        "classification_threshold": args.classification_threshold,
        "risk_label_thresholds": {"LOW": [0.0, 0.40], "MODERATE": [0.40, 0.70], "HIGH": [0.70, 1.0]},
        "calibration_method": args.calibration_method,
    }
    model_path = model_dir / f"{args.model_name}_{model_version}.joblib"
    joblib.dump(artifact, model_path)

    pl.DataFrame([metrics]).write_csv(output_dir / "00_evaluation_metrics.csv")
    pl.DataFrame({"feature_name": MODEL_FEATURES}).write_csv(output_dir / "01_model_feature_list.csv")
    pl.DataFrame({"excluded_column": LEAKAGE_OR_EXCLUDED_COLUMNS}).write_csv(
        output_dir / "02_excluded_columns.csv"
    )
    pl.DataFrame(top_features).write_csv(output_dir / "03_top_feature_importance.csv")
    prediction_df.head(1000).write_csv(output_dir / "04_prediction_sample.csv")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if args.save_predictions:
        sql_path = args.sql_dir / "01_create_risk_predictions.sql"
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            create_predictions_table(conn, sql_path, args.schema)
            save_predictions(
                conn,
                args.schema,
                prediction_df,
                args.model_name,
                model_version,
            )

    print(f"Reports written to: {output_dir}")
    print(f"Model artifact written to: {model_path}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"Brier score: {metrics['brier_score']:.4f}")
    print(f"Recall for risk_target=1: {metrics['recall']:.4f}")
    if args.save_predictions:
        print("Predictions saved to asclena.risk_predictions.")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_config = repo_root / "configs" / "db_config.yaml"
    default_output = repo_root / "reports" / "modeling"
    default_model_dir = repo_root / "models"
    default_sql_dir = repo_root / "sql" / "modeling"

    parser = argparse.ArgumentParser(
        description="Train Asclena XGBoost risk model and save probability predictions."
    )
    parser.add_argument("--schema", default="asclena")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--model-dir", type=Path, default=default_model_dir)
    parser.add_argument("--sql-dir", type=Path, default=default_sql_dir)
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
    )
    return parser.parse_args()


if __name__ == "__main__":
    train_and_evaluate(parse_args())
