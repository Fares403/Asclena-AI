#!/usr/bin/env python3
"""Build and validate the Asclena patient feature store.

This script creates one row per cleaned ED stay in asclena.patient_feature_store.
It does not train a model. risk_target is populated from the MVP outcome
definition in the feature-store SQL.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING
from urllib.parse import quote_plus

if TYPE_CHECKING:
    import polars as pl
    from sqlalchemy.engine import Connection, Engine


pl = None
create_engine = None
text = None


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
    global create_engine, pl, text

    try:
        import polars as polars_module
        from sqlalchemy import create_engine as sqlalchemy_create_engine
        from sqlalchemy import text as sqlalchemy_text
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Python dependency. Install with: "
            "pip install -r requirements-data-cleaning.txt"
        ) from exc

    pl = polars_module
    create_engine = sqlalchemy_create_engine
    text = sqlalchemy_text


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


def execute_script(conn: "Connection", script_path: Path) -> None:
    conn.exec_driver_sql(script_path.read_text(encoding="utf-8"))


def fetch_one(conn: "Connection", sql: str, params: dict[str, Any] | None = None) -> Any:
    return conn.execute(text(sql), params or {}).scalar_one()


def fetch_dicts(
    conn: "Connection", sql: str, params: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    rows = conn.execute(text(sql), params or {}).mappings().all()
    return [dict(row) for row in rows]


def get_columns(conn: "Connection", schema: str, table: str) -> list[str]:
    rows = fetch_dicts(
        conn,
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema
          AND table_name = :table
        ORDER BY ordinal_position
        """,
        {"schema": schema, "table": table},
    )
    return [row["column_name"] for row in rows]


def validation_report(conn: "Connection", schema: str, run_id: str) -> "pl.DataFrame":
    feature_count = int(
        fetch_one(conn, f"SELECT COUNT(*) FROM {table_ref(schema, 'patient_feature_store')}")
    )
    cleaned_count = int(
        fetch_one(conn, f"SELECT COUNT(*) FROM {table_ref(schema, 'cleaned_ed_stays')}")
    )
    distinct_stay = int(
        fetch_one(
            conn,
            f"SELECT COUNT(DISTINCT stay_id) FROM {table_ref(schema, 'patient_feature_store')}",
        )
    )
    null_ids = int(
        fetch_one(
            conn,
            f"""
            SELECT COUNT(*)
            FROM {table_ref(schema, 'patient_feature_store')}
            WHERE stay_id IS NULL OR subject_id IS NULL
            """,
        )
    )
    orphan_rows = int(
        fetch_one(
            conn,
            f"""
            SELECT COUNT(*)
            FROM {table_ref(schema, 'patient_feature_store')} f
            LEFT JOIN {table_ref(schema, 'cleaned_ed_stays')} e
              ON f.stay_id = e.stay_id
            WHERE e.stay_id IS NULL
            """,
        )
    )
    duplicate_groups = int(
        fetch_one(
            conn,
            f"""
            SELECT COUNT(*)
            FROM (
              SELECT stay_id
              FROM {table_ref(schema, 'patient_feature_store')}
              GROUP BY stay_id
              HAVING COUNT(*) > 1
            ) d
            """,
        )
    )
    null_targets = int(
        fetch_one(
            conn,
            f"""
            SELECT COUNT(*)
            FROM {table_ref(schema, 'patient_feature_store')}
            WHERE risk_target IS NULL
            """,
        )
    )
    positive_targets = int(
        fetch_one(
            conn,
            f"""
            SELECT COUNT(*)
            FROM {table_ref(schema, 'patient_feature_store')}
            WHERE risk_target = 1
            """,
        )
    )
    negative_targets = int(
        fetch_one(
            conn,
            f"""
            SELECT COUNT(*)
            FROM {table_ref(schema, 'patient_feature_store')}
            WHERE risk_target = 0
            """,
        )
    )

    records = [
        {
            "run_id": run_id,
            "check_name": "matches_cleaned_ed_stays_count",
            "value": feature_count,
            "expected_value": cleaned_count,
            "passes": feature_count == cleaned_count,
        },
        {
            "run_id": run_id,
            "check_name": "one_row_per_stay_id",
            "value": distinct_stay,
            "expected_value": feature_count,
            "passes": distinct_stay == feature_count,
        },
        {
            "run_id": run_id,
            "check_name": "no_null_identifiers",
            "value": null_ids,
            "expected_value": 0,
            "passes": null_ids == 0,
        },
        {
            "run_id": run_id,
            "check_name": "no_orphan_feature_rows",
            "value": orphan_rows,
            "expected_value": 0,
            "passes": orphan_rows == 0,
        },
        {
            "run_id": run_id,
            "check_name": "no_duplicate_stay_id_groups",
            "value": duplicate_groups,
            "expected_value": 0,
            "passes": duplicate_groups == 0,
        },
        {
            "run_id": run_id,
            "check_name": "risk_target_has_no_nulls",
            "value": null_targets,
            "expected_value": 0,
            "passes": null_targets == 0,
        },
        {
            "run_id": run_id,
            "check_name": "risk_target_has_positive_class",
            "value": positive_targets,
            "expected_value": 1,
            "passes": positive_targets > 0,
        },
        {
            "run_id": run_id,
            "check_name": "risk_target_has_negative_class",
            "value": negative_targets,
            "expected_value": 1,
            "passes": negative_targets > 0,
        },
    ]
    return pl.DataFrame(records)


def feature_null_report(conn: "Connection", schema: str, run_id: str) -> "pl.DataFrame":
    columns = get_columns(conn, schema, "patient_feature_store")
    row_count = int(
        fetch_one(conn, f"SELECT COUNT(*) FROM {table_ref(schema, 'patient_feature_store')}")
    )
    records = []

    for column in columns:
        null_count = int(
            fetch_one(
                conn,
                f"""
                SELECT COUNT(*)
                FROM {table_ref(schema, 'patient_feature_store')}
                WHERE {quote_ident(column)} IS NULL
                """,
            )
        )
        records.append(
            {
                "run_id": run_id,
                "column_name": column,
                "null_count": null_count,
                "null_percentage": round(100.0 * null_count / row_count, 4)
                if row_count
                else 0.0,
            }
        )

    return pl.DataFrame(records)


def feature_summary(conn: "Connection", schema: str, run_id: str) -> "pl.DataFrame":
    row = fetch_dicts(
        conn,
        f"""
        SELECT
          COUNT(*) AS feature_rows,
          COUNT(*) FILTER (WHERE vital_row_count > 0) AS rows_with_vitals,
          COUNT(*) FILTER (WHERE vital_row_count = 0) AS rows_without_vitals,
          COUNT(*) FILTER (WHERE diagnosis_count > 0) AS rows_with_diagnosis,
          COUNT(*) FILTER (WHERE diagnosis_count = 0) AS rows_without_diagnosis,
          COUNT(*) FILTER (WHERE pyxis_med_count > 0) AS rows_with_pyxis,
          COUNT(*) FILTER (WHERE med_recon_count > 0) AS rows_with_med_recon,
          COUNT(*) FILTER (WHERE risk_target = 1) AS risk_target_positive,
          COUNT(*) FILTER (WHERE risk_target = 0) AS risk_target_negative
        FROM {table_ref(schema, 'patient_feature_store')}
        """,
    )[0]
    row["run_id"] = run_id
    return pl.DataFrame([row])


def write_report(df: "pl.DataFrame", output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    df.write_csv(path)
    return path


def run_pipeline(args: argparse.Namespace) -> None:
    load_runtime_dependencies()

    run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    engine = build_engine(args.config)
    output_dir = args.output_dir / run_id

    create_script = args.sql_dir / "01_create_patient_feature_store.sql"

    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        print("Step 1: creating patient feature store from cleaned tables.")
        execute_script(conn, create_script)

    with engine.connect() as conn:
        print("Step 2: validating patient feature store.")
        validation = validation_report(conn, args.schema, run_id)
        nulls = feature_null_report(conn, args.schema, run_id)
        summary = feature_summary(conn, args.schema, run_id)

    report_paths = [
        write_report(validation, output_dir, "00_feature_store_validation.csv"),
        write_report(summary, output_dir, "01_feature_store_summary.csv"),
        write_report(nulls, output_dir, "02_feature_store_null_report.csv"),
    ]

    failed = validation.filter(pl.col("passes") == False)  # noqa: E712
    print(f"Reports written to: {output_dir}")
    for path in report_paths:
        print(f"- {path.name}")

    if failed.height:
        print("Validation failures detected:")
        print(failed)
        raise SystemExit(1)

    print("Feature store validation passed.")
    print("asclena.patient_feature_store is ready for XGBoost training.")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_sql_dir = repo_root / "sql" / "feature_engineering"
    default_config = repo_root / "configs" / "db_config.yaml"
    default_output = repo_root / "reports" / "feature_engineering"

    parser = argparse.ArgumentParser(
        description="Build the Asclena patient feature store without model training."
    )
    parser.add_argument("--schema", default="asclena")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--sql-dir", type=Path, default=default_sql_dir)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
