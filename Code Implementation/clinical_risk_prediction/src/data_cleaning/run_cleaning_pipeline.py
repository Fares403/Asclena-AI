#!/usr/bin/env python3
"""Run the Asclena PostgreSQL clinical data cleaning pipeline.

The pipeline intentionally works on the full raw PostgreSQL dataset:

1. Create safety backups of raw tables.
2. Profile full raw tables before cleaning.
3. Create cleaned_* tables from full raw tables.
4. Profile and validate cleaned tables.
5. Write CSV reports for feature-engineering handoff.
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


RAW_TABLES = [
    "ed_stays",
    "triage",
    "vital_sign",
    "diagnosis",
    "pyxis",
    "med_recon",
]

CLEANED_TABLES = [f"cleaned_{table}" for table in RAW_TABLES]

VITAL_RANGES = {
    "temperature": (90, 110),
    "heartrate": (20, 250),
    "resprate": (5, 80),
    "o2sat": (50, 100),
    "sbp": (50, 300),
    "dbp": (20, 200),
    "acuity": (1, 5),
}

NUMERIC_TYPES = {
    "smallint",
    "integer",
    "bigint",
    "decimal",
    "numeric",
    "real",
    "double precision",
}

BUSINESS_DUP_KEYS = {
    "ed_stays": ["stay_id"],
    "cleaned_ed_stays": ["stay_id"],
    "triage": ["stay_id"],
    "cleaned_triage": ["stay_id"],
    "vital_sign": [
        "stay_id",
        "subject_id",
        "charttime",
        "temperature",
        "heartrate",
        "resprate",
        "o2sat",
        "sbp",
        "dbp",
        "rhythm",
        "pain",
    ],
    "cleaned_vital_sign": [
        "stay_id",
        "subject_id",
        "charttime",
        "temperature",
        "heartrate",
        "resprate",
        "o2sat",
        "sbp",
        "dbp",
        "rhythm",
        "pain",
    ],
    "diagnosis": [
        "stay_id",
        "subject_id",
        "seq_num",
        "icd_code",
        "icd_version",
        "icd_title",
    ],
    "cleaned_diagnosis": [
        "stay_id",
        "subject_id",
        "seq_num",
        "icd_code",
        "icd_version",
        "icd_title",
    ],
    "pyxis": [
        "stay_id",
        "subject_id",
        "charttime",
        "med_rn",
        "name",
        "gsn_rn",
        "gsn",
    ],
    "cleaned_pyxis": [
        "stay_id",
        "subject_id",
        "charttime",
        "med_rn",
        "name",
        "gsn_rn",
        "gsn",
    ],
    "med_recon": [
        "stay_id",
        "subject_id",
        "charttime",
        "name",
        "gsn",
        "ndc",
        "etc_rn",
        "etccode",
        "etcdescription",
    ],
    "cleaned_med_recon": [
        "stay_id",
        "subject_id",
        "charttime",
        "name",
        "gsn",
        "ndc",
        "etc_rn",
        "etccode",
        "etcdescription",
    ],
}


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
    sql = script_path.read_text(encoding="utf-8")
    conn.exec_driver_sql(sql)


def fetch_one(conn: "Connection", sql: str, params: dict[str, Any] | None = None) -> Any:
    return conn.execute(text(sql), params or {}).scalar_one()


def fetch_dicts(
    conn: "Connection", sql: str, params: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    rows = conn.execute(text(sql), params or {}).mappings().all()
    return [dict(row) for row in rows]


def get_columns(conn: "Connection", schema: str, table: str) -> list[dict[str, Any]]:
    return fetch_dicts(
        conn,
        """
        SELECT column_name, data_type, ordinal_position
        FROM information_schema.columns
        WHERE table_schema = :schema
          AND table_name = :table
        ORDER BY ordinal_position
        """,
        {"schema": schema, "table": table},
    )


def get_row_count(conn: "Connection", schema: str, table: str) -> int:
    return int(fetch_one(conn, f"SELECT COUNT(*) FROM {table_ref(schema, table)}"))


def null_count(conn: "Connection", schema: str, table: str, column: str) -> int:
    return int(
        fetch_one(
            conn,
            f"""
            SELECT COUNT(*)
            FROM {table_ref(schema, table)}
            WHERE {quote_ident(column)} IS NULL
            """,
        )
    )


def distinct_count(conn: "Connection", schema: str, table: str, column: str) -> int | None:
    columns = {col["column_name"] for col in get_columns(conn, schema, table)}
    if column not in columns:
        return None
    return int(
        fetch_one(
            conn,
            f"SELECT COUNT(DISTINCT {quote_ident(column)}) FROM {table_ref(schema, table)}",
        )
    )


def duplicate_count(conn: "Connection", schema: str, table: str) -> int:
    available = {col["column_name"] for col in get_columns(conn, schema, table)}
    keys = [key for key in BUSINESS_DUP_KEYS.get(table, []) if key in available]
    if not keys:
        return 0

    key_sql = ", ".join(quote_ident(key) for key in keys)
    return int(
        fetch_one(
            conn,
            f"""
            SELECT COALESCE(SUM(row_count - 1), 0)::bigint
            FROM (
              SELECT COUNT(*) AS row_count
              FROM {table_ref(schema, table)}
              GROUP BY {key_sql}
              HAVING COUNT(*) > 1
            ) d
            """,
        )
    )


def exact_row_duplicate_count(conn: "Connection", schema: str, table: str) -> int:
    columns = [col["column_name"] for col in get_columns(conn, schema, table)]
    if not columns:
        return 0

    column_sql = ", ".join(quote_ident(column) for column in columns)
    return int(
        fetch_one(
            conn,
            f"""
            SELECT COALESCE(SUM(row_count - 1), 0)::bigint
            FROM (
              SELECT COUNT(*) AS row_count
              FROM {table_ref(schema, table)}
              GROUP BY {column_sql}
              HAVING COUNT(*) > 1
            ) d
            """,
        )
    )


def numeric_min_max(
    conn: "Connection", schema: str, table: str, column: str
) -> tuple[Any, Any]:
    row = conn.execute(
        text(
            f"""
            SELECT
              MIN({quote_ident(column)}) AS min_value,
              MAX({quote_ident(column)}) AS max_value
            FROM {table_ref(schema, table)}
            """
        )
    ).mappings().one()
    return row["min_value"], row["max_value"]


def invalid_clinical_count(
    conn: "Connection", schema: str, table: str, column: str
) -> int:
    low, high = VITAL_RANGES[column]
    return int(
        fetch_one(
            conn,
            f"""
            SELECT COUNT(*)
            FROM {table_ref(schema, table)}
            WHERE {quote_ident(column)} IS NOT NULL
              AND ({quote_ident(column)} < :low OR {quote_ident(column)} > :high)
            """,
            {"low": low, "high": high},
        )
    )


def recommended_action(table: str, metric: str, value: int | float | None) -> str:
    if value in (None, 0):
        return "No cleaning action needed for this metric."
    base_table = table.replace("cleaned_", "")
    if metric == "duplicate_count" and base_table == "ed_stays":
        return "Keep one row per stay_id using most non-null values, then latest created_at."
    if metric == "duplicate_count" and base_table == "triage":
        return "Keep earliest created_at per stay_id."
    if metric == "duplicate_count":
        return "Remove duplicate business rows while preserving repeated clinical events."
    if metric.startswith("invalid_clinical_count"):
        return "Set out-of-range clinical values to NULL; feature imputation happens later."
    if metric.startswith("null_count") and base_table == "triage":
        return "Keep NULL vitals and add missing indicator columns."
    if metric.startswith("null_count"):
        return "Keep clinically meaningful NULLs; remove only rows violating table cleaning rules."
    return "Review metric during cleaning validation."


def profile_tables(
    conn: "Connection", schema: str, tables: list[str], phase: str, run_id: str
) -> pl.DataFrame:
    records: list[dict[str, Any]] = []

    for table in tables:
        columns = get_columns(conn, schema, table)
        column_names = [col["column_name"] for col in columns]
        row_count = get_row_count(conn, schema, table)
        dup_count = duplicate_count(conn, schema, table)

        table_metrics = {
            "row_count": row_count,
            "distinct_stay_id": distinct_count(conn, schema, table, "stay_id"),
            "distinct_subject_id": distinct_count(conn, schema, table, "subject_id"),
            "duplicate_count": dup_count,
        }
        for metric, value in table_metrics.items():
            records.append(
                {
                    "run_id": run_id,
                    "phase": phase,
                    "table_name": table,
                    "column_name": None,
                    "metric": metric,
                    "value_num": value,
                    "value_text": None,
                    "recommended_cleaning_action": recommended_action(
                        table, metric, value
                    ),
                }
            )

        for column in columns:
            col_name = column["column_name"]
            count = null_count(conn, schema, table, col_name)
            pct = round((100.0 * count / row_count), 4) if row_count else 0.0
            records.append(
                {
                    "run_id": run_id,
                    "phase": phase,
                    "table_name": table,
                    "column_name": col_name,
                    "metric": "null_count",
                    "value_num": count,
                    "value_text": None,
                    "recommended_cleaning_action": recommended_action(
                        table, "null_count", count
                    ),
                }
            )
            records.append(
                {
                    "run_id": run_id,
                    "phase": phase,
                    "table_name": table,
                    "column_name": col_name,
                    "metric": "null_percentage",
                    "value_num": pct,
                    "value_text": None,
                    "recommended_cleaning_action": recommended_action(
                        table, "null_count", count
                    ),
                }
            )

        for column in columns:
            if column["data_type"] not in NUMERIC_TYPES:
                continue
            col_name = column["column_name"]
            min_value, max_value = numeric_min_max(conn, schema, table, col_name)
            records.append(
                {
                    "run_id": run_id,
                    "phase": phase,
                    "table_name": table,
                    "column_name": col_name,
                    "metric": "numeric_min",
                    "value_num": min_value,
                    "value_text": None,
                    "recommended_cleaning_action": "Use min/max to review clinical plausibility before feature engineering.",
                }
            )
            records.append(
                {
                    "run_id": run_id,
                    "phase": phase,
                    "table_name": table,
                    "column_name": col_name,
                    "metric": "numeric_max",
                    "value_num": max_value,
                    "value_text": None,
                    "recommended_cleaning_action": "Use min/max to review clinical plausibility before feature engineering.",
                }
            )

        if table.endswith("triage") or table.endswith("vital_sign"):
            for col_name in column_names:
                if col_name not in VITAL_RANGES:
                    continue
                count = invalid_clinical_count(conn, schema, table, col_name)
                records.append(
                    {
                        "run_id": run_id,
                        "phase": phase,
                        "table_name": table,
                        "column_name": col_name,
                        "metric": "invalid_clinical_count",
                        "value_num": count,
                        "value_text": None,
                        "recommended_cleaning_action": recommended_action(
                            table, "invalid_clinical_count", count
                        ),
                    }
                )

    return pl.DataFrame(records)


def backup_validation(conn: "Connection", schema: str, run_id: str) -> "pl.DataFrame":
    records = []

    for table in RAW_TABLES:
        backup_table = f"{table}_raw_backup"
        raw_count = get_row_count(conn, schema, table)
        backup_count = get_row_count(conn, schema, backup_table)
        records.append(
            {
                "run_id": run_id,
                "raw_table": table,
                "backup_table": backup_table,
                "raw_row_count": raw_count,
                "backup_row_count": backup_count,
                "counts_match": raw_count == backup_count,
            }
        )

    return pl.DataFrame(records)


def row_removal_summary(conn: "Connection", schema: str, run_id: str) -> "pl.DataFrame":
    records = []
    for raw_table in RAW_TABLES:
        cleaned = f"cleaned_{raw_table}"
        raw_count = get_row_count(conn, schema, raw_table)
        cleaned_count = get_row_count(conn, schema, cleaned)
        records.append(
            {
                "run_id": run_id,
                "source_table": raw_table,
                "cleaned_table": cleaned,
                "raw_rows_before_cleaning": raw_count,
                "rows_after_cleaning": cleaned_count,
                "rows_removed": raw_count - cleaned_count,
                "removed_percentage_of_raw": round(
                    100.0 * (raw_count - cleaned_count) / raw_count, 4
                )
                if raw_count
                else 0.0,
            }
        )
    return pl.DataFrame(records)


def invalid_values_fixed_summary(
    before_df: pl.DataFrame, after_df: pl.DataFrame, run_id: str
) -> pl.DataFrame:
    before = before_df.filter(pl.col("metric") == "invalid_clinical_count")
    after = after_df.filter(pl.col("metric") == "invalid_clinical_count")
    records = []

    for row in before.iter_rows(named=True):
        raw_table = row["table_name"]
        column = row["column_name"]
        after_table = f"cleaned_{raw_table}"
        after_count_df = after.filter(
            (pl.col("table_name") == after_table) & (pl.col("column_name") == column)
        )
        after_count = (
            after_count_df["value_num"][0] if after_count_df.height else None
        )
        records.append(
            {
                "run_id": run_id,
                "table_name": raw_table,
                "column_name": column,
                "invalid_values_before_cleaning": row["value_num"],
                "invalid_values_after_cleaning": after_count,
                "invalid_values_fixed_to_null_or_removed": row["value_num"],
            }
        )
    return pl.DataFrame(records)


def null_handling_summary(run_id: str) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "run_id": run_id,
                "table_name": "triage",
                "rule_summary": "Kept NULL numeric vitals, set invalid values to NULL, and added *_missing indicator columns.",
            },
            {
                "run_id": run_id,
                "table_name": "triage",
                "rule_summary": "Set NULL or empty chiefcomplaint and pain to 'Unknown'; kept missing acuity as NULL.",
            },
            {
                "run_id": run_id,
                "table_name": "vital_sign",
                "rule_summary": "Removed rows where all numeric vital measurements are NULL; no time-series imputation performed.",
            },
            {
                "run_id": run_id,
                "table_name": "vital_sign",
                "rule_summary": "Kept rhythm in cleaned_vital_sign as an observed text column with NULLs preserved; never impute missing rhythm as 'Normal'.",
            },
            {
                "run_id": run_id,
                "table_name": "diagnosis",
                "rule_summary": "Removed rows where both icd_code and icd_title are NULL; missing titles became 'Unknown diagnosis'.",
            },
            {
                "run_id": run_id,
                "table_name": "pyxis",
                "rule_summary": "Removed rows where name, gsn, and med_rn are all NULL; medication names received basic text cleanup only.",
            },
            {
                "run_id": run_id,
                "table_name": "med_recon",
                "rule_summary": "Removed rows where name, gsn, ndc, and etcdescription are all NULL; medication names received basic text cleanup only.",
            },
        ]
    )


def feature_exclusion_manifest(run_id: str) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "run_id": run_id,
                "source_table": "asclena.cleaned_vital_sign",
                "column_name": "rhythm",
                "exclude_from_feature_engineering": True,
                "exclude_from_model_training": True,
                "reason": "High missingness and absence of a reliable replacement; imputing Normal would create fake clinical information.",
                "handling_policy": "Keep for schema consistency and auditability. Do not impute, encode, aggregate, or train on rhythm unless a later clinically validated feature policy is approved.",
            }
        ]
    )


def exact_duplicate_validation(conn: "Connection", schema: str, table: str) -> int:
    return duplicate_count(conn, schema, table)


def cleaned_validation(conn: "Connection", schema: str, run_id: str) -> "pl.DataFrame":
    records: list[dict[str, Any]] = []

    for table in CLEANED_TABLES:
        records.append(
            {
                "run_id": run_id,
                "table_name": table,
                "check_name": "row_count",
                "value": get_row_count(conn, schema, table),
                "passes": True,
            }
        )
        records.append(
            {
                "run_id": run_id,
                "table_name": table,
                "check_name": "distinct_stay_id",
                "value": distinct_count(conn, schema, table, "stay_id"),
                "passes": True,
            }
        )
        null_id_count = int(
            fetch_one(
                conn,
                f"""
                SELECT COUNT(*)
                FROM {table_ref(schema, table)}
                WHERE stay_id IS NULL OR subject_id IS NULL
                """,
            )
        )
        records.append(
            {
                "run_id": run_id,
                "table_name": table,
                "check_name": "no_null_stay_id_or_subject_id",
                "value": null_id_count,
                "passes": null_id_count == 0,
            }
        )

    for table in CLEANED_TABLES:
        dup_count = exact_duplicate_validation(conn, schema, table)
        records.append(
            {
                "run_id": run_id,
                "table_name": table,
                "check_name": "no_duplicate_business_rows",
                "value": dup_count,
                "passes": dup_count == 0,
            }
        )
        exact_dup_count = exact_row_duplicate_count(conn, schema, table)
        records.append(
            {
                "run_id": run_id,
                "table_name": table,
                "check_name": "no_exact_duplicate_rows",
                "value": exact_dup_count,
                "passes": exact_dup_count == 0,
            }
        )

    for table in CLEANED_TABLES:
        if table == "cleaned_ed_stays":
            continue
        orphan_count = int(
            fetch_one(
                conn,
                f"""
                SELECT COUNT(*)
                FROM {table_ref(schema, table)} child
                LEFT JOIN {table_ref(schema, "cleaned_ed_stays")} ed
                  ON child.stay_id = ed.stay_id
                WHERE ed.stay_id IS NULL
                """,
            )
        )
        records.append(
            {
                "run_id": run_id,
                "table_name": table,
                "check_name": "no_orphan_stay_id",
                "value": orphan_count,
                "passes": orphan_count == 0,
            }
        )

    for table in ["cleaned_triage", "cleaned_vital_sign"]:
        columns = {col["column_name"] for col in get_columns(conn, schema, table)}
        for column in columns.intersection(VITAL_RANGES):
            invalid_count = invalid_clinical_count(conn, schema, table, column)
            records.append(
                {
                    "run_id": run_id,
                    "table_name": table,
                    "check_name": f"valid_range_{column}",
                    "value": invalid_count,
                    "passes": invalid_count == 0,
                }
            )

    ed_stays_dups = duplicate_count(conn, schema, "cleaned_ed_stays")
    records.append(
        {
            "run_id": run_id,
            "table_name": "cleaned_ed_stays",
            "check_name": "one_row_per_stay_id",
            "value": ed_stays_dups,
            "passes": ed_stays_dups == 0,
        }
    )

    vital_all_null = int(
        fetch_one(
            conn,
            f"""
            SELECT COUNT(*)
            FROM {table_ref(schema, "cleaned_vital_sign")}
            WHERE temperature IS NULL
              AND heartrate IS NULL
              AND resprate IS NULL
              AND o2sat IS NULL
              AND sbp IS NULL
              AND dbp IS NULL
            """,
        )
    )
    records.append(
        {
            "run_id": run_id,
            "table_name": "cleaned_vital_sign",
            "check_name": "no_rows_with_all_vitals_null",
            "value": vital_all_null,
            "passes": vital_all_null == 0,
        }
    )

    return pl.DataFrame(records)


def write_report(df: pl.DataFrame, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    df.write_csv(path)
    return path


def run_pipeline(args: argparse.Namespace) -> None:
    load_runtime_dependencies()

    run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    engine = build_engine(args.config)
    output_dir = args.output_dir / run_id

    backup_script = args.sql_dir / "01_create_raw_backups.sql"
    clean_script = args.sql_dir / "02_create_cleaned_tables.sql"

    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        print("Step 1: creating full raw-table backups.")
        execute_script(conn, backup_script)

    with engine.connect() as conn:
        print("Step 2: profiling full raw tables before cleaning.")
        backup_checks = backup_validation(conn, args.schema, run_id)
        before_profile = profile_tables(
            conn, args.schema, RAW_TABLES, "before_cleaning", run_id
        )
        backup_report = write_report(
            backup_checks, output_dir, "00_raw_backup_validation.csv"
        )
        write_report(
            before_profile, output_dir, "01_data_quality_before_cleaning.csv"
        )

    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        print("Step 3-4: cleaning full raw tables and saving cleaned tables.")
        execute_script(conn, clean_script)

    with engine.connect() as conn:
        print("Step 5-6: profiling and validating cleaned tables.")
        after_profile = profile_tables(
            conn, args.schema, CLEANED_TABLES, "after_cleaning", run_id
        )
        validation = cleaned_validation(conn, args.schema, run_id)
        removals = row_removal_summary(conn, args.schema, run_id)
        invalid_fixed = invalid_values_fixed_summary(
            before_profile, after_profile, run_id
        )
        null_handling = null_handling_summary(run_id)
        feature_exclusions = feature_exclusion_manifest(run_id)

    report_paths = [
        backup_report,
        write_report(after_profile, output_dir, "02_data_quality_after_cleaning.csv"),
        write_report(removals, output_dir, "03_rows_removed_summary.csv"),
        write_report(null_handling, output_dir, "04_null_handling_summary.csv"),
        write_report(invalid_fixed, output_dir, "05_invalid_values_fixed_summary.csv"),
        write_report(validation, output_dir, "06_cleaned_validation.csv"),
        write_report(feature_exclusions, output_dir, "07_feature_exclusion_manifest.csv"),
    ]

    failed = validation.filter(pl.col("passes") == False)  # noqa: E712
    print(f"Reports written to: {output_dir}")
    for path in report_paths:
        print(f"- {path.name}")

    if failed.height:
        print("Validation failures detected:")
        print(failed)
        raise SystemExit(1)

    print("All cleaned-table validation checks passed.")
    print("Cleaned tables are ready for feature engineering.")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_sql_dir = repo_root / "sql" / "data_cleaning"
    default_config = repo_root / "configs" / "db_config.yaml"
    default_output = repo_root / "reports" / "data_cleaning"

    parser = argparse.ArgumentParser(
        description="Run the Asclena PostgreSQL full-dataset clinical data cleaning pipeline."
    )
    parser.add_argument("--schema", default="asclena")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--sql-dir", type=Path, default=default_sql_dir)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
