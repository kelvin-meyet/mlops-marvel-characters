"""Tests for parsing the serving endpoint's inference table."""

import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from marvel_characters.monitoring import (  # noqa: E402
    MODEL_ID,
    RECORD_SCHEMA,
    build_baseline,
    parse_inference_table,
    select_new_requests,
)


def _character(gender: str) -> dict:
    """Build one request record with all 10 features."""
    return {
        "Height": 180.0,
        "Weight": 80.0,
        "Universe": "Earth-616",
        "Identity": "Public",
        "Gender": gender,
        "Marital_Status": "Single",
        "Teams": 1,
        "Origin": "Human",
        "Magic": 0,
        "Mutant": 0,
    }


def test_parse_inference_table_pairs_each_record_with_its_prediction(spark_session: SparkSession) -> None:
    """A request with two characters yields two rows, each with its own prediction mapped to 1/0."""
    if isinstance(spark_session, MagicMock):
        pytest.skip("Needs a real local SparkSession")

    # Built from literals inside Spark (no Python worker), which also works for local Spark on Windows
    inf_table = spark_session.range(1).select(
        F.lit("req-1").alias("databricks_request_id"),
        F.lit(datetime(2026, 9, 28, 12, 0, 0)).alias("request_time"),
        F.lit(25).alias("execution_duration_ms"),
        F.lit(json.dumps({"dataframe_records": [_character("Male"), _character("Female")]})).alias("request"),
        # Response format of MarvelModelWrapper (custom pyfunc model)
        F.lit(json.dumps({"predictions": {"Survival prediction": ["alive", "dead"]}})).alias("response"),
    )

    result = parse_inference_table(inf_table).orderBy("Gender").collect()

    assert [(r["Gender"], r["prediction"]) for r in result] == [("Female", 0), ("Male", 1)]
    assert all(r["model_name"] == MODEL_ID for r in result)


def test_select_new_requests_skips_requests_already_monitored(spark_session: SparkSession) -> None:
    """Rerunning the monitoring job must not append a request twice."""
    if isinstance(spark_session, MagicMock):
        pytest.skip("Needs a real local SparkSession")

    inf_table = spark_session.range(3).select(F.concat(F.lit("req-"), F.col("id")).alias("databricks_request_id"))
    # req-0 was appended by an earlier run (a request with two characters gives two rows)
    monitored = spark_session.range(2).select(F.lit("req-0").alias("databricks_request_id"))

    result = select_new_requests(inf_table, monitored).orderBy("databricks_request_id").collect()

    assert [r["databricks_request_id"] for r in result] == ["req-1", "req-2"]


def test_build_baseline_matches_monitoring_columns(spark_session: SparkSession) -> None:
    """The baseline has the monitoring table's feature types and model id, and drops target and ids."""
    if isinstance(spark_session, MagicMock):
        pytest.skip("Needs a real local SparkSession")

    train_set = spark_session.range(1).select(
        F.lit(180.0).alias("Height"),
        F.lit(80.0).alias("Weight"),
        F.lit("Earth-616").alias("Universe"),
        F.lit("Public").alias("Identity"),
        F.lit("Male").alias("Gender"),
        F.lit("Single").alias("Marital_Status"),
        F.lit(1).cast("bigint").alias("Teams"),
        F.lit("Human").alias("Origin"),
        F.lit(0).cast("bigint").alias("Magic"),
        F.lit(1).cast("bigint").alias("Mutant"),
        F.lit(1).cast("bigint").alias("Alive"),
        F.lit("42").alias("Id"),
    )

    baseline = build_baseline(train_set, target="Alive")
    row = baseline.first()

    assert [(f.name, f.dataType) for f in baseline.schema.fields[:-2]] == [
        (f.name, f.dataType) for f in RECORD_SCHEMA.fields
    ]
    assert baseline.columns[-2:] == ["prediction", "model_name"]
    assert baseline.schema["prediction"].dataType.simpleString() == "int"  # same type as model_monitoring
    assert (row["Teams"], row["Magic"], row["Mutant"], row["prediction"], row["model_name"]) == (
        "1",
        "0",
        "1",
        1,
        MODEL_ID,
    )
