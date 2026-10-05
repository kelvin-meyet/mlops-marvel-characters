"""Model monitoring module for Marvel characters."""

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    MonitorInferenceLogProblemType,
)
from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DoubleType, StringType, StructField, StructType

from marvel_characters.config import ProjectConfig

# Value of the monitor's model_id_col, shared by the monitoring and baseline tables
MODEL_ID = "marvel-characters-model-fe"

# The 10 features of one request record, with the types used in model_monitoring and its baseline
RECORD_SCHEMA = StructType(
    [
        StructField("Height", DoubleType(), True),
        StructField("Weight", DoubleType(), True),
        StructField("Universe", StringType(), True),
        StructField("Identity", StringType(), True),
        StructField("Gender", StringType(), True),
        StructField("Marital_Status", StringType(), True),
        StructField("Teams", StringType(), True),
        StructField("Origin", StringType(), True),
        StructField("Magic", StringType(), True),
        StructField("Mutant", StringType(), True),
    ]
)


def create_or_refresh_monitoring(config: ProjectConfig, spark: SparkSession, workspace: WorkspaceClient) -> None:
    """Append new inference requests to model_monitoring and create or refresh its Lakehouse monitor.

    Only requests not yet in model_monitoring are appended, so the job can run any number of times.
    Drift is measured against model_monitoring_baseline, which is rebuilt from train_set on every run.

    :param config: Configuration object containing catalog and schema names.
    :param spark: Spark session used for executing SQL queries and transformations.
    :param workspace: Workspace object used for managing quality monitors.
    :raises ValueError: If the endpoint has not logged any requests yet
    """
    schema = f"{config.catalog_name}.{config.schema_name}"
    monitoring_table = f"{schema}.model_monitoring"
    baseline_table = f"{schema}.model_monitoring_baseline"

    inf_table = spark.table(f"{schema}.custom_model_payload")
    if inf_table.isEmpty():
        raise ValueError(
            f"No requests in {schema}.custom_model_payload. Send traffic to the serving endpoint, and check that "
            "it was created with inference-table logging (see ModelServing.telemetry_config)."
        )

    if spark.catalog.tableExists(monitoring_table):
        inf_table = select_new_requests(inf_table, spark.table(monitoring_table))

    df_new = parse_inference_table(inf_table)
    new_count = df_new.count()
    df_valid = df_new.dropna(subset=["prediction"])
    valid_count = df_valid.count()
    if valid_count < new_count:
        logger.warning(f"Skipping {new_count - valid_count} records without a readable prediction")
    logger.info(f"New records to append to {monitoring_table}: {valid_count}")

    if valid_count:
        df_valid.write.format("delta").mode("append").saveAsTable(monitoring_table)
    elif not spark.catalog.tableExists(monitoring_table):
        raise ValueError(f"No valid records to create {monitoring_table} from; check the logged responses.")

    build_baseline(spark.table(f"{schema}.train_set"), target=config.target).write.format("delta").mode(
        "overwrite"
    ).option("overwriteSchema", "true").saveAsTable(baseline_table)

    try:
        monitor = workspace.quality_monitors.get(monitoring_table)
    except NotFound:
        create_monitoring_table(config=config, spark=spark, workspace=workspace)
        return

    baseline_added = monitor.baseline_table_name != baseline_table
    if baseline_added:
        # Monitors created before the baseline existed: add it (update replaces the whole configuration)
        workspace.quality_monitors.update(
            table_name=monitoring_table,
            output_schema_name=schema,
            baseline_table_name=baseline_table,
            inference_log=inference_log_config(),
        )
        logger.info(f"Baseline {baseline_table} added to the monitor.")

    if valid_count or baseline_added:
        workspace.quality_monitors.run_refresh(table_name=monitoring_table)
        logger.info("Lakehouse monitor refresh started.")
    else:
        logger.info("No new records since the last run; monitor not refreshed.")


def select_new_requests(inf_table: DataFrame, monitored: DataFrame) -> DataFrame:
    """Keep only the inference-table rows whose request is not in the monitoring table yet.

    :param inf_table: Rows of the endpoint's inference table
    :param monitored: The monitoring table (needs databricks_request_id)
    :return: Inference-table rows with a databricks_request_id not present in monitored
    """
    return inf_table.join(monitored.select("databricks_request_id").distinct(), "databricks_request_id", "left_anti")


def build_baseline(train_set: DataFrame, target: str) -> DataFrame:
    """Turn the training data into a drift baseline with the monitoring table's columns and types.

    The monitor requires a prediction column in the baseline. The true label is used for it, so prediction drift
    compares the served predictions with the alive/dead split the model was trained on.

    :param train_set: The model's training data (the 10 features plus target and ids)
    :param target: Name of the label column (1 = alive, 0 = dead)
    :return: DataFrame with the 10 features, cast like model_monitoring, prediction and model_name
    """
    return train_set.select(
        *[F.col(field.name).cast(field.dataType).alias(field.name) for field in RECORD_SCHEMA.fields],
        F.col(target).cast("int").alias("prediction"),
        F.lit(MODEL_ID).alias("model_name"),
    )


def parse_inference_table(inf_table: DataFrame) -> DataFrame:
    """Turn raw inference-table rows into one row per scored character.

    Each logged request can hold several records; every record is paired with its own prediction
    ("alive" -> 1, "dead" -> 0), since the custom model answers {"predictions": {"Survival prediction": [...]}}.

    :param inf_table: Rows of the endpoint's inference table (request/response JSON, request_time, ...)
    :return: DataFrame with timestamp, request id, the 10 features, prediction and model_name
    """
    request_schema = StructType([StructField("dataframe_records", ArrayType(RECORD_SCHEMA), True)])

    # The custom model (MarvelModelWrapper) answers {"predictions": {"Survival prediction": ["alive", "dead", ...]}}
    response_schema = StructType(
        [
            StructField(
                "predictions",
                StructType([StructField("Survival prediction", ArrayType(StringType()), True)]),
                True,
            ),
            StructField(
                "databricks_output",
                StructType(
                    [StructField("trace", StringType(), True), StructField("databricks_request_id", StringType(), True)]
                ),
                True,
            ),
        ]
    )

    inf_table_parsed = inf_table.withColumn("parsed_request", F.from_json(F.col("request"), request_schema))

    inf_table_parsed = inf_table_parsed.withColumn("parsed_response", F.from_json(F.col("response"), response_schema))

    # posexplode keeps each record's position, so every record is matched with its own prediction
    df_exploded = inf_table_parsed.select(
        "*", F.posexplode(F.col("parsed_request.dataframe_records")).alias("record_pos", "record")
    )
    predicted_label = F.col("parsed_response.predictions.`Survival prediction`").getItem(F.col("record_pos"))

    df_final = df_exploded.withColumn("timestamp_ms", (F.col("request_time").cast("long") * 1000)).select(
        F.col("request_time").alias("timestamp"),  # Use request_time as the timestamp
        F.col("timestamp_ms"),  # Select the newly created timestamp_ms column
        "databricks_request_id",
        "execution_duration_ms",
        *[F.col(f"record.{field.name}").alias(field.name) for field in RECORD_SCHEMA.fields],
        F.when(predicted_label == "alive", 1).when(predicted_label == "dead", 0).alias("prediction"),
        F.lit(MODEL_ID).alias("model_name"),
    )
    return df_final


def inference_log_config() -> MonitorInferenceLog:
    """Inference-log settings of the monitor on model_monitoring.

    :return: Classification inference log over prediction, timestamp and model_name, in 5-minute windows
    """
    return MonitorInferenceLog(
        problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
        prediction_col="prediction",
        timestamp_col="timestamp",
        granularities=["5 minutes"],
        model_id_col="model_name",
    )


def create_monitoring_table(config: ProjectConfig, spark: SparkSession, workspace: WorkspaceClient) -> None:
    """Create the Lakehouse monitor on model_monitoring, with train_set's baseline for drift.

    :param config: Configuration object containing catalog and schema names
    :param spark: SparkSession object for executing SQL commands
    :param workspace: Workspace object for creating quality monitors
    """
    logger.info("Creating new monitoring table..")

    schema = f"{config.catalog_name}.{config.schema_name}"
    monitoring_table = f"{schema}.model_monitoring"

    workspace.quality_monitors.create(
        table_name=monitoring_table,
        assets_dir=f"/Workspace/Shared/lakehouse_monitoring/{monitoring_table}",
        output_schema_name=schema,
        baseline_table_name=f"{schema}.model_monitoring_baseline",
        inference_log=inference_log_config(),
    )

    # Important to update monitoring
    spark.sql(f"ALTER TABLE {monitoring_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

    logger.info("Lakehouse monitoring table is created.")
