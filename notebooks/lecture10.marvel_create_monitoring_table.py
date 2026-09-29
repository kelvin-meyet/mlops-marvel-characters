# Databricks notebook source
# Send traffic to a serving endpoint, then create or refresh the Lakehouse monitor on its inference table.
# Runs locally (VS Code + Databricks Connect) or in the Databricks workspace.
# The endpoint must have inference-table logging on (see README fix 12), and logs arrive in batches,
# so the monitor only sees requests after they land in custom_model_payload (often within an hour).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Settings and connection

# COMMAND ----------

import time

import numpy as np
import requests
from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient

from marvel_characters.config import ProjectConfig
from marvel_characters.utils import is_databricks

ENV = "dev"
ENDPOINT_NAME = f"marvel-characters-model-serving-{ENV}"
PROFILE = "dbc-07d12638-ef42"  # local runs only; inside Databricks the notebook's own login is used

w = WorkspaceClient() if is_databricks() else WorkspaceClient(profile=PROFILE)
spark = (
    DatabricksSession.builder.getOrCreate()
    if is_databricks()
    else DatabricksSession.builder.profile(PROFILE).serverless(True).getOrCreate()
)
config = ProjectConfig.from_yaml(config_path="../project_config_marvel.yml", env=ENV)
print(f"Workspace: {w.config.host}")
print(f"Endpoint:  {ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Send data to the endpoint

# COMMAND ----------

features = config.num_features + config.cat_features
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").toPandas()

# One character per request, so each request becomes one monitoring row. NaN is not valid JSON, so use None.
N_REQUESTS = 100
sampled_records = (
    test_set[features].sample(n=N_REQUESTS, replace=True).replace({np.nan: None}).to_dict(orient="records")
)


def send_request(record: dict) -> tuple[int, str]:
    """Send one character to the serving endpoint and return the status code and response text."""
    response = requests.post(
        f"{w.config.host}/serving-endpoints/{ENDPOINT_NAME}/invocations",
        headers=w.config.authenticate(),  # fresh auth header on every call, never expires mid-run
        json={"dataframe_records": [record]},
        timeout=120,  # the first call after an idle period can be slow (scale-to-zero)
    )
    return response.status_code, response.text


# COMMAND ----------

failures = 0
for i, record in enumerate(sampled_records, start=1):
    status_code, response_text = send_request(record)
    if status_code != 200:
        failures += 1
        print(f"Request {i}: {status_code} {response_text}")
    time.sleep(0.2)
print(f"Sent {N_REQUESTS} requests, {failures} failed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create or refresh the monitor
# MAGIC Run this once the requests above have arrived in `custom_model_payload`.

# COMMAND ----------

from marvel_characters.monitoring import create_or_refresh_monitoring  # noqa: E402

create_or_refresh_monitoring(config=config, spark=spark, workspace=w)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Drift test: make the monitor catch a change in the data
# MAGIC
# MAGIC Drift compares one 5-minute window of traffic with the previous one, so it needs a second batch of
# MAGIC requests in a later window. This section sends a deliberately **skewed** batch (every character from one
# MAGIC universe, very tall and heavy) so the drift is easy to spot.
# MAGIC
# MAGIC **Before you start:** the monitor must already exist, i.e. you ran the sections above once.
# MAGIC If the notebook was restarted, first run **Settings and connection** and the cell that defines
# MAGIC `send_request` (the first cell under *Send data to the endpoint*). Do not rerun the send loop.
# MAGIC
# MAGIC **Then run these three cells in order, one at a time:**
# MAGIC 1. **Drift step 1**: sends 100 skewed requests (waits by itself if the previous batch is in the current window).
# MAGIC 2. **Drift step 2**: waits for the requests to be logged, then updates `model_monitoring` and starts a refresh.
# MAGIC 3. **Drift step 3**: waits for the refresh (usually 10-20 min) and prints the drift per feature.
# MAGIC
# MAGIC **What you should see:** high drift for `Universe`, `Height` and `Weight` (and maybe `prediction`), close to
# MAGIC zero for the other features, in both comparisons step 3 prints: against the previous window (`CONSECUTIVE`)
# MAGIC and against the training data (`BASELINE`). The same numbers appear in the monitor dashboard's drift charts.

# COMMAND ----------

# Drift step 1: send a skewed batch of requests in a new 5-minute window
payload_table = f"{config.catalog_name}.{config.schema_name}.custom_model_payload"
WINDOW_SECONDS = 5 * 60

last_request = spark.sql(f"SELECT unix_timestamp(max(request_time)) AS t FROM {payload_table}").first().t
if last_request and int(time.time()) // WINDOW_SECONDS == last_request // WINDOW_SECONDS:
    wait = WINDOW_SECONDS - int(time.time()) % WINDOW_SECONDS + 5
    print(f"The previous batch is in the current 5-minute window; waiting {wait} s for the next one...")
    time.sleep(wait)

payload_rows_before = spark.table(payload_table).count()

skewed_records = (
    test_set[features].sample(n=N_REQUESTS, replace=True).replace({np.nan: None}).to_dict(orient="records")
)
for record in skewed_records:
    record["Universe"] = "Earth-1610"
    record["Height"] = 250.0
    record["Weight"] = 200.0

failures = 0
for i, record in enumerate(skewed_records, start=1):
    status_code, response_text = send_request(record)
    if status_code != 200:
        failures += 1
        print(f"Request {i}: {status_code} {response_text}")
    time.sleep(0.2)
print(f"Sent {N_REQUESTS} skewed requests, {failures} failed. Now run Drift step 2.")

# COMMAND ----------

# Drift step 2: wait until the skewed requests are logged, then update model_monitoring and start a refresh
expected_rows = payload_rows_before + N_REQUESTS - failures
for _ in range(20):
    logged = spark.table(payload_table).count()
    print(f"Logged requests: {logged}/{expected_rows}")
    if logged >= expected_rows:
        break
    time.sleep(15)
else:
    raise TimeoutError("The skewed requests were not logged within 5 minutes; check the endpoint's logging.")

create_or_refresh_monitoring(config=config, spark=spark, workspace=w)
print("Refresh started. Now run Drift step 3.")

# COMMAND ----------

# Drift step 3: wait for the refresh to finish, then show the drift for the newest window
monitor_table = f"{config.catalog_name}.{config.schema_name}.model_monitoring"
drift_table = f"{config.catalog_name}.{config.schema_name}.model_monitoring_drift_metrics"

while True:
    refresh = w.quality_monitors.list_refreshes(monitor_table).refreshes[0]  # newest first
    state = refresh.state.value
    print(f"{time.strftime('%H:%M:%S')} refresh {refresh.refresh_id}: {state}")
    if state not in ("PENDING", "RUNNING"):
        break
    time.sleep(60)
if state != "SUCCESS":
    raise RuntimeError(f"Refresh ended with {state}: {refresh.message}")

# One row per feature for the newest window, compared with the previous window (drift_type CONSECUTIVE)
# and with the training data (drift_type BASELINE).
# js_distance: 0 = same distribution, 1 = completely different. p-values below 0.05 mean a significant change.
drift = spark.sql(f"""
    SELECT drift_type,
           column_name,
           round(js_distance, 3)                AS js_distance,
           round(ks_test.pvalue, 4)             AS ks_pvalue,
           round(chi_squared_test.pvalue, 4)    AS chi_squared_pvalue
    FROM {drift_table}
    WHERE slice_key IS NULL
      AND window.start = (SELECT max(window.start) FROM {drift_table})
      AND column_name NOT IN (':table', 'timestamp', 'timestamp_ms', 'databricks_request_id', 'execution_duration_ms')
    ORDER BY drift_type, js_distance DESC NULLS LAST
""")
drift.show(50, truncate=False)
print("Features at the top drifted the most. See the monitor dashboard for the same data as charts.")

# COMMAND ----------
