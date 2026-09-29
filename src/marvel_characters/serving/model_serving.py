"""Model serving module for Marvel characters."""

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput

# Inference table <catalog>.<schema>.custom_model_payload, read by monitoring.create_or_refresh_monitoring
INFERENCE_TABLE_PREFIX = "custom_model"


class ModelServing:
    """Manages model serving in Databricks for Marvel characters."""

    def __init__(self, model_name: str, endpoint_name: str) -> None:
        """Initialize the Model Serving Manager.

        :param model_name: Name of the model to be served
        :param endpoint_name: Name of the serving endpoint
        """
        self.workspace = WorkspaceClient()
        self.endpoint_name = endpoint_name
        self.model_name = model_name

    def get_latest_model_version(self) -> str:
        """Retrieve the latest version of the model.

        :return: Latest version of the model as a string
        """
        client = mlflow.MlflowClient()
        latest_version = client.get_model_version_by_alias(self.model_name, alias="latest-model").version
        print(f"Latest model version: {latest_version}")
        return latest_version

    def telemetry_config(self) -> dict:
        """Endpoint telemetry that logs requests/responses to <catalog>.<schema>.custom_model_payload.

        The payload table is a view over the telemetry logs table, both next to the served model.
        The SDK version pinned for serverless env 3 has no telemetry fields, so this is the REST API body.

        :return: telemetry_config section of the serving endpoint create request
        """
        catalog_name, schema_name, _ = self.model_name.split(".")
        prefix = f"{catalog_name}.{schema_name}.{INFERENCE_TABLE_PREFIX}"
        return {
            "table_names": {
                "logs_table": f"{prefix}_otel_logs",
                "metrics_table": f"{prefix}_otel_metrics",
                "traces_table": f"{prefix}_otel_spans",
            },
            "inference_table_config": {"sampling_fraction": 1.0},
        }

    def deploy_or_update_serving_endpoint(
        self, version: str = "latest", workload_size: str = "Small", scale_to_zero: bool = True
    ) -> None:
        """Deploy or update the model serving endpoint in Databricks for Marvel characters.

        :param version: Model version to serve (default: "latest")
        :param workload_size: Size of the serving workload (default: "Small")
        :param scale_to_zero: Whether to enable scale-to-zero (default: True)
        """
        endpoint_exists = any(item.name == self.endpoint_name for item in self.workspace.serving_endpoints.list())
        entity_version = self.get_latest_model_version() if version == "latest" else version

        served_entities = [
            ServedEntityInput(
                entity_name=self.model_name,
                scale_to_zero_enabled=scale_to_zero,
                workload_size=workload_size,
                entity_version=entity_version,
            )
        ]

        if not endpoint_exists:
            # Telemetry must be set when the endpoint is created: added to an existing endpoint it is accepted
            # but never delivers any rows (and AI Gateway inference tables are rejected for this endpoint type).
            self.workspace.api_client.do(
                "POST",
                "/api/2.0/serving-endpoints",
                body={
                    "name": self.endpoint_name,
                    "config": {"served_entities": [entity.as_dict() for entity in served_entities]},
                    "telemetry_config": self.telemetry_config(),
                },
            )
        else:
            endpoint = self.workspace.api_client.do("GET", f"/api/2.0/serving-endpoints/{self.endpoint_name}")
            if "telemetry_config" not in endpoint:
                print(
                    f"WARNING: endpoint {self.endpoint_name} has no inference-table logging, so monitoring gets "
                    "no data. Logging can only be set at creation: delete the endpoint and rerun this deployment."
                )
            self.workspace.serving_endpoints.update_config(name=self.endpoint_name, served_entities=served_entities)
