"""Tests for creating and updating the serving endpoint."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from marvel_characters.serving.model_serving import ModelServing  # noqa: E402

MODEL_NAME = "mlops_dev.marvel_characters.marvel_character_model_custom"
ENDPOINT_NAME = "marvel-characters-model-serving-dev"


def _serving(existing_endpoints: list[str]) -> tuple[ModelServing, MagicMock]:
    """Build a ModelServing whose workspace client is a mock listing the given endpoints."""
    with patch("marvel_characters.serving.model_serving.WorkspaceClient") as workspace_client:
        serving = ModelServing(model_name=MODEL_NAME, endpoint_name=ENDPOINT_NAME)
    workspace = workspace_client.return_value
    endpoints = []
    for name in existing_endpoints:
        endpoint = MagicMock()
        endpoint.name = name
        endpoints.append(endpoint)
    workspace.serving_endpoints.list.return_value = endpoints
    return serving, workspace


def test_new_endpoint_is_created_with_inference_table_logging() -> None:
    """A new endpoint gets telemetry at creation, logging to custom_model_* next to the model."""
    serving, workspace = _serving(existing_endpoints=[])

    serving.deploy_or_update_serving_endpoint(version="7")

    method, path = workspace.api_client.do.call_args.args
    body = workspace.api_client.do.call_args.kwargs["body"]
    assert (method, path) == ("POST", "/api/2.0/serving-endpoints")
    assert body["name"] == ENDPOINT_NAME
    assert body["config"]["served_entities"][0]["entity_version"] == "7"
    assert body["telemetry_config"]["table_names"]["logs_table"] == (
        "mlops_dev.marvel_characters.custom_model_otel_logs"
    )
    workspace.serving_endpoints.update_config.assert_not_called()


def test_existing_endpoint_without_logging_is_updated_with_warning(capsys: pytest.CaptureFixture[str]) -> None:
    """An existing endpoint is updated in place; missing logging is reported, not silently patched."""
    serving, workspace = _serving(existing_endpoints=[ENDPOINT_NAME])
    workspace.api_client.do.return_value = {"name": ENDPOINT_NAME}  # no telemetry_config

    serving.deploy_or_update_serving_endpoint(version="8")

    workspace.serving_endpoints.update_config.assert_called_once()
    assert workspace.serving_endpoints.update_config.call_args.kwargs["served_entities"][0].entity_version == "8"
    assert "no inference-table logging" in capsys.readouterr().out
