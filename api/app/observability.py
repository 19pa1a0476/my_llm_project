import uuid
from typing import Any

import mlflow
from langsmith import Client

from api.app.config import settings


def log_to_mlflow(
    *,
    request_id: uuid.UUID,
    model_name: str,
    model_version: str,
    prompt_version: str,
    retriever_version: str,
    latency_ms: int,
    input_tokens: int,
    output_tokens: int,
    estimated_cost_usd: float,
    cache_hit: bool,
) -> None:
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)
        with mlflow.start_run(run_name=f"query-{request_id}", nested=True):
            mlflow.log_params(
                {
                    "request_id": str(request_id),
                    "model_name": model_name,
                    "model_version": model_version,
                    "prompt_version": prompt_version,
                    "retriever_version": retriever_version,
                    "cache_hit": cache_hit,
                }
            )
            mlflow.log_metrics(
                {
                    "latency_ms": latency_ms,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "estimated_cost_usd": estimated_cost_usd,
                }
            )
    except Exception:
        return


def log_to_langsmith(
    *,
    request_id: uuid.UUID,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    latency_ms: int,
    input_tokens: int,
    output_tokens: int,
    estimated_cost_usd: float,
    cache_hit: bool,
) -> None:
    if not settings.langsmith_tracing:
        return

    try:
        client = Client()
        client.create_run(
            id=request_id,
            name="rag_query",
            run_type="chain",
            project_name=settings.langsmith_project,
            inputs=inputs,
            outputs=outputs,
            extra={
                "metrics": {
                    "latency_ms": latency_ms,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "estimated_cost_usd": estimated_cost_usd,
                },
                "cache_hit": cache_hit,
            },
        )
    except Exception:
        return
