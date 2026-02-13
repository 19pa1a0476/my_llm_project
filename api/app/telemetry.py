import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
import json

from sqlalchemy import text

from api.app.db import SessionLocal


@dataclass
class RequestTelemetry:
    request_id: uuid.UUID
    model_name: str
    model_version: str
    prompt_version: str
    retriever_version: str
    query_text: str
    response_text: str
    retrieval_context: list[str]
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    latency_ms: int
    cache_hit: bool
    created_at: datetime
    tenant_id: str | None = None
    user_id: str | None = None


def estimate_cost_usd(
    input_tokens: int,
    output_tokens: int,
    input_cost_per_1k: float,
    output_cost_per_1k: float,
) -> float:
    input_cost = (input_tokens / 1000.0) * input_cost_per_1k
    output_cost = (output_tokens / 1000.0) * output_cost_per_1k
    return round(input_cost + output_cost, 6)


def persist_request_metrics(metrics: RequestTelemetry) -> None:
    stmt = text(
        """
        INSERT INTO llm_requests (
            request_id,
            tenant_id,
            user_id,
            model_name,
            model_version,
            prompt_version,
            retriever_version,
            query_text,
            response_text,
            retrieval_context,
            input_tokens,
            output_tokens,
            estimated_cost_usd,
            latency_ms,
            cache_hit,
            created_at
        ) VALUES (
            :request_id,
            :tenant_id,
            :user_id,
            :model_name,
            :model_version,
            :prompt_version,
            :retriever_version,
            :query_text,
            :response_text,
            CAST(:retrieval_context AS JSONB),
            :input_tokens,
            :output_tokens,
            :estimated_cost_usd,
            :latency_ms,
            :cache_hit,
            :created_at
        )
        """
    )
    payload = {
        "request_id": metrics.request_id,
        "tenant_id": metrics.tenant_id,
        "user_id": metrics.user_id,
        "model_name": metrics.model_name,
        "model_version": metrics.model_version,
        "prompt_version": metrics.prompt_version,
        "retriever_version": metrics.retriever_version,
        "query_text": metrics.query_text,
        "response_text": metrics.response_text,
        "retrieval_context": json.dumps(metrics.retrieval_context),
        "input_tokens": metrics.input_tokens,
        "output_tokens": metrics.output_tokens,
        "estimated_cost_usd": metrics.estimated_cost_usd,
        "latency_ms": metrics.latency_ms,
        "cache_hit": metrics.cache_hit,
        "created_at": metrics.created_at,
    }
    with SessionLocal() as db:
        db.execute(stmt, payload)
        db.commit()


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
