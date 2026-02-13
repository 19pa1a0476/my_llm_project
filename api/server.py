import uuid
from time import perf_counter

from fastapi import FastAPI, Header, Query
from sqlalchemy import text

from api.app.cache import redis_client
from api.app.db import SessionLocal
from api.app.config import settings
from api.app.observability import log_to_langsmith, log_to_mlflow
from api.app.telemetry import RequestTelemetry, estimate_cost_usd, now_utc, persist_request_metrics
from api.rag_pipeline import get_answer

app = FastAPI(title='LLM Quality API', version='0.1.0')


@app.get('/health')
def health() -> dict:
    db_ok = False
    redis_ok = False

    try:
        with SessionLocal() as db:
            db.execute(text('SELECT 1'))
        db_ok = True
    except Exception:
        db_ok = False

    try:
        redis_ok = bool(redis_client.ping())
    except Exception:
        redis_ok = False

    status = 'ok' if db_ok and redis_ok else 'degraded'
    return {'status': status, 'postgres': db_ok, 'redis': redis_ok}


@app.get('/query')
def query(
    q: str = Query(..., description='User question'),
    x_tenant_id: str | None = Header(default=None),
    x_user_id: str | None = Header(default=None),
) -> dict:
    request_id = uuid.uuid4()
    started = perf_counter()
    result = get_answer(q)
    latency_ms = int((perf_counter() - started) * 1000)

    input_tokens = int(result["input_tokens"])
    output_tokens = int(result["output_tokens"])
    estimated_cost = estimate_cost_usd(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost_per_1k=settings.input_cost_per_1k_tokens_usd,
        output_cost_per_1k=settings.output_cost_per_1k_tokens_usd,
    )

    metrics = RequestTelemetry(
        request_id=request_id,
        tenant_id=x_tenant_id,
        user_id=x_user_id,
        model_name=result["model_name"],
        model_version=result["model_version"],
        prompt_version=result["prompt_version"],
        retriever_version=result["retriever_version"],
        query_text=q,
        response_text=result["answer"],
        retrieval_context=result["context_docs"],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=estimated_cost,
        latency_ms=latency_ms,
        cache_hit=bool(result["cache_hit"]),
        created_at=now_utc(),
    )

    try:
        persist_request_metrics(metrics)
    except Exception:
        pass

    log_to_mlflow(
        request_id=request_id,
        model_name=result["model_name"],
        model_version=result["model_version"],
        prompt_version=result["prompt_version"],
        retriever_version=result["retriever_version"],
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=estimated_cost,
        cache_hit=bool(result["cache_hit"]),
    )
    log_to_langsmith(
        request_id=request_id,
        inputs={"query": q},
        outputs={"answer": result["answer"]},
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=estimated_cost,
        cache_hit=bool(result["cache_hit"]),
    )

    return {
        'request_id': str(request_id),
        'query': q,
        'answer': result["answer"],
        'cache_hit': result["cache_hit"],
        'latency_ms': latency_ms,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'estimated_cost_usd': estimated_cost,
    }
