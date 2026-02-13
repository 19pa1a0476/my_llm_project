from fastapi import FastAPI, Query
from sqlalchemy import text

from api.app.cache import redis_client
from api.app.db import SessionLocal
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
def query(q: str = Query(..., description='User question')) -> dict:
    answer, cache_hit = get_answer(q)
    return {'query': q, 'answer': answer, 'cache_hit': cache_hit}
