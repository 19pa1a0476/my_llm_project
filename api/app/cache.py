import hashlib
import json
from typing import Any

import redis

from app.config import settings

redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)


def make_cache_key(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode('utf-8')).hexdigest()
    return f'llm:answer:{digest}'


def get_cached_answer(cache_key: str) -> str | None:
    return redis_client.get(cache_key)


def set_cached_answer(cache_key: str, value: str, ttl_seconds: int = 600) -> None:
    redis_client.setex(cache_key, ttl_seconds, value)
