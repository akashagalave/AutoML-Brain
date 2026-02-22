import json
import hashlib
import redis
import logging

from .config import REDIS_HOST, REDIS_PORT, REDIS_DB, CACHE_TTL_SECONDS

logger = logging.getLogger("redis-cache")

try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True
    )
    redis_client.ping()
    REDIS_AVAILABLE = True
    logger.info("Redis connected successfully")

except Exception as e:
    REDIS_AVAILABLE = False
    logger.warning(f"Redis unavailable: {e}")


def generate_cache_key(features: dict) -> str:
    feature_string = json.dumps(features, sort_keys=True)
    hash_value = hashlib.sha256(feature_string.encode()).hexdigest()
    return f"churn_prediction:{hash_value}"


def get_cached_prediction(features: dict):
    if not REDIS_AVAILABLE:
        return None

    try:
        key = generate_cache_key(features)
        cached = redis_client.get(key)
        if cached:
            logger.info("Cache hit")
            return json.loads(cached)
    except Exception as e:
        logger.warning(f"Redis get failed: {e}")

    return None


def set_cached_prediction(features: dict, prediction: dict):
    if not REDIS_AVAILABLE:
        return

    try:
        key = generate_cache_key(features)
        redis_client.setex(
            key,
            CACHE_TTL_SECONDS,
            json.dumps(prediction)
        )
        logger.info("Cache stored")
    except Exception as e:
        logger.warning(f"Redis set failed: {e}")