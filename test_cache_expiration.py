from datetime import datetime, timedelta, timezone

from src.utils.model_manager import (
    is_cache_expired,
    MODEL_CACHE_EXPIRATION_DAYS,
)


now = datetime.now(timezone.utc)


fresh_state = {
    "installed_at": now.isoformat()
}


old_time = now - timedelta(
    days=MODEL_CACHE_EXPIRATION_DAYS + 1
)

old_state = {
    "installed_at": old_time.isoformat()
}


missing_state = {}


print("Expiration period:", MODEL_CACHE_EXPIRATION_DAYS, "days")

print(
    "Fresh cache expired:",
    is_cache_expired(fresh_state)
)

print(
    "Old cache expired:",
    is_cache_expired(old_state)
)

print(
    "Missing timestamp expired:",
    is_cache_expired(missing_state)
)