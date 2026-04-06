from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

# 4 model threads (person, weapon, fire, road_anomaly) + 1 for scene + 1 spare
# Shared across all requests to avoid overhead
_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="jerico-worker")

def get_executor():
    return _EXECUTOR

logger.info("Worker pool initialized with 8 threads.")
