from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

# 4 model threads per frame * (up to 4 frames in parallel) = 16
# Shared across all requests to avoid overhead
_EXECUTOR = ThreadPoolExecutor(max_workers=16, thread_name_prefix="jerico-worker")

def get_executor():
    return _EXECUTOR

logger.info("Worker pool initialized with 8 threads.")
