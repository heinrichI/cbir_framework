import time
from functools import wraps


def mesure_time_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start_time = time.time()

        fn(*args, **kwargs)

        elapsed_time = time.time() - start_time
        return elapsed_time

    return wrapper
