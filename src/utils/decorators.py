# --------------------------------------------------------------------------------
# Decorators used throughout the project
# --------------------------------------------------------------------------------

import time

def chronometer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)  # Call the original function
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"{func.__name__} took {elapsed:.4f} seconds to complete.")
        return result
    return wrapper

