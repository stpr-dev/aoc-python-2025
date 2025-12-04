import timeit
from collections.abc import Callable
from typing import Any


def time_callable(fn: Callable[..., Any], *args: Any, number: int = 5) -> float:
    """Time a single callable using timeit, returning average seconds per run."""
    timer = timeit.Timer(lambda: fn(*args))
    total_time = timer.timeit(number=number)
    return total_time / number


__all__ = ["time_callable"]
