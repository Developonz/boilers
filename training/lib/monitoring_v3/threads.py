"""Graceful handling of Windows DLL enumeration in non-ASCII workspace paths."""
from contextlib import contextmanager, nullcontext
import warnings


@contextmanager
def numerical_threads(count):
    from threadpoolctl import threadpool_limits
    try:
        manager=threadpool_limits(limits=count)
    except (FileNotFoundError,OSError) as exc:
        # Some threadpoolctl/Windows combinations enumerate Unicode DLL paths as '??'.
        # CatBoost's explicit thread_count still applies; do not alter third-party binaries.
        warnings.warn(f'Native numerical thread limit unavailable: {exc}. CatBoost uses its own thread_count.',RuntimeWarning)
        manager=nullcontext()
    with manager:
        yield
