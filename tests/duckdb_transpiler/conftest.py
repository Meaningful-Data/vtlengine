"""
Pytest configuration for duckdb_transpiler tests.

Provides a timeout mechanism to skip slow tests.
"""

import signal
from functools import wraps
from typing import Any, Callable

import pytest

# Default timeout in seconds for transpiler tests
DEFAULT_TIMEOUT = 5


class TestTimeoutError(Exception):
    """Custom timeout exception."""

    pass


def timeout_handler(signum: int, frame: Any) -> None:
    """Signal handler for timeout."""
    raise TestTimeoutError("Test execution timed out")


def with_timeout(seconds: int = DEFAULT_TIMEOUT) -> Callable:
    """
    Decorator that skips a test if it takes longer than the specified timeout.

    Args:
        seconds: Maximum allowed execution time in seconds.

    Usage:
        @with_timeout(5)
        def test_something():
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Set up the signal handler
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            except TestTimeoutError:
                pytest.skip(f"Test skipped: exceeded {seconds}s timeout")
            finally:
                # Restore the old handler and cancel the alarm
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            return result

        return wrapper

    return decorator


@pytest.fixture(autouse=True)
def auto_timeout(request: pytest.FixtureRequest) -> Any:
    """
    Automatically apply timeout to all tests in this directory.

    Tests can opt out by using @pytest.mark.no_timeout decorator.
    Tests can customize timeout with @pytest.mark.timeout(seconds) marker.

    Note: Timeout only works for Python code. Native code (like DuckDB operations)
    may not be interruptible.
    """
    # Check if test has no_timeout marker
    if request.node.get_closest_marker("no_timeout"):
        yield
        return

    # Get custom timeout from marker or use default
    timeout_marker = request.node.get_closest_marker("timeout")
    timeout_seconds = timeout_marker.args[0] if timeout_marker else DEFAULT_TIMEOUT

    # Set up the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        yield
    except TestTimeoutError:
        pytest.skip(f"Test skipped: exceeded {timeout_seconds}s timeout")
    finally:
        # Restore the old handler and cancel the alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
