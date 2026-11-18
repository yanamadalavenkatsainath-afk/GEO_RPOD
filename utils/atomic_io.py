"""Ground-only atomic file replacement with bounded Windows lock retries."""
import math
import os
import time


_TRANSIENT_WINDOWS_ERRORS = frozenset((5, 32, 33))


def atomic_replace(source, target, *, timeout_s=2.75):
    """Atomically replace a closed temporary file, tolerating brief sync locks.

    Retry only Windows access-denied, sharing-violation, or lock-violation
    errors. Backoff is deterministic and capped by a monotonic deadline of
    at most three seconds. No random state is used. Each attempt remains
    ``os.replace``: there is no delete/copy fallback or cleanup on failure,
    so the old target and completed temporary file remain available when
    replacement cannot succeed. Other operating-system errors propagate.
    """
    if not math.isfinite(timeout_s) or not 0 <= timeout_s <= 3.0:
        raise ValueError('Atomic replacement retry budget must be between 0 and 3 seconds')
    deadline = time.monotonic() + timeout_s
    delay = .05
    while True:
        try:
            os.replace(source, target)
            return
        except OSError as error:
            if getattr(error, 'winerror', None) not in _TRANSIENT_WINDOWS_ERRORS:
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(delay, remaining))
            if time.monotonic() >= deadline:
                raise
            delay = min(2 * delay, .5)
