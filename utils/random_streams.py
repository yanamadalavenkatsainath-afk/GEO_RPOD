"""Stable named RNG streams independent of component construction order."""
import hashlib
from numbers import Integral

import numpy as np

from utils.validation import text_label


def random_generator(*, seed=None, rng=None):
    if seed is not None and rng is not None:
        raise ValueError("Provide seed or rng, not both")
    if rng is not None:
        if not isinstance(rng, np.random.Generator):
            raise ValueError("rng must be numpy.random.Generator")
        return rng
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    return np.random.default_rng(seed)


class RandomStreams:
    """One persistent stream per component name and explicit run seed.

    Use different run seeds for MC trials. Names use SHA256 rather than Python's
    process-randomized hash. Reproducibility assumes the recorded NumPy version.
    """
    def __init__(self, seed):
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.seed = int(seed)
        self._streams = {}

    def generator(self, name):
        text_label(name, "stream name")
        if name not in self._streams:
            identity = hashlib.sha256(name.encode("utf-8")).digest()
            words = [int.from_bytes(identity[k:k+4], "little") for k in range(0, 32, 4)]
            sequence = np.random.SeedSequence(self.seed, spawn_key=tuple(words))
            self._streams[name] = np.random.default_rng(sequence)
        return self._streams[name]

    def reset(self):
        """Forget named generators; reacquire them after reset."""
        self._streams.clear()
