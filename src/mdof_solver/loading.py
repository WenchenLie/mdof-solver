from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


class LoadHistory:
    """A bounded vector-valued load history.

    The primary interface accepts uniformly sampled force vectors and ``dt``.
    The older ``LoadHistory(times, forces)`` form remains accepted so existing
    models are not broken.
    """

    def __init__(
        self,
        samples_or_function: ArrayLike | Callable[[float], ArrayLike],
        forces: ArrayLike | None = None,
        *,
        dt: float | None = None,
        t0: float = 0.0,
        t_min: float | None = None,
        t_max: float | None = None,
        size: int | None = None,
    ):
        if callable(samples_or_function):
            if t_min is None or t_max is None or size is None:
                raise ValueError("A functional load requires t_min, t_max, and size")
            if not t_max > t_min or size <= 0:
                raise ValueError("Invalid time range or vector size for a functional load")
            self._function = samples_or_function
            self.t_min, self.t_max, self.size = float(t_min), float(t_max), int(size)
            self.dt = None
            self.times = None
            self.forces = None
        elif forces is None:
            values = np.asarray(samples_or_function, dtype=float)
            if dt is None or not np.isfinite(dt) or dt <= 0:
                raise ValueError("A uniformly sampled load requires a finite dt greater than zero")
            if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
                raise ValueError("The load array must have shape (time points, DOFs) with at least two time points")
            if not np.isfinite(t0) or not np.all(np.isfinite(values)):
                raise ValueError("The load history contains non-finite values")
            self.dt = float(dt)
            self.t_min = float(t0)
            self.t_max = self.t_min + self.dt * (values.shape[0] - 1)
            self.times = self.t_min + self.dt * np.arange(values.shape[0])
            self.forces = values
            self.size = values.shape[1]
            self._function = None
        else:
            if dt is not None:
                raise ValueError("dt cannot be provided together with explicit times")
            times = np.asarray(samples_or_function, dtype=float)
            values = np.asarray(forces, dtype=float)
            if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0):
                raise ValueError("Load times must be a strictly increasing one-dimensional array")
            if values.ndim != 2 or values.shape[0] != len(times):
                raise ValueError("The load array must have shape (time points, DOFs)")
            if not np.all(np.isfinite(times)) or not np.all(np.isfinite(values)):
                raise ValueError("The load history contains non-finite values")
            self.times, self.forces = times, values
            self.t_min, self.t_max = float(times[0]), float(times[-1])
            increments = np.diff(times)
            self.dt = float(increments[0]) if np.allclose(increments, increments[0]) else None
            self.size = values.shape[1]
            self._function = None

    def __call__(self, time: float) -> NDArray[np.float64]:
        t = float(time)
        tolerance = 1e-12 * max(1.0, abs(self.t_min), abs(self.t_max))
        if t < self.t_min - tolerance or t > self.t_max + tolerance:
            raise ValueError(f"Load query time {t:g} is outside [{self.t_min:g}, {self.t_max:g}]")
        t = min(max(t, self.t_min), self.t_max)
        if self._function is not None:
            value = np.asarray(self._function(t), dtype=float)
        else:
            value = np.array([np.interp(t, self.times, self.forces[:, i]) for i in range(self.size)])
        if value.shape != (self.size,) or not np.all(np.isfinite(value)):
            raise ValueError("The load function returned an invalid shape or non-finite values")
        return value

    def __add__(self, other: "LoadHistory") -> "LoadHistory":
        if not isinstance(other, LoadHistory) or self.size != other.size:
            raise ValueError("Loads being added must have the same number of DOFs")
        lower, upper = max(self.t_min, other.t_min), min(self.t_max, other.t_max)
        if upper <= lower:
            raise ValueError("Loads being added have no common time range")
        return LoadHistory(lambda t: self(t) + other(t), t_min=lower, t_max=upper, size=self.size)

    __radd__ = __add__
