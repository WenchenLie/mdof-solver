from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ConvergenceResult:
    converged: bool
    value: float
    limit: float
    name: str


class ConvergenceTest:
    """Metric used to decide whether one nonlinear iteration has converged."""

    name = "ConvergenceTest"

    def evaluate(
        self,
        residual: NDArray[np.float64],
        increment: NDArray[np.float64],
        solution: NDArray[np.float64],
        force_scale: float,
        rtol: float,
        atol: float,
    ) -> ConvergenceResult:
        value = float(self.value(residual, increment))
        scale = float(self.scale(solution, force_scale))
        limit = float(atol + rtol * scale)
        return ConvergenceResult(value <= limit, value, limit, self.name)

    def value(self, residual: NDArray[np.float64], increment: NDArray[np.float64]) -> float:
        raise NotImplementedError

    def scale(self, solution: NDArray[np.float64], force_scale: float) -> float:
        raise NotImplementedError


def _validate_norm_type(norm_type: float) -> float:
    value = float(norm_type)
    if np.isnan(value) or value < 1.0:
        raise ValueError("norm_type 必须不小于 1，或为 np.inf")
    return value


@dataclass(frozen=True)
class NormUnbalance(ConvergenceTest):
    """OpenSees NormUnbalance: norm of the updated residual-force vector."""

    norm_type: float = 2.0
    name = "NormUnbalance"

    def __post_init__(self) -> None:
        object.__setattr__(self, "norm_type", _validate_norm_type(self.norm_type))

    def value(self, residual, increment) -> float:
        return float(np.linalg.norm(residual, ord=self.norm_type))

    def scale(self, solution, force_scale) -> float:
        return max(1.0, float(force_scale))


@dataclass(frozen=True)
class NormDispIncr(ConvergenceTest):
    """OpenSees NormDispIncr: norm of the displacement correction just applied."""

    norm_type: float = 2.0
    name = "NormDispIncr"

    def __post_init__(self) -> None:
        object.__setattr__(self, "norm_type", _validate_norm_type(self.norm_type))

    def value(self, residual, increment) -> float:
        return float(np.linalg.norm(increment, ord=self.norm_type))

    def scale(self, solution, force_scale) -> float:
        return max(1.0, float(np.linalg.norm(solution, ord=self.norm_type)))


@dataclass(frozen=True)
class EnergyIncr(ConvergenceTest):
    """OpenSees EnergyIncr: half the absolute residual/increment inner product."""

    name = "EnergyIncr"

    def value(self, residual, increment) -> float:
        return 0.5 * abs(float(increment @ residual))

    def scale(self, solution, force_scale) -> float:
        reference = 0.5 * float(np.linalg.norm(solution)) * float(force_scale)
        return max(1.0, reference)
