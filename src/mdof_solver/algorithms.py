from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class EquilibriumProblem(Protocol):
    def evaluate(self, x: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64], float]: ...
    def check_convergence(
        self,
        residual: NDArray[np.float64],
        increment: NDArray[np.float64],
        solution: NDArray[np.float64],
        force_scale: float,
    ): ...


@dataclass(frozen=True)
class SolveOutcome:
    converged: bool
    x: NDArray[np.float64]
    iterations: int
    residual_norm: float
    reason: str = ""
    test_value: float = float("nan")
    test_name: str = ""
    test_limit: float = float("nan")


class SolutionAlgorithm:
    name = "algorithm"

    def solve(self, problem: EquilibriumProblem, initial: NDArray[np.float64], rtol: float, atol: float, max_iter: int) -> SolveOutcome:
        raise NotImplementedError

    @staticmethod
    def _outcome(converged, x, iteration, residual, test, reason="") -> SolveOutcome:
        return SolveOutcome(
            converged, x, iteration, float(np.linalg.norm(residual)), reason,
            test.value, test.name, test.limit,
        )


@dataclass(frozen=True)
class Linear(SolutionAlgorithm):
    name = "Linear"

    def solve(self, problem, initial, rtol, atol, max_iter) -> SolveOutcome:
        try:
            residual, tangent, _ = problem.evaluate(initial)
            increment = np.linalg.solve(tangent, residual)
            x = initial + increment
            final, _, final_scale = problem.evaluate(x)
            test = problem.check_convergence(final, increment, x, final_scale)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
            return SolveOutcome(False, initial, 1, float("inf"), str(exc))
        return self._outcome(True, x, 1, final, test)


@dataclass(frozen=True)
class Newton(SolutionAlgorithm):
    name = "Newton"

    def solve(self, problem, initial, rtol, atol, max_iter) -> SolveOutcome:
        x = initial.copy()
        residual = np.full_like(x, np.inf)
        test = None
        iteration = 0
        try:
            for iteration in range(1, max_iter + 1):
                residual, tangent, scale = problem.evaluate(x)
                increment = np.linalg.solve(tangent, residual)
                x = x + increment
                residual, _, scale = problem.evaluate(x)
                test = problem.check_convergence(residual, increment, x, scale)
                if test.converged:
                    return self._outcome(True, x, iteration, residual, test)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
            return SolveOutcome(False, x, iteration, float(np.linalg.norm(residual)), str(exc))
        return self._outcome(False, x, max_iter, residual, test, "达到最大迭代次数")


@dataclass(frozen=True)
class NewtonLineSearch(SolutionAlgorithm):
    tolerance: float = 0.8
    max_search_iter: int = 10
    min_eta: float = 0.1
    max_eta: float = 10.0
    name = "NewtonLineSearch"

    def solve(self, problem, initial, rtol, atol, max_iter) -> SolveOutcome:
        x = initial.copy()
        residual = np.full_like(x, np.inf)
        test = None
        iteration = 0
        try:
            for iteration in range(1, max_iter + 1):
                residual, tangent, scale = problem.evaluate(x)
                direction = np.linalg.solve(tangent, residual)
                s0 = float(direction @ residual)
                eta = 1.0
                candidate = x + direction
                candidate_residual, _, _ = problem.evaluate(candidate)
                s_eta = float(direction @ candidate_residual)
                for _ in range(self.max_search_iter):
                    if abs(s_eta) <= self.tolerance * max(abs(s0), np.finfo(float).eps):
                        break
                    denominator = s0 - s_eta
                    if abs(denominator) <= np.finfo(float).eps:
                        eta *= 0.5
                    else:
                        eta = eta * s0 / denominator
                    eta = float(np.clip(eta, self.min_eta, self.max_eta))
                    candidate = x + eta * direction
                    candidate_residual, _, _ = problem.evaluate(candidate)
                    s_eta = float(direction @ candidate_residual)
                increment = candidate - x
                x = candidate
                residual, _, scale = problem.evaluate(x)
                test = problem.check_convergence(residual, increment, x, scale)
                if test.converged:
                    return self._outcome(True, x, iteration, residual, test)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
            return SolveOutcome(False, x, iteration, float(np.linalg.norm(residual)), str(exc))
        return self._outcome(False, x, max_iter, residual, test, "达到最大迭代次数")


@dataclass(frozen=True)
class KrylovNewton(SolutionAlgorithm):
    max_dimension: int = 3
    name = "KrylovNewton"

    def __post_init__(self) -> None:
        if self.max_dimension < 1:
            raise ValueError("KrylovNewton.max_dimension 必须大于 0")

    def solve(self, problem, initial, rtol, atol, max_iter) -> SolveOutcome:
        x = initial.copy()
        corrections: list[NDArray[np.float64]] = []
        raw_history: list[NDArray[np.float64]] = []
        test = None
        iteration = 0
        try:
            residual, frozen_tangent, scale = problem.evaluate(x)
            for iteration in range(1, max_iter + 1):
                raw = np.linalg.solve(frozen_tangent, residual)
                if raw_history:
                    differences = [old - new for old, new in zip(raw_history, raw_history[1:] + [raw], strict=True)]
                    A = np.column_stack(differences)
                    coefficients, _, rank, _ = np.linalg.lstsq(A, raw, rcond=None)
                    if rank < A.shape[1]:
                        corrections.clear(); raw_history.clear()
                        _, frozen_tangent, _ = problem.evaluate(x)
                        raw = np.linalg.solve(frozen_tangent, residual)
                        correction = raw
                    else:
                        correction = raw.copy()
                        for coefficient, previous, difference in zip(coefficients, corrections, differences, strict=True):
                            correction += coefficient * (previous - difference)
                else:
                    correction = raw
                corrections.append(correction.copy())
                raw_history.append(raw.copy())
                x = x + correction
                residual, _, scale = problem.evaluate(x)
                test = problem.check_convergence(residual, correction, x, scale)
                if test.converged:
                    return self._outcome(True, x, iteration, residual, test)
                if len(raw_history) > self.max_dimension:
                    corrections.clear(); raw_history.clear()
                    _, frozen_tangent, _ = problem.evaluate(x)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
            return SolveOutcome(False, x, iteration, float(np.linalg.norm(residual)), str(exc))
        return self._outcome(False, x, max_iter, residual, test, "达到最大迭代次数")
