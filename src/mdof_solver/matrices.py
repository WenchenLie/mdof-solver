from __future__ import annotations

from numbers import Real
from typing import Any, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .materials import StiffnessExpression, UniaxialMaterial


def _square_float_array(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


class NumericMatrix:
    _name = "Matrix"

    def __init__(self, values: ArrayLike):
        self.values = _square_float_array(values, self._name)

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape

    def __array__(self, dtype: Any = None) -> NDArray[np.float64]:
        return np.asarray(self.values, dtype=dtype)

    def _new(self, values: NDArray[np.float64]) -> "NumericMatrix":
        return type(self)(values)

    def __add__(self, other: Any) -> "NumericMatrix":
        rhs = other.values if isinstance(other, type(self)) else _square_float_array(other, self._name)
        return self._new(self.values + rhs)

    __radd__ = __add__

    def __sub__(self, other: Any) -> "NumericMatrix":
        rhs = other.values if isinstance(other, type(self)) else _square_float_array(other, self._name)
        return self._new(self.values - rhs)

    def __mul__(self, factor: Real) -> "NumericMatrix":
        return self._new(self.values * float(factor))

    __rmul__ = __mul__


class MassMatrix(NumericMatrix):
    _name = "Mass matrix"

    def __init__(self, values: ArrayLike):
        super().__init__(values)
        if not np.allclose(self.values, self.values.T, rtol=1e-10, atol=1e-12):
            raise ValueError("The mass matrix must be symmetric")
        try:
            np.linalg.cholesky(self.values)
        except np.linalg.LinAlgError as exc:
            raise ValueError("The mass matrix must be positive definite") from exc


class DampingMatrix(NumericMatrix):
    _name = "Damping matrix"

    def __init__(self, values: ArrayLike):
        super().__init__(values)
        if not np.allclose(self.values, self.values.T, rtol=1e-10, atol=1e-12):
            raise ValueError("The damping matrix must be symmetric")


class StiffnessMatrix:
    """A square matrix containing numbers and material expressions."""

    def __init__(self, values: Iterable[Iterable[Any]]):
        rows = [list(row) for row in values]
        if not rows or any(len(row) != len(rows) for row in rows):
            raise ValueError("The stiffness matrix must be a nonempty square matrix")
        self.entries = np.empty((len(rows), len(rows)), dtype=object)
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                self.entries[i, j] = StiffnessExpression.coerce(value)

    @property
    def shape(self) -> tuple[int, int]:
        return self.entries.shape

    def _binary(self, other: Any, subtract: bool = False) -> "StiffnessMatrix":
        rhs = other if isinstance(other, StiffnessMatrix) else StiffnessMatrix(other)
        if rhs.shape != self.shape:
            raise ValueError("Stiffness matrix shapes do not match")
        sign = -1.0 if subtract else 1.0
        return StiffnessMatrix([
            [self.entries[i, j] + sign * rhs.entries[i, j] for j in range(self.shape[1])]
            for i in range(self.shape[0])
        ])

    def __add__(self, other: Any) -> "StiffnessMatrix":
        return self._binary(other)

    __radd__ = __add__

    def __sub__(self, other: Any) -> "StiffnessMatrix":
        return self._binary(other, subtract=True)

    def __mul__(self, factor: Real) -> "StiffnessMatrix":
        return StiffnessMatrix([
            [self.entries[i, j] * factor for j in range(self.shape[1])]
            for i in range(self.shape[0])
        ])

    __rmul__ = __mul__

    def split(self) -> tuple[NDArray[np.float64], list[tuple[UniaxialMaterial, NDArray[np.float64]]]]:
        """Return the numeric matrix and one coefficient matrix per template identity."""
        n = self.shape[0]
        numeric = np.zeros((n, n), dtype=float)
        order: list[int] = []
        templates: dict[int, UniaxialMaterial] = {}
        coefficients: dict[int, NDArray[np.float64]] = {}
        for i in range(n):
            for j in range(n):
                expression: StiffnessExpression = self.entries[i, j]
                numeric[i, j] += expression.constant
                for term in expression.terms:
                    key = id(term.material)
                    if key not in templates:
                        order.append(key)
                        templates[key] = term.material
                        coefficients[key] = np.zeros((n, n), dtype=float)
                    coefficients[key][i, j] += term.coefficient
        if not np.all(np.isfinite(numeric)):
            raise ValueError("The stiffness matrix contains non-finite numeric values")
        return numeric, [(templates[key], coefficients[key]) for key in order]
