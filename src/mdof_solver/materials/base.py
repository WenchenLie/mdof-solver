from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any


@dataclass(frozen=True)
class MaterialTerm:
    coefficient: float
    material: "UniaxialMaterial"


class StiffnessExpression:
    """A scalar constant plus references to uniaxial-material templates."""

    def __init__(self, constant: Real = 0.0, terms: list[MaterialTerm] | None = None):
        self.constant = float(constant)
        self.terms = list(terms or [])

    @classmethod
    def coerce(cls, value: Any) -> "StiffnessExpression":
        if isinstance(value, cls):
            return value
        if isinstance(value, UniaxialMaterial):
            return cls(terms=[MaterialTerm(1.0, value)])
        if isinstance(value, Real):
            return cls(value)
        raise TypeError(f"Unsupported stiffness matrix entry type: {type(value).__name__}")

    def _combine(self, other: Any, sign: float) -> "StiffnessExpression":
        rhs = self.coerce(other)
        return StiffnessExpression(
            self.constant + sign * rhs.constant,
            self.terms + [MaterialTerm(sign * x.coefficient, x.material) for x in rhs.terms],
        )

    def __add__(self, other: Any) -> "StiffnessExpression":
        return self._combine(other, 1.0)

    def __radd__(self, other: Any) -> "StiffnessExpression":
        return self.coerce(other)._combine(self, 1.0)

    def __sub__(self, other: Any) -> "StiffnessExpression":
        return self._combine(other, -1.0)

    def __rsub__(self, other: Any) -> "StiffnessExpression":
        return self.coerce(other)._combine(self, -1.0)

    def __neg__(self) -> "StiffnessExpression":
        return self * -1.0

    def __mul__(self, factor: Real) -> "StiffnessExpression":
        if not isinstance(factor, Real):
            return NotImplemented
        value = float(factor)
        return StiffnessExpression(
            self.constant * value,
            [MaterialTerm(value * x.coefficient, x.material) for x in self.terms],
        )

    __rmul__ = __mul__

    def __truediv__(self, factor: Real) -> "StiffnessExpression":
        return self * (1.0 / float(factor))


def validate_tag(tag: Any) -> int:
    if isinstance(tag, bool) or not isinstance(tag, Integral):
        raise TypeError("Material tag must be an integer")
    return int(tag)


class UniaxialMaterial(ABC):
    """OpenSees-style public protocol required by the solver.

    A material object placed in a stiffness matrix is a template. Each inferred
    physical connection receives a private copy and therefore private history.
    """

    tag: int

    def _expr(self) -> StiffnessExpression:
        return StiffnessExpression(terms=[MaterialTerm(1.0, self)])

    def __add__(self, other: Any) -> StiffnessExpression:
        return self._expr() + other

    def __radd__(self, other: Any) -> StiffnessExpression:
        return StiffnessExpression.coerce(other) + self._expr()

    def __sub__(self, other: Any) -> StiffnessExpression:
        return self._expr() - other

    def __rsub__(self, other: Any) -> StiffnessExpression:
        return StiffnessExpression.coerce(other) - self._expr()

    def __neg__(self) -> StiffnessExpression:
        return -self._expr()

    def __mul__(self, factor: Real) -> StiffnessExpression:
        return self._expr() * factor

    __rmul__ = __mul__

    @abstractmethod
    def setTrialStrain(self, strain: float, strainRate: float = 0.0) -> None: ...

    @abstractmethod
    def getStrain(self) -> float: ...

    @abstractmethod
    def getStress(self) -> float: ...

    @abstractmethod
    def getTangent(self) -> float: ...

    @abstractmethod
    def getInitialTangent(self) -> float: ...

    @abstractmethod
    def commitState(self) -> None: ...

    @abstractmethod
    def revertToLastCommit(self) -> None: ...

    @abstractmethod
    def revertToStart(self) -> None: ...

    @abstractmethod
    def getCopy(self) -> "UniaxialMaterial": ...

    def setStrain(self, strain: float, strainRate: float = 0.0) -> None:
        self.setTrialStrain(strain, strainRate)
        self.commitState()
