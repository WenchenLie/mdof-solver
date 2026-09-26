from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real

from .base import UniaxialMaterial, validate_tag


class Parallel(UniaxialMaterial):
    """Independent uniaxial materials sharing one strain (OpenSees Parallel)."""

    def __init__(
        self,
        tag: int,
        *materials: UniaxialMaterial,
        factors: Sequence[Real] | None = None,
    ) -> None:
        self.tag = validate_tag(tag)
        if len(materials) == 1 and isinstance(materials[0], (list, tuple)):
            materials = tuple(materials[0])
        if not materials:
            raise ValueError("Parallel requires at least one material")
        if any(not isinstance(material, UniaxialMaterial) for material in materials):
            raise TypeError("Parallel children must be uniaxial materials")
        if factors is None:
            factors = (1.0,) * len(materials)
        if len(factors) != len(materials):
            raise ValueError("Parallel factors must match the number of materials")
        self.factors = tuple(float(factor) for factor in factors)
        if not all(math.isfinite(factor) for factor in self.factors):
            raise ValueError("Parallel factors must be finite")
        # A wrapped material is a template too: each child needs private history.
        self.materials = tuple(material.getCopy() for material in materials)
        self.Cstrain = self.Tstrain = 0.0
        self.CstrainRate = self.TstrainRate = 0.0

    def setTrialStrain(self, strain: float, strainRate: float = 0.0) -> None:
        strain, strainRate = float(strain), float(strainRate)
        if not math.isfinite(strain) or not math.isfinite(strainRate):
            raise ValueError("Parallel trial strain and rate must be finite")
        try:
            for material in self.materials:
                material.revertToLastCommit()
                material.setTrialStrain(strain, strainRate)
        except Exception:
            self.revertToLastCommit()
            raise
        self.Tstrain, self.TstrainRate = strain, strainRate

    def getStrain(self) -> float:
        return self.Tstrain

    def getStress(self) -> float:
        return sum(factor * material.getStress()
                   for factor, material in zip(self.factors, self.materials, strict=True))

    def getTangent(self) -> float:
        return sum(factor * material.getTangent()
                   for factor, material in zip(self.factors, self.materials, strict=True))

    def getInitialTangent(self) -> float:
        return sum(factor * material.getInitialTangent()
                   for factor, material in zip(self.factors, self.materials, strict=True))

    def commitState(self) -> None:
        checkpoints = tuple(material.getCopy() for material in self.materials)
        try:
            for material in self.materials:
                material.commitState()
        except Exception:
            self.materials = checkpoints
            raise
        self.Cstrain, self.CstrainRate = self.Tstrain, self.TstrainRate

    def revertToLastCommit(self) -> None:
        for material in self.materials:
            material.revertToLastCommit()
        self.Tstrain, self.TstrainRate = self.Cstrain, self.CstrainRate

    def revertToStart(self) -> None:
        for material in self.materials:
            material.revertToStart()
        self.Cstrain = self.Tstrain = 0.0
        self.CstrainRate = self.TstrainRate = 0.0

    def getCopy(self) -> Parallel:
        result = object.__new__(type(self))
        result.tag = self.tag
        result.factors = self.factors
        result.materials = tuple(material.getCopy() for material in self.materials)
        result.Cstrain, result.Tstrain = self.Cstrain, self.Tstrain
        result.CstrainRate, result.TstrainRate = self.CstrainRate, self.TstrainRate
        return result
