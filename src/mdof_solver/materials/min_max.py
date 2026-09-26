from __future__ import annotations

import math

from .base import UniaxialMaterial, validate_tag


class MinMax(UniaxialMaterial):
    """Permanent strain-limit failure of another material (OpenSees MinMax)."""

    def __init__(
        self,
        tag: int,
        material: UniaxialMaterial,
        min_strain: float = -1.0e16,
        max_strain: float = 1.0e16,
    ) -> None:
        self.tag = validate_tag(tag)
        if not isinstance(material, UniaxialMaterial):
            raise TypeError("MinMax requires a uniaxial material")
        self.min_strain, self.max_strain = float(min_strain), float(max_strain)
        if not (math.isfinite(self.min_strain) and math.isfinite(self.max_strain)
                and self.min_strain < self.max_strain):
            raise ValueError("MinMax requires finite limits with min_strain < max_strain")
        self.material = material.getCopy()
        self.Cfailed = self.Tfailed = False

    def setTrialStrain(self, strain: float, strainRate: float = 0.0) -> None:
        strain, strainRate = float(strain), float(strainRate)
        if not math.isfinite(strain) or not math.isfinite(strainRate):
            raise ValueError("MinMax trial strain and rate must be finite")
        if self.Cfailed or strain <= self.min_strain or strain >= self.max_strain:
            self.Tfailed = True
            return
        self.material.revertToLastCommit()
        self.material.setTrialStrain(strain, strainRate)
        self.Tfailed = False

    def getStrain(self) -> float:
        # OpenSees exposes the last strain actually given to the inner material.
        return self.material.getStrain()

    def getStress(self) -> float:
        return 0.0 if self.Tfailed else self.material.getStress()

    def getTangent(self) -> float:
        return 1.0e-8 * self.material.getInitialTangent() if self.Tfailed else self.material.getTangent()

    def getInitialTangent(self) -> float:
        return self.material.getInitialTangent()

    def commitState(self) -> None:
        if not self.Tfailed:
            self.material.commitState()
        self.Cfailed = self.Tfailed

    def revertToLastCommit(self) -> None:
        self.material.revertToLastCommit()
        self.Tfailed = self.Cfailed

    def revertToStart(self) -> None:
        self.material.revertToStart()
        self.Cfailed = self.Tfailed = False

    def getCopy(self) -> MinMax:
        result = object.__new__(type(self))
        result.tag = self.tag
        result.material = self.material.getCopy()
        result.min_strain, result.max_strain = self.min_strain, self.max_strain
        result.Cfailed, result.Tfailed = self.Cfailed, self.Tfailed
        return result
