from __future__ import annotations

from dataclasses import dataclass

from .base import UniaxialMaterial, validate_tag


@dataclass
class Elastic(UniaxialMaterial):
    tag: int
    k: float

    def __post_init__(self) -> None:
        self.tag = validate_tag(self.tag)
        if not self.k > 0:
            raise ValueError("Elastic.k must be greater than zero")
        self.revertToStart()

    def setTrialStrain(self, strain: float, strainRate: float = 0.0) -> None:
        self.Tstrain = float(strain)
        self.Tstress = self.k * self.Tstrain

    def getStrain(self) -> float:
        return self.Tstrain

    def getStress(self) -> float:
        return self.Tstress

    def getTangent(self) -> float:
        return self.k

    def getInitialTangent(self) -> float:
        return self.k

    def commitState(self) -> None:
        self.Cstrain = self.Tstrain
        self.Cstress = self.Tstress

    def revertToLastCommit(self) -> None:
        self.Tstrain = self.Cstrain
        self.Tstress = self.Cstress

    def revertToStart(self) -> None:
        self.Cstrain = self.Tstrain = 0.0
        self.Cstress = self.Tstress = 0.0

    def getCopy(self) -> "Elastic":
        other = Elastic(self.tag, self.k)
        other.Cstrain, other.Tstrain = self.Cstrain, self.Tstrain
        other.Cstress, other.Tstress = self.Cstress, self.Tstress
        return other
