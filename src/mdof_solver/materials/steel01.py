from __future__ import annotations

import copy
import sys

from .base import UniaxialMaterial, validate_tag


class Steel01(UniaxialMaterial):
    """Bilinear Steel01 rule ported from MyOpenSees.

    The original stress path is retained. The tangent is the active branch
    derivative, following the corresponding OpenSees Steel01 implementation.
    """

    def __init__(self, tag: int, Fy: float, k: float, b: float):
        self.tag = validate_tag(tag)
        self.Fy = float(Fy)
        self.k = float(k)
        self.b = float(b)
        if self.Fy <= 0 or self.k <= 0:
            raise ValueError("Steel01 requires Fy > 0 and k > 0")
        self.revertToStart()

    def setTrialStrain(self, strain: float, strainRate: float = 0.0) -> None:
        self.revertToLastCommit()
        self.Tstrain = float(strain)
        dStrain = self.Tstrain - self.Cstrain
        if abs(dStrain) > sys.float_info.epsilon:
            self.Ttangent = self.k
            f = self.Tstress + dStrain * self.k
            if f > self.b * self.k * (self.Tstrain - self.uy) + self.Fy:
                f = self.b * self.k * (self.Tstrain - self.uy) + self.Fy
                self.Ttangent = self.k * self.b
            elif f < self.b * self.k * (self.Tstrain + self.uy) - self.Fy:
                f = self.b * self.k * (self.Tstrain + self.uy) - self.Fy
                self.Ttangent = self.k * self.b
            self.Tstress = f
        else:
            self.Tstress = self.Cstress
            self.Ttangent = self.Ctangent

    def getStrain(self) -> float:
        return self.Tstrain

    def getStress(self) -> float:
        return self.Tstress

    def getTangent(self) -> float:
        return self.Ttangent

    def getInitialTangent(self) -> float:
        return self.k

    def commitState(self) -> None:
        self.Cstrain = self.Tstrain
        self.Cstress = self.Tstress
        self.Ctangent = self.Ttangent

    def revertToLastCommit(self) -> None:
        self.Tstrain = self.Cstrain
        self.Tstress = self.Cstress
        self.Ttangent = self.Ctangent

    def revertToStart(self) -> None:
        self.Cstrain = self.Tstrain = 0.0
        self.Cstress = self.Tstress = 0.0
        self.Ctangent = self.Ttangent = self.k
        self.uy = self.Fy / self.k

    def getCopy(self) -> "Steel01":
        result = object.__new__(type(self))
        result.__dict__ = copy.deepcopy(self.__dict__)
        return result
