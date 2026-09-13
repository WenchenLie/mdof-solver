from __future__ import annotations

import copy
import sys

from .base import UniaxialMaterial, validate_tag


class ModTakeda(UniaxialMaterial):
    """Modified Takeda rule ported from MyOpenSees."""

    def __init__(
        self,
        tag: int,
        Fy: float,
        k0: float,
        r: float,
        alpha: float,
        beta: float,
    ):
        self.tag = validate_tag(tag)
        self.Fy = float(Fy)
        self.k0 = float(k0)
        self.r = float(r)
        self.alpha = float(alpha)
        self.beta = float(beta)
        if self.Fy <= 0 or self.k0 <= 0:
            raise ValueError("ModTakeda requires Fy > 0 and k0 > 0")
        if min(self.r, self.alpha, self.beta) < 0:
            raise ValueError("ModTakeda requires r, alpha, and beta to be nonnegative")
        self.revertToStart()

    def setTrialStrain(self, strain: float, strainRate: float = 0.0) -> None:
        self.revertToLastCommit()
        self.Tstrain = float(strain)
        dStrain = self.Tstrain - self.Cstrain
        if abs(dStrain) > sys.float_info.epsilon:
            if dStrain > 0:
                u_flag = max(self.uy, self.Cdm_pos - self.beta * (self.Cdm_pos - self.uy))
                F_flag = max(self.Fy, self.CFm_pos - self.beta * (self.Cdm_pos - self.uy) * self.r * self.k0)
                if self.Cstress < 0:
                    ku = self.k0 * abs(self.uy / self.Cdm_pos) ** self.alpha
                    ku = max(ku, abs((self.Cstress + sys.float_info.epsilon) / (self.Cstrain + sys.float_info.epsilon)))
                    if self.Cstress + ku * dStrain > 0:
                        dStrain1 = -self.Cstress / ku
                        dStrain2 = dStrain - dStrain1
                        u0 = self.Cstrain + dStrain1
                        kr = F_flag / (u_flag - u0) if self.Tstrain < u_flag else self.k0
                        self.Tstress = kr * dStrain2
                    else:
                        self.Tstress = self.Cstress + ku * dStrain
                else:
                    if u_flag > self.Tstrain:
                        kr = (F_flag - self.Cstress) / (u_flag - self.Cstrain)
                        self.Tstress = self.Cstress + kr * dStrain
                    else:
                        self.Tstress = self.Cstress + dStrain * self.k0
                envelope = self.r * self.k0 * (self.Tstrain - self.uy) + self.Fy
                self.Tstress = min(self.Tstress, envelope)
            else:
                u_flag = min(-self.uy, self.Cdm_neg - self.beta * (self.Cdm_neg + self.uy))
                F_flag = min(-self.Fy, self.CFm_neg - self.beta * (self.Cdm_neg + self.uy) * self.r * self.k0)
                if self.Cstress > 0:
                    ku = self.k0 * abs(self.uy / self.Cdm_neg) ** self.alpha
                    ku = max(ku, abs((self.Cstress + sys.float_info.epsilon) / (self.Cstrain + sys.float_info.epsilon)))
                    if self.Cstress + ku * dStrain < 0:
                        dStrain1 = -self.Cstress / ku
                        dStrain2 = dStrain - dStrain1
                        u0 = self.Cstrain + dStrain1
                        kr = F_flag / (u_flag - u0) if self.Tstrain > u_flag else self.k0
                        self.Tstress = kr * dStrain2
                    else:
                        self.Tstress = self.Cstress + ku * dStrain
                else:
                    if u_flag < self.Tstrain:
                        kr = (F_flag - self.Cstress) / (u_flag - self.Cstrain)
                        self.Tstress = self.Cstress + kr * dStrain
                    else:
                        self.Tstress = self.Cstress + dStrain * self.k0
                envelope = self.r * self.k0 * (self.Tstrain + self.uy) - self.Fy
                self.Tstress = max(self.Tstress, envelope)
            if dStrain > 0:
                self.Tdm_pos = max(self.Tdm_pos, self.Tstrain)
                self.TFm_pos = max(self.TFm_pos, self.Tstress)
            else:
                self.Tdm_neg = min(self.Tdm_neg, self.Tstrain)
                self.TFm_neg = min(self.TFm_neg, self.Tstress)
            self.Ttangent = (self.Tstress - self.Cstress) / dStrain
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
        return self.k0

    def commitState(self) -> None:
        self.Cstrain, self.Cstress, self.Ctangent = self.Tstrain, self.Tstress, self.Ttangent
        self.Cdm_pos, self.Cdm_neg = self.Tdm_pos, self.Tdm_neg
        self.CFm_pos, self.CFm_neg = self.TFm_pos, self.TFm_neg
        self.step += 1

    def revertToLastCommit(self) -> None:
        self.Tstrain, self.Tstress, self.Ttangent = self.Cstrain, self.Cstress, self.Ctangent
        self.Tdm_pos, self.Tdm_neg = self.Cdm_pos, self.Cdm_neg
        self.TFm_pos, self.TFm_neg = self.CFm_pos, self.CFm_neg

    def revertToStart(self) -> None:
        self.Cstrain = self.Tstrain = 0.0
        self.Cstress = self.Tstress = 0.0
        self.Ctangent = self.Ttangent = self.k0
        self.uy = self.Fy / self.k0
        self.Cdm_pos = self.Tdm_pos = self.uy
        self.Cdm_neg = self.Tdm_neg = -self.uy
        self.CFm_pos = self.TFm_pos = self.Fy
        self.CFm_neg = self.TFm_neg = -self.Fy
        self.step = 1

    def getCopy(self) -> "ModTakeda":
        result = object.__new__(type(self))
        result.__dict__ = copy.deepcopy(self.__dict__)
        return result
