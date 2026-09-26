from __future__ import annotations

import copy
import math
from dataclasses import dataclass

from .base import UniaxialMaterial, validate_tag


@dataclass
class _State:
    strain: float = 0.0
    stress: float = 0.0
    tangent: float = 0.0
    inc: int = 0
    nloop: int = 0
    outp: int = 0
    outn: int = 0
    Rp: float = 0.0
    Rn: float = 0.0
    Rptwoprev: float = 0.0
    Rntwoprev: float = 0.0
    erp: float = 0.0
    sigrp: float = 0.0
    ern: float = 0.0
    sigrn: float = 0.0
    erpmaxmax: float = 0.0
    ernmaxmax: float = 0.0
    e0p: float = 0.0
    sig0p: float = 0.0
    e0n: float = 0.0
    sig0n: float = 0.0
    erptwoprev: float = 0.0
    sigrptwoprev: float = 0.0
    e0ptwoprev: float = 0.0
    sig0ptwoprev: float = 0.0
    erntwoprev: float = 0.0
    sigrntwoprev: float = 0.0
    e0ntwoprev: float = 0.0
    sig0ntwoprev: float = 0.0


class SteelMPF(UniaxialMaterial):
    """OpenSees SteelMPF Menegotto–Pinto–Filippou hysteresis.

    The response and active-branch tangent follow SteelMPF.cpp. ``sigyieldn``
    is the positive magnitude of the compressive yield stress.
    """

    def __init__(
        self, tag: int, sigyieldp: float, sigyieldn: float, E0: float,
        bp: float, bn: float, R0: float, cR1: float, cR2: float,
        a3: float = 0.0, a4: float = 1.0, a5: float = 0.0, a6: float = 1.0,
    ) -> None:
        self.tag = validate_tag(tag)
        (self.sigyieldp, self.sigyieldn, self.E0, self.bp, self.bn,
         self.R0, self.cR1, self.cR2, self.a3, self.a4, self.a5, self.a6) = (
            float(value) for value in (
                sigyieldp, sigyieldn, E0, bp, bn, R0, cR1, cR2,
                a3, a4, a5, a6,
            )
        )
        values = (self.sigyieldp, self.sigyieldn, self.E0, self.bp, self.bn,
                  self.R0, self.cR1, self.cR2, self.a3, self.a4, self.a5, self.a6)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("SteelMPF parameters must be finite")
        if not (self.sigyieldp > 0 and self.sigyieldn > 0 and self.E0 > 0 and
                0 <= self.bp < 1 and 0 <= self.bn < 1 and self.R0 > 0 and
                0 <= self.cR1 < 1 and self.cR2 > 0 and
                min(self.a3, self.a4, self.a5, self.a6) >= 0):
            raise ValueError("Invalid SteelMPF parameters")
        self.eyieldp = self.sigyieldp / self.E0
        self.eyieldn = self.sigyieldn / self.E0
        # The C++ code multiplies cR1 by R0 before reducing curvature.
        self._curvature_reduction = self.cR1 * self.R0
        self.revertToStart()

    @staticmethod
    def _curve(
        strain: float, reversal_strain: float, reversal_stress: float,
        target_strain: float, target_stress: float, curvature: float,
        hardening_ratio: float,
    ) -> tuple[float, float]:
        chord_strain = target_strain - reversal_strain
        if chord_strain == 0 or curvature <= 0:
            raise FloatingPointError("SteelMPF has an invalid reversal chord")
        normalized = (strain - reversal_strain) / chord_strain
        # Loading paths use nonnegative normalized strain. abs also keeps a
        # trial beyond a reversal chord real, with the same smooth derivative.
        power = abs(normalized) ** curvature
        denominator = (1.0 + power) ** (1.0 / curvature)
        slope = (target_stress - reversal_stress) / chord_strain
        stress = reversal_stress + (target_stress - reversal_stress) * (
            hardening_ratio * normalized + (1.0 - hardening_ratio) * normalized / denominator
        )
        tangent = slope * (
            hardening_ratio + (1.0 - hardening_ratio) / denominator *
            (1.0 - power / (1.0 + power))
        )
        return stress, tangent

    def _positive(self, state: _State, strain: float) -> tuple[float, float]:
        return self._curve(strain, state.erp, state.sigrp, state.e0p,
                           state.sig0p, state.Rp, self.bp)

    def _negative(self, state: _State, strain: float) -> tuple[float, float]:
        return self._curve(strain, state.ern, state.sigrn, state.e0n,
                           state.sig0n, state.Rn, self.bn)

    def _positive_limit(self, state: _State, strain: float) -> tuple[float, float]:
        return self._curve(strain, state.erptwoprev, state.sigrptwoprev,
                           state.e0ptwoprev, state.sig0ptwoprev,
                           state.Rptwoprev, self.bp)

    def _negative_limit(self, state: _State, strain: float) -> tuple[float, float]:
        return self._curve(strain, state.erntwoprev, state.sigrntwoprev,
                           state.e0ntwoprev, state.sig0ntwoprev,
                           state.Rntwoprev, self.bn)

    def setTrialStrain(self, strain: float, strainRate: float = 0.0) -> None:
        strain = float(strain)
        if not math.isfinite(strain):
            raise ValueError("SteelMPF trial strain must be finite")
        old = self._committed
        state = copy.copy(old)
        state.strain = strain

        if old.inc == 0:
            self._initial_trial(state)
        else:
            state.inc = 1 if strain > old.strain else -1 if strain < old.strain else old.inc
            if old.inc == 1:
                if strain < old.strain:
                    self._reverse_negative(state, old)
                else:
                    self._continue_positive(state)
            else:
                if strain > old.strain:
                    self._reverse_positive(state, old)
                else:
                    self._continue_negative(state)
        if not math.isfinite(state.stress) or not math.isfinite(state.tangent):
            raise FloatingPointError("SteelMPF returned a non-finite response")
        self._trial = state

    def _initial_trial(self, state: _State) -> None:
        state.Rptwoprev = state.Rntwoprev = self.R0
        state.outp = state.outn = 1
        state.erp = state.ern = state.sigrp = state.sigrn = 0.0
        state.erptwoprev = state.erntwoprev = 0.0
        state.sigrptwoprev = state.sigrntwoprev = 0.0
        state.erpmaxmax = state.ernmaxmax = 0.0
        state.e0p = state.e0ptwoprev = self.eyieldp
        state.sig0p = state.sig0ptwoprev = self.sigyieldp
        state.e0n = state.e0ntwoprev = -self.eyieldn
        state.sig0n = state.sig0ntwoprev = -self.sigyieldn
        state.Rp = state.Rn = self.R0
        state.inc = 1 if state.strain > 1e-14 else -1 if state.strain < -1e-14 else 0
        state.nloop = 1 if state.inc else 0
        if state.inc == 1:
            state.stress, state.tangent = self._positive(state, state.strain)
        elif state.inc == -1:
            state.stress, state.tangent = self._negative(state, state.strain)
        else:
            state.stress, state.tangent = 0.0, self.E0

    def _reverse_negative(self, state: _State, old: _State) -> None:
        state.Rntwoprev = old.Rn
        state.erntwoprev, state.sigrntwoprev = old.ern, old.sigrn
        state.e0ntwoprev, state.sig0ntwoprev = old.e0n, old.sig0n
        state.ern, state.sigrn = old.strain, old.stress
        state.nloop = old.nloop + 1
        state.ernmaxmax = max(state.ern, old.ernmaxmax)
        shift = self.sigyieldn * self.a3 * (
            abs(state.ernmaxmax) / self.eyieldp - self.a4
        )
        if abs(state.ernmaxmax) < self.eyieldp:
            shift = 0.0
        yield_stress = -self.sigyieldn - max(shift, 0.0)
        state.e0n = (yield_stress * (1.0 - self.bn) +
                     self.E0 * state.ern - state.sigrn) / (self.E0 * (1.0 - self.bn))
        state.sig0n = state.sigrn + self.E0 * (state.e0n - state.ern)
        reference = (state.e0n if state.nloop == 1 else
                     -self.eyieldn if state.nloop == 2 else state.erp)
        ksi = abs((reference - state.e0n) / self.eyieldn)
        state.Rn = min(self.R0 - self._curvature_reduction * ksi / (self.cR2 + ksi), old.Rn)
        stress, tangent = self._negative(state, state.strain)
        limit_stress, limit_tangent = self._negative_limit(state, state.strain)
        control_stress, _ = self._negative_limit(state, state.ern)
        state.outn = int(state.sigrn < control_stress)
        if state.ern < state.erntwoprev and state.outn == 0 and stress < limit_stress:
            stress, tangent = limit_stress, limit_tangent
            state.ern, state.sigrn = state.erntwoprev, state.sigrntwoprev
            state.e0n, state.sig0n, state.Rn = (
                state.e0ntwoprev, state.sig0ntwoprev, state.Rntwoprev
            )
        state.stress, state.tangent = stress, tangent

    def _reverse_positive(self, state: _State, old: _State) -> None:
        state.Rptwoprev = old.Rp
        state.erptwoprev, state.sigrptwoprev = old.erp, old.sigrp
        state.e0ptwoprev, state.sig0ptwoprev = old.e0p, old.sig0p
        state.erp, state.sigrp = old.strain, old.stress
        state.nloop = old.nloop + 1
        state.erpmaxmax = min(state.erp, old.erpmaxmax)
        shift = self.sigyieldp * self.a5 * (
            abs(state.erpmaxmax) / self.eyieldn - self.a6
        )
        if abs(state.erpmaxmax) < self.eyieldn:
            shift = 0.0
        yield_stress = self.sigyieldp + max(shift, 0.0)
        state.e0p = (yield_stress * (1.0 - self.bp) +
                     self.E0 * state.erp - state.sigrp) / (self.E0 * (1.0 - self.bp))
        state.sig0p = state.sigrp + self.E0 * (state.e0p - state.erp)
        reference = (state.e0p if state.nloop == 1 else
                     self.eyieldp if state.nloop == 2 else state.ern)
        ksi = abs((reference - state.e0p) / self.eyieldp)
        state.Rp = min(self.R0 - self._curvature_reduction * ksi / (self.cR2 + ksi), old.Rp)
        stress, tangent = self._positive(state, state.strain)
        limit_stress, limit_tangent = self._positive_limit(state, state.strain)
        control_stress, _ = self._positive_limit(state, state.erp)
        state.outp = int(state.sigrp > control_stress)
        if state.erp > state.erptwoprev and state.outp == 0 and stress > limit_stress:
            stress, tangent = limit_stress, limit_tangent
            state.erp, state.sigrp = state.erptwoprev, state.sigrptwoprev
            state.e0p, state.sig0p, state.Rp = (
                state.e0ptwoprev, state.sig0ptwoprev, state.Rptwoprev
            )
        state.stress, state.tangent = stress, tangent

    def _continue_positive(self, state: _State) -> None:
        stress, tangent = self._positive(state, state.strain)
        limit_stress, limit_tangent = self._positive_limit(state, state.strain)
        if state.erp > state.erptwoprev and state.outp == 0 and stress > limit_stress:
            stress, tangent = limit_stress, limit_tangent
            state.erp, state.sigrp = state.erptwoprev, state.sigrptwoprev
            state.e0p, state.sig0p, state.Rp = (
                state.e0ptwoprev, state.sig0ptwoprev, state.Rptwoprev
            )
        state.stress, state.tangent = stress, tangent

    def _continue_negative(self, state: _State) -> None:
        stress, tangent = self._negative(state, state.strain)
        limit_stress, limit_tangent = self._negative_limit(state, state.strain)
        if state.ern < state.erntwoprev and state.outn == 0 and stress < limit_stress:
            stress, tangent = limit_stress, limit_tangent
            state.ern, state.sigrn = state.erntwoprev, state.sigrntwoprev
            state.e0n, state.sig0n, state.Rn = (
                state.e0ntwoprev, state.sig0ntwoprev, state.Rntwoprev
            )
        state.stress, state.tangent = stress, tangent

    def getStrain(self) -> float:
        return self._trial.strain

    def getStress(self) -> float:
        return self._trial.stress

    def getTangent(self) -> float:
        return self._trial.tangent

    def getInitialTangent(self) -> float:
        return self.E0

    def commitState(self) -> None:
        self._committed = copy.copy(self._trial)

    def revertToLastCommit(self) -> None:
        self._trial = copy.copy(self._committed)

    def revertToStart(self) -> None:
        self._committed = _State(tangent=self.E0, Rp=self.R0, Rn=self.R0)
        self._trial = copy.copy(self._committed)

    def getCopy(self) -> SteelMPF:
        return copy.deepcopy(self)
