from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class KinematicState:
    u: NDArray[np.float64]
    v: NDArray[np.float64]
    a: NDArray[np.float64]


@dataclass(frozen=True)
class Newmark:
    gamma: float = 0.5
    beta: float = 0.25

    def __post_init__(self) -> None:
        if self.gamma <= 0 or self.beta <= 0:
            raise ValueError("Newmark gamma and beta must be greater than zero")

    def prepare(self, state: KinematicState, dt: float):
        h, beta, gamma = dt, self.beta, self.gamma
        u_predictor = state.u + h * state.v + h * h * (0.5 - beta) * state.a
        v_predictor = state.v + h * (1.0 - gamma) * state.a
        ca = 1.0 / (beta * h * h)
        cv = gamma / (beta * h)

        def endpoint(u):
            a = ca * (u - u_predictor)
            v = v_predictor + gamma * h * a
            return KinematicState(np.asarray(u), v, a)

        return endpoint, 1.0, cv, ca, 1.0


@dataclass(frozen=True)
class HHT:
    """Hilber-Hughes-Taylor integrator using OpenSees' positive alpha."""

    alpha: float = 0.9
    gamma: float | None = None
    beta: float | None = None

    def __post_init__(self) -> None:
        if not 2.0 / 3.0 <= self.alpha <= 1.0:
            raise ValueError("HHT.alpha must be in [2/3, 1]")
        gamma = 1.5 - self.alpha if self.gamma is None else self.gamma
        beta = (2.0 - self.alpha) ** 2 / 4.0 if self.beta is None else self.beta
        if gamma <= 0 or beta <= 0:
            raise ValueError("HHT gamma and beta must be greater than zero")
        object.__setattr__(self, "gamma", float(gamma))
        object.__setattr__(self, "beta", float(beta))

    def prepare(self, state: KinematicState, dt: float):
        base = Newmark(self.gamma, self.beta)
        endpoint, _, cv, ca, _ = base.prepare(state, dt)
        alpha = self.alpha

        def evaluated(u):
            end = endpoint(u)
            return KinematicState(
                (1.0 - alpha) * state.u + alpha * end.u,
                (1.0 - alpha) * state.v + alpha * end.v,
                end.a,
            )

        return endpoint, alpha, alpha * cv, ca, alpha, evaluated


@dataclass(frozen=True)
class CentralDifference:
    """Fixed-step explicit central difference in full-state form."""

    pass
