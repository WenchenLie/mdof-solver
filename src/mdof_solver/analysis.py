from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .algorithms import KrylovNewton, Linear, Newton, NewtonLineSearch, SolutionAlgorithm
from .convergence import ConvergenceTest, NormUnbalance
from .exceptions import AnalysisFailure
from .integrators import CentralDifference, HHT, KinematicState, Newmark
from .materials import Elastic
from .model import RuntimeModel, System
from .recording import AnalysisResult, Recorder, ResponseCollector


class _ImplicitProblem:
    def __init__(
        self, system: System, runtime: RuntimeModel, state: KinematicState,
        time: float, dt: float, integrator, convergence_test: ConvergenceTest,
        rtol: float, atol: float,
        external_load: Callable[[float], NDArray[np.float64]],
    ):
        self.system, self.runtime, self.state = system, runtime, state
        self.time, self.dt, self.integrator = time, dt, integrator
        prepared = integrator.prepare(state, dt)
        self.endpoint = prepared[0]
        if isinstance(integrator, HHT):
            _, self.k_factor, self.cv, self.ca, time_factor, self.evaluated = prepared
        else:
            _, self.k_factor, self.cv, self.ca, time_factor = prepared
            self.evaluated = self.endpoint
        self.load_time = time + time_factor * dt
        self.last_ele_force = np.empty(len(runtime.eles))
        self.convergence_test = convergence_test
        self.rtol, self.atol = rtol, atol
        self.external_load = external_load

    def evaluate(self, u: NDArray[np.float64]):
        trial = self.evaluated(u)
        restoring, stiffness, forces = self.runtime.evaluate(trial.u, trial.v)
        load = self.external_load(self.load_time)
        inertia = self.system.mass.values @ trial.a
        damping = self.system.damping.values @ trial.v
        residual = load - inertia - damping - restoring
        tangent = self.ca * self.system.mass.values + self.cv * self.system.damping.values + self.k_factor * stiffness
        norm_type = getattr(self.convergence_test, "norm_type", 2.0)
        scale = max(
            1.0,
            *(float(np.linalg.norm(x, ord=norm_type)) for x in (load, inertia, damping, restoring)),
        )
        self.last_ele_force = forces
        return residual, tangent, scale

    def check_convergence(self, residual, increment, solution, force_scale):
        return self.convergence_test.evaluate(
            residual, increment, solution, force_scale, self.rtol, self.atol,
        )


@dataclass
class TransientAnalysis:
    system: System
    integrator: Newmark | HHT | CentralDifference = Newmark()
    algorithms: tuple[SolutionAlgorithm, ...] = (Newton(), NewtonLineSearch(), KrylovNewton())
    rtol: float = 1e-6
    atol: float = 1e-8
    max_iter: int = 30
    convergence_test: ConvergenceTest = NormUnbalance()
    min_factor: float = 1e-6
    shrink_factor: float = 0.5
    growth_factor: float = 2.0
    growth_after: int = 2

    def __post_init__(self) -> None:
        if self.rtol < 0 or self.atol < 0 or self.max_iter <= 0:
            raise ValueError("Invalid convergence parameters")
        if not isinstance(self.convergence_test, ConvergenceTest):
            raise TypeError("convergence_test must be a ConvergenceTest instance")
        if not 0 < self.min_factor <= 1 or not 0 < self.shrink_factor < 1:
            raise ValueError("Invalid time-step reduction parameters")
        if self.growth_factor <= 1 or self.growth_after <= 0:
            raise ValueError("Invalid time-step recovery parameters")
        if not self.algorithms:
            raise ValueError("At least one solution algorithm is required")

    def run(
        self,
        dt: float,
        t_end: float | None = None,
        *,
        t_start: float | None = None,
        free_vibration_duration: float = 0.0,
        u0: ArrayLike | None = None,
        v0: ArrayLike | None = None,
        recorder: Recorder | None = None,
    ) -> AnalysisResult:
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and greater than zero")
        start = self.system.load.t_min if t_start is None else float(t_start)
        load_end = self.system.load.t_max if t_end is None else float(t_end)
        free_duration = float(free_vibration_duration)
        if (
            not np.isfinite(start)
            or not np.isfinite(load_end)
            or start < self.system.load.t_min
            or load_end > self.system.load.t_max
            or load_end <= start
        ):
            raise ValueError(
                "The forced-vibration interval must lie within the load range, "
                "and its end time must be later than its start time"
            )
        if not np.isfinite(free_duration) or free_duration < 0:
            raise ValueError("free_vibration_duration must be finite and nonnegative")
        end = load_end + free_duration
        if not np.isfinite(end):
            raise ValueError("The total analysis end time must be finite")
        n = self.system.ndof
        u = np.zeros(n) if u0 is None else np.asarray(u0, dtype=float).copy()
        v = np.zeros(n) if v0 is None else np.asarray(v0, dtype=float).copy()
        if u.shape != (n,) or v.shape != (n,) or not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
            raise ValueError("Initial displacement and velocity must be finite DOF vectors")
        runtime = self.system.create_runtime()
        if not isinstance(self.integrator, CentralDifference) and any(
            isinstance(algorithm, Linear) for algorithm in self.algorithms
        ) and any(not isinstance(x.material, Elastic) for x in runtime.eles):
            raise ValueError("Linear cannot be a candidate algorithm for implicit nonlinear analysis")
        zero_load = np.zeros(n)

        def forced_load(time: float) -> NDArray[np.float64]:
            return self.system.load(min(time, load_end))

        def free_load(time: float) -> NDArray[np.float64]:
            return zero_load

        if isinstance(self.integrator, CentralDifference):
            ratio = (end - start) / dt
            if not np.isclose(ratio, round(ratio), rtol=1e-10, atol=1e-12):
                raise ValueError("The central-difference analysis end must lie on the fixed time-step grid")
            load_end_ratio = (load_end - start) / dt
            if not np.isclose(load_end_ratio, round(load_end_ratio), rtol=1e-10, atol=1e-12):
                raise ValueError("The central-difference load end must lie on the fixed time-step grid")
            if recorder is not None and recorder.interval is not None:
                interval_ratio = recorder.interval / dt
                if interval_ratio < 1 or not np.isclose(interval_ratio, round(interval_ratio), rtol=1e-10, atol=1e-12):
                    raise ValueError("The central-difference recording interval must be an integer multiple of dt")
        restoring, _, forces = runtime.evaluate(u, v)
        a = np.linalg.solve(self.system.mass.values, forced_load(start) - self.system.damping.values @ v - restoring)
        runtime.commit()
        state = KinematicState(u, v, a)
        collector = ResponseCollector(recorder or Recorder(), n, tuple(x.id for x in self.system.eles))
        collector.record(start, u, v, a, forces, runtime.deformations(u))
        if isinstance(self.integrator, CentralDifference):
            return self._run_central(
                state, start, end, load_end, dt, runtime, collector, forced_load, free_load,
            )
        return self._run_implicit(
            state, start, end, load_end, dt, runtime, collector, forced_load, free_load,
        )

    def _run_central(
        self, state, time, end, load_end, dt, runtime, collector, forced_load, free_load,
    ):
        ratio = (end - time) / dt
        if not np.isclose(ratio, round(ratio), rtol=1e-10, atol=1e-12):
            raise ValueError("The central-difference analysis end must lie on the fixed time-step grid")
        mass, damping = self.system.mass.values, self.system.damping.values
        effective_mass = mass + 0.5 * dt * damping
        steps = int(round(ratio))
        forced_steps = int(round((load_end - time) / dt))
        record_every = 1
        if collector.config.interval is not None:
            interval_ratio = collector.config.interval / dt
            if interval_ratio < 1 or not np.isclose(interval_ratio, round(interval_ratio), rtol=1e-10, atol=1e-12):
                raise ValueError("The central-difference recording interval must be an integer multiple of dt")
            record_every = int(round(interval_ratio))
        try:
            for step_number in range(1, steps + 1):
                next_time = time + dt
                external_load = forced_load if step_number <= forced_steps else free_load
                u_next = state.u + dt * state.v + 0.5 * dt * dt * state.a
                half_velocity = state.v + 0.5 * dt * state.a
                restoring, _, _ = runtime.evaluate(u_next, half_velocity)
                a_next = np.linalg.solve(effective_mass, external_load(next_time) - restoring - damping @ half_velocity)
                v_next = half_velocity + 0.5 * dt * a_next
                restoring, _, forces = runtime.evaluate(u_next, v_next)
                residual_norm = float(np.linalg.norm(
                    external_load(next_time) - mass @ a_next - damping @ v_next - restoring
                ))
                runtime.commit()
                state = KinematicState(u_next, v_next, a_next)
                time = next_time
                collector.record(time, state.u, state.v, state.a, forces, runtime.deformations(state.u),
                                 store_history=(step_number % record_every == 0 or step_number == steps))
                collector.diagnostic(time=time, dt=dt, algorithm="Linear", iterations=1,
                                     residual=residual_norm, status="accepted")
        except Exception as exc:
            runtime.revert()
            partial = collector.finish(False, str(exc))
            raise AnalysisFailure(f"Central-difference analysis failed after t={time:g}: {exc}", partial) from exc
        return collector.finish()

    def _attempt(self, state, time, dt, runtime, algorithm, external_load):
        problem = _ImplicitProblem(
            self.system, runtime, state, time, dt, self.integrator,
            self.convergence_test, self.rtol, self.atol,
            external_load,
        )
        guess = state.u + dt * state.v + 0.5 * dt * dt * state.a
        outcome = algorithm.solve(problem, guess, self.rtol, self.atol, self.max_iter)
        if not outcome.converged:
            runtime.revert()
            return outcome, None, None
        endpoint = problem.endpoint(outcome.x)
        try:
            _, _, forces = runtime.evaluate(endpoint.u, endpoint.v)
            runtime.commit()
        except Exception as exc:
            runtime.revert()
            return type(outcome)(False, outcome.x, outcome.iterations, outcome.residual_norm, str(exc)), None, None
        return outcome, endpoint, forces

    def _run_implicit(
        self, state, time, end, load_end, initial_dt, runtime, collector, forced_load, free_load,
    ):
        minimum = initial_dt * self.min_factor
        next_boundary = min(time + initial_dt, end)
        current_dt = initial_dt
        algorithm_index = 0
        successes = 0
        epsilon = 1e-12 * max(1.0, abs(end))
        next_record = end + 1.0 if collector.config.interval is None else min(time + collector.config.interval, end)
        while time < end - epsilon:
            remaining = next_boundary - time
            step = min(current_dt, remaining, next_record - time)
            if time < load_end - epsilon:
                step = min(step, load_end - time)
            if step <= epsilon:
                next_boundary = min(next_boundary + initial_dt, end)
                algorithm_index = 0
                continue
            external_load = forced_load if time < load_end - epsilon else free_load
            outcome, endpoint, forces = self._attempt(
                state, time, step, runtime, self.algorithms[algorithm_index], external_load,
            )
            if outcome.converged:
                time = min(time + step, end)
                state = endpoint
                at_record = collector.config.interval is None or time >= next_record - epsilon or time >= end - epsilon
                collector.record(
                    time, state.u, state.v, state.a, forces, runtime.deformations(state.u),
                    store_history=at_record,
                )
                if collector.config.interval is not None and time >= next_record - epsilon:
                    next_record = min(next_record + collector.config.interval, end)
                collector.diagnostic(time=time, dt=step, algorithm=self.algorithms[algorithm_index].name,
                                     iterations=outcome.iterations, residual=outcome.residual_norm,
                                     test=outcome.test_name, test_value=outcome.test_value,
                                     test_limit=outcome.test_limit, status="accepted")
                successes += 1
                if successes >= self.growth_after:
                    current_dt = min(initial_dt, current_dt * self.growth_factor)
                    successes = 0
                if time >= next_boundary - epsilon:
                    next_boundary = min(next_boundary + initial_dt, end)
                    algorithm_index = 0
                continue
            collector.diagnostic(time=time, dt=step, algorithm=self.algorithms[algorithm_index].name,
                                 iterations=outcome.iterations, residual=outcome.residual_norm,
                                 test=outcome.test_name, test_value=outcome.test_value,
                                 test_limit=outcome.test_limit,
                                 status="rejected", reason=outcome.reason)
            successes = 0
            if step > minimum * (1.0 + 1e-12):
                current_dt = max(minimum, step * self.shrink_factor)
                continue
            algorithm_index += 1
            if algorithm_index >= len(self.algorithms):
                message = (
                    f"All candidate algorithms failed after t={time:g} at the minimum "
                    f"time step {minimum:g}; last reason: {outcome.reason}"
                )
                partial = collector.finish(False, message)
                raise AnalysisFailure(message, partial)
        return collector.finish()
