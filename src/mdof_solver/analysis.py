from __future__ import annotations

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

    def evaluate(self, u: NDArray[np.float64]):
        trial = self.evaluated(u)
        restoring, stiffness, forces = self.runtime.evaluate(trial.u, trial.v)
        load = self.system.load(self.load_time)
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
            raise ValueError("收敛参数无效")
        if not isinstance(self.convergence_test, ConvergenceTest):
            raise TypeError("convergence_test 必须是 ConvergenceTest 实例")
        if not 0 < self.min_factor <= 1 or not 0 < self.shrink_factor < 1:
            raise ValueError("步长缩小参数无效")
        if self.growth_factor <= 1 or self.growth_after <= 0:
            raise ValueError("步长恢复参数无效")
        if not self.algorithms:
            raise ValueError("至少需要一种求解算法")

    def run(
        self,
        dt: float,
        t_end: float | None = None,
        *,
        t_start: float | None = None,
        u0: ArrayLike | None = None,
        v0: ArrayLike | None = None,
        recorder: Recorder | None = None,
    ) -> AnalysisResult:
        if dt <= 0:
            raise ValueError("dt 必须大于 0")
        start = self.system.load.t_min if t_start is None else float(t_start)
        end = self.system.load.t_max if t_end is None else float(t_end)
        if start < self.system.load.t_min or end > self.system.load.t_max or end <= start:
            raise ValueError("分析时间必须位于荷载有效范围内且终点晚于起点")
        n = self.system.ndof
        u = np.zeros(n) if u0 is None else np.asarray(u0, dtype=float).copy()
        v = np.zeros(n) if v0 is None else np.asarray(v0, dtype=float).copy()
        if u.shape != (n,) or v.shape != (n,) or not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
            raise ValueError("初始位移和速度必须是有限的自由度向量")
        runtime = self.system.create_runtime()
        if not isinstance(self.integrator, CentralDifference) and any(
            isinstance(algorithm, Linear) for algorithm in self.algorithms
        ) and any(not isinstance(x.material, Elastic) for x in runtime.eles):
            raise ValueError("隐式非线性分析的候选算法不能包含 Linear")
        if isinstance(self.integrator, CentralDifference):
            ratio = (end - start) / dt
            if not np.isclose(ratio, round(ratio), rtol=1e-10, atol=1e-12):
                raise ValueError("中心差分分析终点必须位于固定步长网格上")
            if recorder is not None and recorder.interval is not None:
                interval_ratio = recorder.interval / dt
                if interval_ratio < 1 or not np.isclose(interval_ratio, round(interval_ratio), rtol=1e-10, atol=1e-12):
                    raise ValueError("中心差分的记录间隔必须是计算步长的整数倍")
        restoring, _, forces = runtime.evaluate(u, v)
        a = np.linalg.solve(self.system.mass.values, self.system.load(start) - self.system.damping.values @ v - restoring)
        runtime.commit()
        state = KinematicState(u, v, a)
        collector = ResponseCollector(recorder or Recorder(), n, tuple(x.id for x in self.system.eles))
        collector.record(start, u, v, a, forces, runtime.deformations(u))
        if isinstance(self.integrator, CentralDifference):
            return self._run_central(state, start, end, dt, runtime, collector)
        return self._run_implicit(state, start, end, dt, runtime, collector)

    def _run_central(self, state, time, end, dt, runtime, collector):
        ratio = (end - time) / dt
        if not np.isclose(ratio, round(ratio), rtol=1e-10, atol=1e-12):
            raise ValueError("中心差分分析终点必须位于固定步长网格上")
        mass, damping = self.system.mass.values, self.system.damping.values
        effective_mass = mass + 0.5 * dt * damping
        steps = int(round(ratio))
        record_every = 1
        if collector.config.interval is not None:
            interval_ratio = collector.config.interval / dt
            if interval_ratio < 1 or not np.isclose(interval_ratio, round(interval_ratio), rtol=1e-10, atol=1e-12):
                raise ValueError("中心差分的记录间隔必须是计算步长的整数倍")
            record_every = int(round(interval_ratio))
        try:
            for step_number in range(1, steps + 1):
                next_time = time + dt
                u_next = state.u + dt * state.v + 0.5 * dt * dt * state.a
                half_velocity = state.v + 0.5 * dt * state.a
                restoring, _, _ = runtime.evaluate(u_next, half_velocity)
                a_next = np.linalg.solve(effective_mass, self.system.load(next_time) - restoring - damping @ half_velocity)
                v_next = half_velocity + 0.5 * dt * a_next
                restoring, _, forces = runtime.evaluate(u_next, v_next)
                residual_norm = float(np.linalg.norm(
                    self.system.load(next_time) - mass @ a_next - damping @ v_next - restoring
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
            raise AnalysisFailure(f"中心差分分析在 t={time:g} 后失败: {exc}", partial) from exc
        return collector.finish()

    def _attempt(self, state, time, dt, runtime, algorithm):
        problem = _ImplicitProblem(
            self.system, runtime, state, time, dt, self.integrator,
            self.convergence_test, self.rtol, self.atol,
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

    def _run_implicit(self, state, time, end, initial_dt, runtime, collector):
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
            if step <= epsilon:
                next_boundary = min(next_boundary + initial_dt, end)
                algorithm_index = 0
                continue
            outcome, endpoint, forces = self._attempt(state, time, step, runtime, self.algorithms[algorithm_index])
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
                message = f"t={time:g} 后在最小步长 {minimum:g} 耗尽候选算法；最后原因: {outcome.reason}"
                partial = collector.finish(False, message)
                raise AnalysisFailure(message, partial)
        return collector.finish()
