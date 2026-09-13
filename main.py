import numpy as np

from mdof_solver.algorithms import KrylovNewton, Newton, NewtonLineSearch
from mdof_solver.analysis import TransientAnalysis
from mdof_solver.convergence import NormUnbalance
from mdof_solver.integrators import Newmark
from mdof_solver.loading import LoadHistory
from mdof_solver.materials import Elastic, Steel01
from mdof_solver.matrices import DampingMatrix, MassMatrix, StiffnessMatrix
from mdof_solver.model import System
from mdof_solver.recording import Recorder


def main() -> None:
    # 1. Define the time step, free-vibration duration, and uniform ground-acceleration samples.
    dt = 0.01
    duration = 5.0
    free_vibration_duration = 5.0
    sample_count = round(duration / dt) + 1
    sample_index = np.arange(sample_count)
    time = sample_index * dt  # Used only to generate the example waveform; not passed to LoadHistory.
    ground_acceleration = 4.0 * np.sin(2.0 * np.pi * 1.2 * time) * np.exp(-0.35 * time)

    # 2. Build the four-DOF mass matrix.
    masses = np.array([2.0, 1.8, 1.5, 1.2])
    mass = MassMatrix(np.diag(masses))

    # 3. Convert ground acceleration into nodal inertia loads using D'Alembert's principle.
    influence = np.ones(4)
    force_pattern = mass.values @ influence
    load_samples = -ground_acceleration[:, None] * force_pattern[None, :]
    load = LoadHistory(load_samples, dt=dt)

    # 4. Define material-rule templates and an ordinary numeric stiffness.
    k1 = Steel01(tag=1, Fy=6.0, k=320.0, b=0.02)
    k2 = Steel01(tag=2, Fy=5.0, k=280.0, b=0.015)
    k3 = Elastic(tag=3, k=240.0)
    k4 = 180.0

    # 5. Enter the four-DOF stiffness matrix, including material templates.
    stiffness = StiffnessMatrix([
        [k1 + k2, -k2, 0.0, 0.0],
        [-k2, k2 + k3, -k3, 0.0],
        [0.0, -k3, k3 + k4, -k4],
        [0.0, 0.0, -k4, k4],
    ])

    # 6. Run modal analysis and build Rayleigh damping from the first two modes.
    modal_system = System(
        mass=mass,
        damping=np.zeros_like(mass.values),
        stiffness=stiffness,
        load=load,
    )
    modal = modal_system.eigen(n_modes=4, influence=influence)
    eigenvalues = modal.eigenvalues
    omega = modal.circular_frequencies
    frequencies = modal.frequencies
    periods = modal.periods
    shapes = modal.mode_shapes
    participation = modal.participation_factors
    effective_mass = modal.effective_modal_masses
    effective_mass_ratio = modal.effective_modal_mass_ratios
    omega1, omega2 = omega[:2]
    damping_ratio = 0.05
    a0 = 2.0 * damping_ratio * omega1 * omega2 / (omega1 + omega2)
    a1 = 2.0 * damping_ratio / (omega1 + omega2)
    damping = DampingMatrix(
        a0 * mass + a1 * modal_system.initial_stiffness()
    )

    # 7. Build the final dynamic system.
    system = System(
        mass=mass,
        damping=damping,
        stiffness=stiffness,
        load=load,
    )

    # 8. Define the integrator, candidate algorithms, convergence test, and analysis.
    integrator = Newmark(gamma=0.5, beta=0.25)
    algorithms = (
        Newton(),
        NewtonLineSearch(),
        KrylovNewton(max_dimension=3),
    )
    convergence_test = NormUnbalance(norm_type=2)
    analysis = TransientAnalysis(
        system=system,
        integrator=integrator,
        algorithms=algorithms,
        convergence_test=convergence_test,
        rtol=1e-6,
        atol=1e-8,
        max_iter=30,
    )

    # 9. Select nodal responses, element forces, and element deformations to save.
    recorder = Recorder(
        disp="all",
        velo=[3],
        accel=[3],
        ele_force="all",
        ele_defo="all",
        mode="both",
        diagnostics=True,
    )

    # 10. Run the forced response followed by zero-load free vibration, then read the results.
    result = analysis.run(
        dt=dt,
        free_vibration_duration=free_vibration_duration,
        recorder=recorder,
    )
    print("模态特征值:", eigenvalues)
    print("模态圆频率(rad/s):", omega)
    print("模态频率(Hz):", frequencies)
    print("模态周期(s):", periods)
    print("质量归一化振型（每列对应一阶模态）:\n", shapes)
    print("模态参与系数:", participation)
    print("有效模态质量:", effective_mass)
    print("有效模态质量比:", effective_mass_ratio)
    print("自动生成的单元:")
    for ele in system.eles:
        print(ele)
    print("末步节点位移:", result.disp[-1])
    print("末步单元变形:", result.ele_defo[-1])
    print("单元内力绝对峰值:", {
        ele_id: peak.absolute
        for ele_id, peak in result.peaks["ele_force"].items()
    })


if __name__ == "__main__":
    main()
