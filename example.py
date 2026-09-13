import matplotlib.pyplot as plt
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
    ground_acceleration = np.loadtxt('data/gm1(dt=0.01).txt') * 9800
    free_vibration_duration = 30

    # 2. Build the four-DOF mass matrix.
    masses = np.array([2.0, 1.8, 1.5, 1.2])
    mass = MassMatrix(np.diag(masses))

    # 3. Convert ground acceleration into nodal inertia loads using D'Alembert's principle.
    influence = np.ones(4)
    force_pattern = mass.values @ influence
    load_samples = -ground_acceleration[:, None] * force_pattern[None, :]
    load = LoadHistory(load_samples, dt=dt)

    # 4. Define material-rule templates and an ordinary numeric stiffness.
    k1 = Steel01(tag=1, Fy=100, k=320.0, b=0.02)
    k2 = Steel01(tag=2, Fy=80, k=280.0, b=0.015)
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
    print("Eigenvalues:", eigenvalues)
    print("Circular frequencies (rad/s):", omega)
    print("Frequencies (Hz):", frequencies)
    print("Periods (s):", periods)
    print("Mass-normalized mode shapes (one mode per column):\n", shapes)
    print("Modal participation factors:", participation)
    print("Effective modal masses:", effective_mass)
    print("Effective modal mass ratios:", effective_mass_ratio)
    print("Automatically generated elements:")
    for ele in system.eles:
        print(ele)
    print("Final nodal displacements:", result.disp[-1])
    print("Final element deformations:", result.ele_defo[-1])
    print("Absolute peak element forces:", {
        ele_id: peak.absolute
        for ele_id, peak in result.peaks["ele_force"].items()
    })

    # 11. Plot response histories and the hysteresis loop from the recorded result interfaces.
    dof_4 = 3
    connecting_element = next((ele for ele in system.eles if ele.dofs == (0, 1)), None)
    if connecting_element is None:
        raise ValueError("No element connects DOF 1 and DOF 2")
    element_id = connecting_element.id
    disp_column = result.columns["disp"].index(dof_4)
    accel_column = result.columns["accel"].index(dof_4)
    force_column = result.columns["ele_force"].index(element_id)
    deformation_column = result.columns["ele_defo"].index(element_id)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(result.times, result.disp[:, disp_column], linewidth=1.0)
    axes[0, 0].set_title("DOF 4 displacement history")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Displacement")

    axes[0, 1].plot(result.times, result.accel[:, accel_column], linewidth=1.0)
    axes[0, 1].set_title("DOF 4 acceleration history")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Acceleration")

    axes[1, 0].plot(result.times, result.ele_force[:, force_column], linewidth=1.0)
    axes[1, 0].set_title(f"Element 1 shear history (solver ID {element_id})")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Shear force")

    axes[1, 1].plot(
        result.ele_defo[:, deformation_column],
        result.ele_force[:, force_column],
        linewidth=1.0,
    )
    axes[1, 1].set_title(f"Element 1 hysteresis loop (solver ID {element_id})")
    axes[1, 1].set_xlabel("Relative displacement (DOF 2 - DOF 1)")
    axes[1, 1].set_ylabel("Shear force")

    for axis in axes.flat:
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    main()
