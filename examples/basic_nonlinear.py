import numpy as np

from mdof_solver.algorithms import Newton, NewtonLineSearch
from mdof_solver.analysis import TransientAnalysis
from mdof_solver.integrators import Newmark
from mdof_solver.loading import LoadHistory
from mdof_solver.materials import Steel01
from mdof_solver.matrices import DampingMatrix, StiffnessMatrix
from mdof_solver.model import System
from mdof_solver.recording import Recorder

steel = Steel01(tag=1, Fy=10.0, k=100.0, b=0.02)
stiffness = StiffnessMatrix([[2 * steel, -steel], [-steel, steel]])
dt = 0.01
sample_index = np.arange(201)
forces = np.zeros((len(sample_index), 2))
forces[:, 1] = 8.0 * np.sin(4.0 * sample_index * dt)
mass = np.eye(2)
load = LoadHistory(forces, dt=dt)

modal_system = System(mass, np.zeros((2, 2)), stiffness, load)
omega1, omega2 = modal_system.eigen(n_modes=2).circular_frequencies
damping_ratio = 0.05
rayleigh_mass = (
    2.0 * damping_ratio * omega1 * omega2 / (omega1 + omega2)
)
rayleigh_stiffness = 2.0 * damping_ratio / (omega1 + omega2)
damping = DampingMatrix(
    rayleigh_mass * mass
    + rayleigh_stiffness * modal_system.initial_stiffness()
)

system = System(
    mass=mass,
    damping=damping,
    stiffness=stiffness,
    load=load,
)
result = TransientAnalysis(
    system, Newmark(), (Newton(), NewtonLineSearch()),
).run(
    dt=dt,
    recorder=Recorder(
        disp="all", ele_force="all",
        ele_defo="all", mode="both", diagnostics=True,
    ),
)

print("自动生成的单元:")
for ele in system.eles:
    print(ele)
print("末步节点位移:", result.disp[-1])
print("单元内力绝对峰值:", {
    key: value.absolute for key, value in result.peaks["ele_force"].items()
})
