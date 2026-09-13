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
    # 1. 定义计算步长和等步长地面加速度样本。
    dt = 0.01
    duration = 5.0
    sample_count = round(duration / dt) + 1
    sample_index = np.arange(sample_count)
    time = sample_index * dt  # 只用于生成示例波形，不传给 LoadHistory。
    ground_acceleration = 4.0 * np.sin(2.0 * np.pi * 1.2 * time) * np.exp(-0.35 * time)

    # 2. 构造四自由度质量矩阵。
    masses = np.array([2.0, 1.8, 1.5, 1.2])
    mass = MassMatrix(np.diag(masses))

    # 3. 用户按达朗贝尔原理自行把地面加速度换成节点惯性荷载。
    influence = np.ones(4)
    force_pattern = mass.values @ influence
    load_samples = -ground_acceleration[:, None] * force_pattern[None, :]
    load = LoadHistory(load_samples, dt=dt)

    # 4. 定义材料规则模板和普通数值刚度。
    k1 = Steel01(tag=1, Fy=6.0, k=320.0, b=0.02)
    k2 = Steel01(tag=2, Fy=5.0, k=280.0, b=0.015)
    k3 = Elastic(tag=3, k=240.0)
    k4 = 180.0

    # 5. 直接填写含材料模板的四自由度刚度矩阵。
    stiffness = StiffnessMatrix([
        [k1 + k2, -k2, 0.0, 0.0],
        [-k2, k2 + k3, -k3, 0.0],
        [0.0, -k3, k3 + k4, -k4],
        [0.0, 0.0, -k4, k4],
    ])

    # 6. 先做一次模态分析，再基于前两阶模态构造 Rayleigh 阻尼矩阵。
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

    # 7. 构建最终动力体系。
    system = System(
        mass=mass,
        damping=damping,
        stiffness=stiffness,
        load=load,
    )

    # 8. 分别定义积分器、候选算法、收敛检验和分析器。
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

    # 9. 选择要保存的节点响应、单元内力和单元变形。
    recorder = Recorder(
        disp="all",
        velo=[3],
        accel=[3],
        ele_force="all",
        ele_defo="all",
        mode="both",
        diagnostics=True,
    )

    # 10. 执行分析并读取结果。
    result = analysis.run(
        dt=dt,
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
