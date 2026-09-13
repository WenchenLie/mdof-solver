# mdof-solver

`mdof-solver` 是一个用于小中型非线性多自由度体系的 Python 时程分析程序，求解：

\[
M\ddot u+C\dot u+f_{\mathrm{int}}(u)=p(t)
\]

用户负责提供质量、阻尼、刚度和节点荷载时程；程序负责材料状态、时程积分、非线性迭代、自适应缩步、模态分析和选择性结果记录。

## 安装

项目使用 [uv](https://docs.astral.sh/uv/) 管理环境：

```bash
uv sync --group dev
uv run pytest
```

## 最小示例

```python
import numpy as np
from mdof_solver.algorithms import Newton
from mdof_solver.analysis import TransientAnalysis
from mdof_solver.convergence import NormUnbalance
from mdof_solver.integrators import Newmark
from mdof_solver.loading import LoadHistory
from mdof_solver.materials import Steel01
from mdof_solver.matrices import DampingMatrix, StiffnessMatrix
from mdof_solver.model import System
from mdof_solver.recording import Recorder

steel = Steel01(tag=1, Fy=10.0, k=100.0, b=0.02)
stiffness = StiffnessMatrix([[2 * steel, -steel], [-steel, steel]])
dt = 0.01
forces = np.zeros((201, 2))
forces[:, 1] = 5.0 * np.sin(4.0 * np.arange(201) * dt)
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
    system, Newmark(), (Newton(),),
    convergence_test=NormUnbalance(),
).run(
    dt=dt,
    recorder=Recorder(
        disp="all",
        ele_force="all",
        ele_defo="all",
        mode="both",
    ),
)
print(system.eles)
print(result.disp)
```

矩阵中的材料对象是规则模板。上例会自动生成两个独立材料状态：一条使用 `u[0]`，另一条使用 `u[1]-u[0]`。单元变形可以通过记录器按需输出。

完整接口、算法设置、输出格式和新材料开发方法见[中文用户指南](docs/用户指南.md)。可运行案例位于 [examples/basic_nonlinear.py](examples/basic_nonlinear.py)。
