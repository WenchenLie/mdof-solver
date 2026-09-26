import numpy as np
import pytest

from mdof_solver import Elastic, LoadHistory, ModTakeda, Steel01, TSSCB, StiffnessMatrix, System, UniaxialMaterial


def zero_load(ndof=2):
    return LoadHistory(np.zeros((2, ndof)), dt=1.0)


def test_material_protocol_copy_revert_and_initial_tangent():
    material = Steel01(1, Fy=10.0, k=100.0, b=0.02)
    assert material.getInitialTangent() == 100.0
    assert material.getTangent() == 100.0
    material.setTrialStrain(0.2)
    first_trial = material.getStress()
    assert first_trial == pytest.approx(10.2)
    assert material.getTangent() == pytest.approx(2.0)
    epsilon = 1e-7
    material.setTrialStrain(0.2 + epsilon)
    stress_plus = material.getStress()
    material.setTrialStrain(0.2 - epsilon)
    stress_minus = material.getStress()
    assert (stress_plus - stress_minus) / (2 * epsilon) == pytest.approx(2.0, rel=1e-7)
    material.setTrialStrain(0.2)
    assert material.getStress() == first_trial
    material.revertToLastCommit()
    assert material.getStress() == 0.0
    material.setTrialStrain(0.2)
    material.commitState()
    copied = material.getCopy()
    copied.setTrialStrain(-0.2)
    assert copied.getStress() != material.getStress()
    copied.revertToStart()
    assert copied.getStress() == 0.0


def test_mod_takeda_trial_does_not_survive_revert():
    material = ModTakeda(2, Fy=10, k0=100, r=0.02, alpha=0.5, beta=0.2)
    material.setTrialStrain(0.2)
    material.revertToLastCommit()
    material.setTrialStrain(0.05)
    material.commitState()
    clean = ModTakeda(3, Fy=10, k0=100, r=0.02, alpha=0.5, beta=0.2)
    clean.setTrialStrain(0.05)
    clean.commitState()
    assert material.getStress() == pytest.approx(clean.getStress())


def test_mod_takeda_tangent_is_local_slope_after_crossing_yield():
    material = ModTakeda(2, Fy=10, k0=100, r=0.02, alpha=0.5, beta=0.2)
    material.setTrialStrain(0.2)
    assert material.getStress() == pytest.approx(10.2)
    assert material.getTangent() == pytest.approx(2.0, rel=1e-6)
    assert (material.getStress() - material.Cstress) / (material.getStrain() - material.Cstrain) == pytest.approx(51.0)


@pytest.mark.parametrize("config_type,fractured_force", [(1, 2.5), (2, 5.0)])
def test_tsscb_cyclic_stress_matches_reference_rule(config_type, fractured_force):
    material = TSSCB(
        7, F1=5, k0=100, ugap=0.2, F2=20, k1=80, k2=8, beta=0.4,
        hardening="-hardening", uh=0.65, r1=0.05, r2=0.01, r3=0.3,
        minmax="-minmax", uf=0.95, configType=config_type, up=0.15,
    )
    assert material.getInitialTangent() == 100
    path = [0.03, 0.12, 0.24, 0.4, 0.75, 0.4, -0.1, -0.35, -0.8, -1.0, -0.7, 0.5]
    expected = [3, 5, 8.2, 20.1, 22.6258024691358, 11.65987654320988,
                -5, -15.094444444444441, -21.52111111111111,
                -fractured_force, 0 if config_type == 1 else 5, fractured_force]
    for strain, stress in zip(path, expected, strict=True):
        material.setTrialStrain(strain)
        assert material.getStress() == pytest.approx(stress)
        material.commitState()


def test_tsscb_tangent_and_trial_state_are_consistent():
    material = TSSCB(8, F1=5, k0=100, ugap=0.2, F2=20, k1=80, k2=8,
                     beta=0.4, uf=0.9, up=0.2)
    material.setTrialStrain(0.4)
    assert material.getTangent() == pytest.approx(8, rel=1e-6)
    material.commitState()
    material.setTrialStrain(0.8)
    material.commitState()
    material.setTrialStrain(0.91)
    first = (material.getStress(), material.getTangent(), material.Trp)
    material.setTrialStrain(0.91)
    assert (material.getStress(), material.getTangent(), material.Trp) == pytest.approx(first)
    material.revertToLastCommit()
    assert material.Trp == 0
    assert not material.Tfracturing
    copied = material.getCopy()
    copied.setTrialStrain(1.1)
    assert copied.getStress() != material.getStress()
    material.revertToStart()
    assert material.getStress() == 0
    assert material.getTangent() == 100


def test_tsscb_zero_gap_initial_tangent_and_matrix_assembly():
    material = TSSCB(9, F1=5, k0=100, ugap=0, F2=20, k1=80, k2=8, beta=0.4)
    assert material.getInitialTangent() == 80
    system = System([[1]], [[0]], [[material]], LoadHistory([0, 1], [[0], [0]]))
    assert system.initial_stiffness()[0, 0] == pytest.approx(80)
    runtime = system.create_runtime()
    force, tangent, _ = runtime.evaluate(np.array([0.4]), np.array([0.0]))
    assert force[0] == pytest.approx(21.2)
    assert tangent[0, 0] == pytest.approx(8, rel=1e-6)


@pytest.mark.parametrize("invalid", [{"k1": 0}, {"uh": 0.2}, {"configType": 3}])
def test_tsscb_rejects_invalid_parameters(invalid):
    params = dict(tag=9, F1=5, k0=100, ugap=0.2, F2=20, k1=80,
                  k2=8, beta=0.4, uh=0.65)
    params.update(invalid)
    with pytest.raises(ValueError):
        TSSCB(**params)


def test_matrix_infers_two_independent_runtime_connections():
    material = Steel01(1, Fy=10, k=100, b=0.02)
    stiffness = StiffnessMatrix([[2 * material, -material], [-material, material]])
    system = System(np.eye(2), np.zeros((2, 2)), stiffness, zero_load())
    assert [(e.id, e.dofs, e.weight) for e in system.eles] == [
        ("E1", (0,), 1.0),
        ("E2", (0, 1), 1.0),
    ]
    runtime = system.create_runtime()
    assert runtime.eles[0].material is not runtime.eles[1].material
    _, _, forces = runtime.evaluate(np.array([0.2, 0.25]), np.zeros(2))
    assert forces[0] != forces[1]
    assert np.allclose(system.initial_stiffness(), [[200, -100], [-100, 100]])


def test_invalid_material_coefficient_matrix_is_rejected():
    material = Elastic(9, 10)
    with pytest.raises(ValueError, match="Off-diagonal coefficients"):
        System(np.eye(2), np.zeros((2, 2)), [[material, material], [material, material]], zero_load())


def test_modal_analysis_uses_public_initial_tangent():
    material = Elastic(4, 4.0)
    system = System([[2.0]], [[0.0]], [[material]], LoadHistory([0, 1], [[0], [0]]))
    modal = system.eigen(influence=[1.0])
    assert modal.eigenvalues[0] == pytest.approx(2.0)
    assert modal.circular_frequencies[0] == pytest.approx(np.sqrt(2.0))
    assert modal.effective_modal_mass_ratios[0] == pytest.approx(1.0)
    assert system.central_difference_limit() == pytest.approx(np.sqrt(2.0))


def test_modal_file_outputs(tmp_path):
    system = System([[1.0]], [[0.0]], [[4.0]], LoadHistory([0, 1], [[0], [0]]))
    modal = system.eigen(influence=[1.0])
    modal.save_csv(tmp_path / "modes.csv")
    modal.save_npz(tmp_path / "modes.npz")
    header = (tmp_path / "modes.csv").read_text().splitlines()[0]
    assert "mode_shape_dof_0" in header
    assert header.endswith("classification")
    with np.load(tmp_path / "modes.npz") as data:
        assert data["mode_shapes"].shape == (1, 1)


def test_uniform_load_history_needs_no_time_array():
    load = LoadHistory([[0.0, 0.0], [2.0, 4.0], [4.0, 8.0]], dt=0.1, t0=1.0)
    assert load.t_min == 1.0
    assert load.t_max == pytest.approx(1.2)
    assert np.allclose(load(1.05), [1.0, 2.0])


class OpaqueMaterial(UniaxialMaterial):
    """Uses unrelated internal names to enforce the public protocol."""

    def __init__(self, tag, stiffness):
        self.tag = tag
        self._secret_modulus = stiffness
        self.revertToStart()

    def setTrialStrain(self, strain, strainRate=0):
        self._trial_coordinate = strain
        self._trial_action = self._secret_modulus * strain

    def getStrain(self):
        return self._trial_coordinate

    def getStress(self):
        return self._trial_action

    def getTangent(self):
        return self._secret_modulus

    def getInitialTangent(self):
        return self._secret_modulus

    def commitState(self):
        self._saved_coordinate = self._trial_coordinate
        self._saved_action = self._trial_action

    def revertToLastCommit(self):
        self._trial_coordinate = self._saved_coordinate
        self._trial_action = self._saved_action

    def revertToStart(self):
        self._saved_coordinate = self._trial_coordinate = 0.0
        self._saved_action = self._trial_action = 0.0

    def getCopy(self):
        result = OpaqueMaterial(self.tag, self._secret_modulus)
        result._saved_coordinate = self._saved_coordinate
        result._trial_coordinate = self._trial_coordinate
        result._saved_action = self._saved_action
        result._trial_action = self._trial_action
        return result


def test_solver_uses_only_public_material_protocol():
    material = OpaqueMaterial(8, 7.5)
    system = System([[2.0]], [[0.0]], [[material]], LoadHistory([0, 1], [[0], [0]]))
    assert system.initial_stiffness()[0, 0] == pytest.approx(7.5)
    runtime = system.create_runtime()
    force, tangent, ele_force = runtime.evaluate(np.array([0.2]), np.array([0.0]))
    assert force[0] == pytest.approx(1.5)
    assert tangent[0, 0] == pytest.approx(7.5)
    assert ele_force[0] == pytest.approx(1.5)


@pytest.mark.parametrize("factory", [
    lambda: Elastic("invalid", 1.0),
    lambda: Steel01("invalid", 1.0, 10.0, 0.01),
    lambda: ModTakeda("invalid", 1.0, 10.0, 0.01, 0.5, 0.2),
])
def test_material_tag_must_be_integer(factory):
    with pytest.raises(TypeError, match="tag must be an integer"):
        factory()


def test_material_tags_are_unique_within_system():
    first = Elastic(1, 10.0)
    second = Elastic(1, 20.0)
    with pytest.raises(ValueError, match="tag 1.*duplicated"):
        System(np.eye(2), np.zeros((2, 2)), [[first, 0], [0, second]], zero_load())
