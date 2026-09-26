import numpy as np
import pytest

from mdof_solver import Elastic, LoadHistory, MinMax, Parallel, Steel01, SteelMPF, System


def make_parallel():
    return Parallel(
        3, Elastic(1, 100), Steel01(2, Fy=10, k=100, b=0.01),
        factors=(2.0, 0.5),
    )


def test_parallel_and_minmax_match_opensees_committed_path():
    material = MinMax(4, make_parallel(), min_strain=-0.2, max_strain=0.2)
    # OpenSees.exe testUniaxial: both bounds are inclusive; failure persists.
    path = [0, 0.05, 0.15, 0.2, 0.1, -0.2, 0]
    expected = [(0, 250), (12.5, 250), (35.025, 200.5)] + [(0, 2.5e-6)] * 4
    for strain, (stress, tangent) in zip(path, expected, strict=True):
        material.setTrialStrain(strain)
        assert material.getStress() == pytest.approx(stress)
        assert material.getTangent() == pytest.approx(tangent)
        material.commitState()
    assert material.Cfailed
    assert material.getStrain() == pytest.approx(0.15)


def test_minmax_trial_failure_can_be_reverted_before_commit():
    material = MinMax(2, Steel01(1, Fy=10, k=100, b=0.01), -0.1, 0.1)
    material.setTrialStrain(0.05)
    material.commitState()
    material.setTrialStrain(0.1)
    assert material.Tfailed
    assert material.getStress() == 0
    material.revertToLastCommit()
    assert not material.Tfailed
    assert material.getStress() == pytest.approx(5)
    material.setTrialStrain(0.06)
    material.commitState()
    assert not material.Cfailed
    material.setTrialStrain(-0.1)
    material.commitState()
    assert material.Cfailed
    material.revertToLastCommit()
    material.setTrialStrain(0.0)
    assert material.getStress() == 0
    assert material.getTangent() == pytest.approx(1e-6)
    material.revertToStart()
    assert not material.Cfailed
    assert material.getStress() == 0


def test_parallel_children_and_copies_have_private_history():
    source = Steel01(1, Fy=10, k=100, b=0.01)
    material = Parallel(2, source, source)
    material.setTrialStrain(0.2)
    assert material.getStress() == pytest.approx(20.2)
    assert material.getTangent() == pytest.approx(2)
    assert source.getStress() == 0
    material.commitState()
    clone = material.getCopy()
    clone.setTrialStrain(-0.2)
    assert clone.getStress() != material.getStress()
    assert clone.materials[0] is not material.materials[0]
    material.revertToStart()
    assert material.getInitialTangent() == pytest.approx(200)
    assert material.getStress() == 0


def test_nested_wrappers_assemble_into_system_with_steel_mpf():
    steel = SteelMPF(1, 10, 10, 100, 0.02, 0.02, 20, 0.925, 0.15)
    material = MinMax(4, Parallel(3, Elastic(2, 5), steel), -0.1, 0.1)
    system = System([[1]], [[0]], [[material]], LoadHistory([0, 1], [[0], [0]]))
    assert system.initial_stiffness()[0, 0] == pytest.approx(105)
    runtime = system.create_runtime()
    force, tangent, _ = runtime.evaluate(np.array([0.02]), np.array([0.0]))
    assert force[0] > 0
    assert tangent[0, 0] > 0
    force, tangent, _ = runtime.evaluate(np.array([0.1]), np.array([0.0]))
    assert force[0] == 0
    assert tangent[0, 0] == pytest.approx(1.05e-6)
    runtime.revert()
    force, _, _ = runtime.evaluate(np.array([0.02]), np.array([0.0]))
    assert force[0] > 0


def test_minmax_inside_parallel_only_removes_its_branch():
    material = Parallel(
        3, Elastic(1, 100),
        MinMax(2, Steel01(4, Fy=10, k=100, b=0.01), -0.1, 0.1),
    )
    material.setTrialStrain(0.1)
    assert material.getStress() == pytest.approx(10)
    assert material.getTangent() == pytest.approx(100.000001)
    material.commitState()
    material.setTrialStrain(0.05)
    assert material.getStress() == pytest.approx(5)


def test_wrappers_copy_existing_committed_material_state():
    source = Steel01(1, Fy=10, k=100, b=0.01)
    source.setStrain(0.05)
    limited = MinMax(2, source, -0.1, 0.1)
    assert limited.getStrain() == pytest.approx(0.05)
    assert limited.getStress() == pytest.approx(5)
    combined = Parallel(3, source, Elastic(4, 10))
    assert combined.getStress() == pytest.approx(5)
    source.setStrain(0.08)
    assert limited.getStress() == pytest.approx(5)
    assert combined.getStress() == pytest.approx(5)


@pytest.mark.parametrize("factory,error", [
    (lambda: Parallel(1), ValueError),
    (lambda: Parallel(1, Elastic(2, 10), factors=(1, 2)), ValueError),
    (lambda: MinMax(1, Elastic(2, 10), 0.1, 0.1), ValueError),
    (lambda: MinMax(1, object()), TypeError),
])
def test_wrapper_validation(factory, error):
    with pytest.raises(error):
        factory()
