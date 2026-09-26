import numpy as np
import pytest

from mdof_solver import LoadHistory, SteelMPF, System


def make_material(**overrides):
    parameters = dict(
        tag=1, sigyieldp=400, sigyieldn=350, E0=200000,
        bp=0.01, bn=0.015, R0=20, cR1=0.925, cR2=0.15,
    )
    parameters.update(overrides)
    return SteelMPF(**parameters)


def test_steel_mpf_stress_and_tangent_match_opensees_cyclic_path():
    material = make_material(a3=0.05, a4=1, a5=0.08, a6=1)
    # Values from OpenSees.exe's testUniaxial material test.
    path = [0, 0.0005, 0.002, 0.005, 0.001, -0.001, -0.003,
            0, 0.004, -0.006, 0.008]
    expected = [
        (0, 200000),
        (100, 200000),
        (386.510786254, 97627.6965636),
        (405.999999782, 2000.00087081),
        (-222.629119653, 76086.3242122),
        (-317.460690526, 27443.3926891),
        (-354.074612292, 12069.4678001),
        (165.183992171, 116421.515888),
        (378.402304375, 18811.8407374),
        (-374.083561077, 7023.37877418),
        (470.505710392, 5057.65291825),
    ]
    for strain, (stress, tangent) in zip(path, expected, strict=True):
        material.setTrialStrain(strain)
        assert material.getStress() == pytest.approx(stress, rel=1e-9, abs=1e-7)
        assert material.getTangent() == pytest.approx(tangent, rel=1e-9, abs=1e-5)
        material.commitState()


def test_steel_mpf_trial_revert_copy_and_local_tangent():
    material = make_material()
    material.setTrialStrain(0.005)
    material.commitState()
    material.setTrialStrain(0.001)
    first = (material.getStress(), material.getTangent())
    material.setTrialStrain(-0.001)
    material.setTrialStrain(0.001)
    assert (material.getStress(), material.getTangent()) == pytest.approx(first)
    h = 1e-8
    plus = material.getCopy()
    minus = material.getCopy()
    plus.setTrialStrain(0.001 + h)
    minus.setTrialStrain(0.001 - h)
    assert material.getTangent() == pytest.approx(
        (plus.getStress() - minus.getStress()) / (2 * h), rel=1e-6
    )
    material.revertToLastCommit()
    assert material.getStrain() == pytest.approx(0.005)
    assert material.getStress() == pytest.approx(405.999999782, abs=1e-6)
    material.revertToStart()
    assert material.getStress() == 0
    assert material.getTangent() == material.getInitialTangent() == 200000


def test_steel_mpf_assembles_into_system():
    material = make_material()
    system = System([[1]], [[0]], [[material]], LoadHistory([0, 1], [[0], [0]]))
    assert system.initial_stiffness()[0, 0] == pytest.approx(200000)
    runtime = system.create_runtime()
    force, tangent, _ = runtime.evaluate(np.array([0.005]), np.array([0.0]))
    assert force[0] == pytest.approx(405.999999782, abs=1e-6)
    assert tangent[0, 0] == pytest.approx(2000.00087081, abs=1e-5)


@pytest.mark.parametrize("invalid", [{"E0": 0}, {"bp": 1}, {"R0": 0}, {"cR2": 0}])
def test_steel_mpf_rejects_invalid_parameters(invalid):
    with pytest.raises(ValueError):
        make_material(**invalid)
