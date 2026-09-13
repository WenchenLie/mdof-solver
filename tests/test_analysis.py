import numpy as np
import pytest

from mdof_solver import (
    CentralDifference,
    Elastic,
    HHT,
    KrylovNewton,
    LoadHistory,
    Newmark,
    Newton,
    NewtonLineSearch,
    Recorder,
    System,
    TransientAnalysis,
)
from mdof_solver.algorithms import SolveOutcome, SolutionAlgorithm


def oscillator(load_values=None):
    values = [[0.0], [0.0]] if load_values is None else load_values
    return System([[1.0]], [[0.0]], [[Elastic(1, 4.0)]], LoadHistory([0.0, 1.0], values))


def test_newmark_matches_undamped_free_vibration():
    system = oscillator()
    result = TransientAnalysis(system, Newmark(), (Newton(),), rtol=1e-10, atol=1e-12).run(
        0.002, 1.0, u0=[1.0], recorder=Recorder(disp="all", velo="all", accel="all")
    )
    assert result.responses["disp"][-1, 0] == pytest.approx(np.cos(2.0), abs=2e-5)
    assert result.responses["velo"][-1, 0] == pytest.approx(-2 * np.sin(2.0), abs=3e-5)


def test_hht_alpha_one_equals_newmark():
    system = oscillator([[0.0], [1.0]])
    recorder = Recorder(disp="all", velo="all", accel="all", ele_force="all")
    newmark = TransientAnalysis(system, Newmark(), (Newton(),)).run(0.01, 1.0, recorder=recorder)
    hht = TransientAnalysis(system, HHT(alpha=1.0), (Newton(),)).run(0.01, 1.0, recorder=recorder)
    for key in newmark.responses:
        assert np.allclose(newmark.responses[key], hht.responses[key], atol=1e-11, rtol=1e-10)


def test_central_difference_records_complete_endpoint_state():
    system = oscillator()
    result = TransientAnalysis(system, CentralDifference()).run(
        0.01, 0.1, u0=[1.0], v0=[0.25],
        recorder=Recorder(
            disp="all", velo="all", accel="all",
            ele_force="all", ele_defo="all",
        ),
    )
    assert result.times[0] == 0.0
    assert result.times[-1] == pytest.approx(0.1)
    assert all(values.shape == (11, 1) for values in result.responses.values())
    assert np.all(np.isfinite(np.concatenate(list(result.responses.values()))))


def test_element_deformation_output_uses_internal_mapping():
    material = Elastic(2, 10.0)
    system = System(
        np.eye(2), np.zeros((2, 2)),
        [[2 * material, -material], [-material, material]],
        LoadHistory(np.zeros((2, 2)), dt=0.1),
    )
    result = TransientAnalysis(system, Newmark(), (Newton(),)).run(
        0.1, u0=[0.2, 0.25],
        recorder=Recorder(ele_defo="all", ele_force="all"),
    )
    assert np.allclose(result.responses["ele_defo"][0], [0.2, 0.05])
    assert np.allclose(result.responses["ele_force"][0], [2.0, 0.5])


def test_peaks_and_file_outputs(tmp_path):
    result = TransientAnalysis(oscillator(), Newmark(), (Newton(),)).run(
        0.01, 0.1, u0=[1.0],
        recorder=Recorder(
            disp="all",
            ele_force="all",
            ele_defo="all",
            mode="both",
            csv_path=tmp_path / "response.csv",
        ),
    )
    peak = result.peaks["disp"][0]
    assert peak.maximum == pytest.approx(1.0)
    assert (tmp_path / "response.csv").read_text().startswith(
        "time,disp[0],ele_force[E1],ele_defo[E1]"
    )
    result.save_npz(tmp_path / "response.npz")
    with np.load(tmp_path / "response.npz", allow_pickle=True) as data:
        assert np.allclose(data["times"], result.times)


def test_central_difference_rejects_off_grid_end():
    with pytest.raises(ValueError, match="fixed time-step grid"):
        TransientAnalysis(oscillator(), CentralDifference()).run(0.03, 0.1)


def test_free_vibration_extends_analysis_with_zero_load():
    system = System(
        [[1.0]], [[0.0]], [[0.0]],
        LoadHistory([[1.0], [1.0]], dt=0.1),
    )
    result = TransientAnalysis(system, Newmark(), (Newton(),)).run(
        0.1,
        free_vibration_duration=0.1,
        recorder=Recorder(disp="all", velo="all", accel="all"),
    )
    assert np.allclose(result.times, [0.0, 0.1, 0.2])
    assert result.accel[:, 0] == pytest.approx([1.0, 1.0, 0.0])
    assert result.disp[-1, 0] == pytest.approx(0.0175)


def test_central_difference_supports_on_grid_free_vibration():
    result = TransientAnalysis(oscillator(), CentralDifference()).run(
        0.01,
        t_end=0.1,
        free_vibration_duration=0.1,
        u0=[1.0],
        recorder=Recorder(disp="all"),
    )
    assert result.times[-1] == pytest.approx(0.2)
    assert result.disp.shape == (21, 1)


def test_implicit_analysis_splits_step_at_load_end():
    result = TransientAnalysis(oscillator(), Newmark(), (Newton(),)).run(
        0.06,
        t_end=0.1,
        free_vibration_duration=0.05,
        u0=[1.0],
        recorder=Recorder(disp="all"),
    )
    assert result.times == pytest.approx([0.0, 0.06, 0.1, 0.12, 0.15])


@pytest.mark.parametrize("duration", [-0.1, np.inf, np.nan])
def test_free_vibration_duration_must_be_finite_and_nonnegative(duration):
    with pytest.raises(ValueError, match="free_vibration_duration"):
        TransientAnalysis(oscillator()).run(0.01, free_vibration_duration=duration)


def test_fixed_record_interval_still_tracks_substep_peaks():
    result = TransientAnalysis(oscillator(), Newmark(), (Newton(),)).run(
        0.01, 0.1, u0=[1.0],
        recorder=Recorder(disp="all", mode="both", interval=0.05),
    )
    assert np.allclose(result.times, [0.0, 0.05, 0.1])
    assert result.peaks["disp"][0].maximum == pytest.approx(1.0)


@pytest.mark.parametrize("algorithm", [Newton(), NewtonLineSearch(), KrylovNewton()])
def test_all_nonlinear_algorithms_solve_same_steel_problem(algorithm):
    from mdof_solver import Steel01

    material = Steel01(2, Fy=1.0, k=10.0, b=0.02)
    system = System([[1.0]], [[0.02]], [[material]], LoadHistory([0, 1], [[0], [5]]))
    result = TransientAnalysis(system, Newmark(), (algorithm,), rtol=1e-7, atol=1e-9).run(
        0.01, 1.0, recorder=Recorder(disp="all", ele_force="all")
    )
    assert result.responses["disp"][-1, 0] == pytest.approx(0.63041377, rel=2e-6)


class AlwaysFail(SolutionAlgorithm):
    name = "AlwaysFail"

    def solve(self, problem, initial, rtol, atol, max_iter):
        return SolveOutcome(False, initial, 0, 1.0, "injected failure")


def test_algorithm_switches_after_minimum_step_failure():
    result = TransientAnalysis(
        oscillator(), Newmark(), (AlwaysFail(), Newton()), min_factor=1.0,
    ).run(0.01, 0.02, recorder=Recorder(disp="all", diagnostics=True))
    assert [row["status"] for row in result.diagnostics[:2]] == ["rejected", "accepted"]
    assert result.diagnostics[1]["algorithm"] == "Newton"
