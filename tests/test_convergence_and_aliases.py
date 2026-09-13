import numpy as np
import pytest

from mdof_solver import (
    Elastic,
    EnergyIncr,
    LoadHistory,
    Newmark,
    Newton,
    NormDispIncr,
    NormUnbalance,
    Recorder,
    System,
    TransientAnalysis,
)


def oscillator():
    return System(
        [[1.0]], [[0.02]], [[Elastic(1, 4.0)]],
        LoadHistory(np.zeros((3, 1)), dt=0.05),
    )


def test_convergence_metrics_match_opensees_definitions():
    residual = np.array([3.0, 4.0])
    increment = np.array([6.0, 8.0])
    solution = np.array([1.0, 2.0])

    unbalance = NormUnbalance().evaluate(residual, increment, solution, 5.0, 0.0, 1e9)
    disp = NormDispIncr().evaluate(residual, increment, solution, 5.0, 0.0, 1e9)
    energy = EnergyIncr().evaluate(residual, increment, solution, 5.0, 0.0, 1e9)

    assert unbalance.value == pytest.approx(5.0)
    assert disp.value == pytest.approx(10.0)
    assert energy.value == pytest.approx(25.0)


@pytest.mark.parametrize("convergence_test", [NormUnbalance(), NormDispIncr(), EnergyIncr()])
def test_all_convergence_tests_run_in_transient_analysis(convergence_test):
    result = TransientAnalysis(
        oscillator(), Newmark(), (Newton(),),
        rtol=0.0, atol=1e-10, max_iter=10,
        convergence_test=convergence_test,
    ).run(
        0.05, u0=[1.0],
        recorder=Recorder(disp="all", diagnostics=True),
    )
    assert result.completed
    assert result.diagnostics[-1]["test"] == convergence_test.name
    assert result.diagnostics[-1]["test_value"] <= result.diagnostics[-1]["test_limit"]


def test_recorder_and_result_use_concise_names():
    result = TransientAnalysis(oscillator(), Newmark(), (Newton(),)).run(
        0.05,
        recorder=Recorder(
            disp="all", velo="all", accel="all",
            ele_force="all", ele_defo="all", mode="both",
        ),
    )
    assert set(result.responses) == {"disp", "velo", "accel", "ele_force", "ele_defo"}
    assert result.disp is result.responses["disp"]
    assert result.ele_defo is result.responses["ele_defo"]
    assert result.columns["ele_force"] == ("E1",)
    assert result.peaks["ele_defo"]["E1"].absolute == pytest.approx(0.0)
    assert oscillator().eles[0].id == "E1"
    with pytest.raises(KeyError):
        _ = result.responses["displacement"]
    assert not hasattr(oscillator(), "elements")


def test_full_recorder_parameter_names_are_removed():
    with pytest.raises(TypeError, match="unexpected keyword"):
        Recorder(displacements="all")


def test_peak_is_the_only_peak_only_mode_name():
    assert Recorder(mode="peak").mode == "peak"
    with pytest.raises(ValueError, match="history.*peak.*both"):
        Recorder(mode="peaks")

    result = TransientAnalysis(oscillator(), Newmark(), (Newton(),)).run(
        0.05, u0=[1.0], recorder=Recorder(disp="all", mode="peak"),
    )
    assert result.times.size == 0
    assert result.responses == {}
    assert result.peaks["disp"][0].absolute == pytest.approx(1.0)
