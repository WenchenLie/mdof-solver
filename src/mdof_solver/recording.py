from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray


Selection = list[int] | tuple[int, ...] | Literal["all"] | None
EleSelection = list[str] | tuple[str, ...] | Literal["all"] | None


@dataclass(frozen=True)
class Recorder:
    disp: Selection = None
    velo: Selection = None
    accel: Selection = None
    ele_force: EleSelection = None
    ele_defo: EleSelection = None
    mode: Literal["history", "peak", "both"] = "history"
    diagnostics: bool = False
    csv_path: str | Path | None = None
    keep_in_memory: bool = True
    interval: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("history", "peak", "both"):
            raise ValueError("Recorder.mode 必须是 history、peak 或 both")
        if not self.keep_in_memory and self.csv_path is None:
            raise ValueError("keep_in_memory=False 时必须设置 csv_path")
        if self.interval is not None and self.interval <= 0:
            raise ValueError("Recorder.interval 必须大于 0")


@dataclass(frozen=True)
class Peak:
    minimum: float
    minimum_time: float
    maximum: float
    maximum_time: float
    absolute: float
    absolute_time: float


@dataclass
class AnalysisResult:
    times: NDArray[np.float64]
    responses: dict[str, NDArray[np.float64]]
    columns: dict[str, tuple[int | str, ...]]
    peaks: dict[str, dict[int | str, Peak]]
    diagnostics: list[dict[str, float | int | str]] = field(default_factory=list)
    completed: bool = True
    message: str = ""

    @property
    def disp(self) -> NDArray[np.float64]:
        return self.responses["disp"]

    @property
    def velo(self) -> NDArray[np.float64]:
        return self.responses["velo"]

    @property
    def accel(self) -> NDArray[np.float64]:
        return self.responses["accel"]

    @property
    def ele_force(self) -> NDArray[np.float64]:
        return self.responses["ele_force"]

    @property
    def ele_defo(self) -> NDArray[np.float64]:
        return self.responses["ele_defo"]

    def save_npz(self, path: str | Path) -> None:
        payload = {"times": self.times}
        payload.update({key: value for key, value in self.responses.items()})
        payload["metadata"] = np.array({
            "columns": self.columns,
            "peaks": self.peaks,
            "diagnostics": self.diagnostics,
            "completed": self.completed,
            "message": self.message,
        }, dtype=object)
        np.savez_compressed(path, **payload)


class ResponseCollector:
    def __init__(self, recorder: Recorder, ndof: int, ele_ids: tuple[str, ...]):
        self.config = recorder
        self.node_indices = {
            "disp": self._node_selection(recorder.disp, ndof),
            "velo": self._node_selection(recorder.velo, ndof),
            "accel": self._node_selection(recorder.accel, ndof),
        }
        self.ele_indices, self.ele_ids = self._ele_selection(recorder.ele_force, ele_ids)
        self.defo_indices, self.defo_ids = self._ele_selection(recorder.ele_defo, ele_ids)
        self.columns: dict[str, tuple[int | str, ...]] = {
            key: tuple(value) for key, value in self.node_indices.items() if value
        }
        if self.ele_indices:
            self.columns["ele_force"] = self.ele_ids
        if self.defo_indices:
            self.columns["ele_defo"] = self.defo_ids
        self.keep_history = recorder.mode in ("history", "both") and recorder.keep_in_memory
        self.keep_peak = recorder.mode in ("peak", "both")
        self.times: list[float] = []
        self.history: dict[str, list[NDArray[np.float64]]] = {key: [] for key in self.columns}
        self._peak_values: dict[str, dict[int | str, list[float]]] = {
            key: {column: [np.inf, np.nan, -np.inf, np.nan, 0.0, np.nan] for column in columns}
            for key, columns in self.columns.items()
        }
        self.diagnostics: list[dict[str, float | int | str]] = []
        self._file = None
        self._writer = None
        if recorder.csv_path is not None:
            path = Path(recorder.csv_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = path.open("w", newline="", encoding="utf-8")
            names = ["time"] + [f"{key}[{column}]" for key, columns in self.columns.items() for column in columns]
            self._writer = csv.writer(self._file)
            self._writer.writerow(names)

    @staticmethod
    def _node_selection(selection: Selection, size: int) -> list[int]:
        if selection is None:
            return []
        result = list(range(size)) if selection == "all" else [int(x) for x in selection]
        if len(set(result)) != len(result) or any(x < 0 or x >= size for x in result):
            raise ValueError("节点记录索引无效或重复")
        return result

    @staticmethod
    def _ele_selection(selection: EleSelection, ids: tuple[str, ...]) -> tuple[list[int], tuple[str, ...]]:
        if selection is None:
            return [], ()
        chosen = ids if selection == "all" else tuple(selection)
        if len(set(chosen)) != len(chosen) or any(x not in ids for x in chosen):
            raise ValueError("单元记录编号无效或重复")
        return [ids.index(x) for x in chosen], chosen

    def record(
        self, time: float, u, v, a, ele_force, ele_defo,
        *, store_history: bool = True,
    ) -> None:
        values = {
            "disp": np.asarray(u)[self.node_indices["disp"]],
            "velo": np.asarray(v)[self.node_indices["velo"]],
            "accel": np.asarray(a)[self.node_indices["accel"]],
            "ele_force": np.asarray(ele_force)[self.ele_indices],
            "ele_defo": np.asarray(ele_defo)[self.defo_indices],
        }
        values = {key: values[key] for key in self.columns}
        if self.keep_history and store_history:
            self.times.append(float(time))
            for key, value in values.items():
                self.history[key].append(value.copy())
        if self.keep_peak:
            for key, vector in values.items():
                for column, value in zip(self.columns[key], vector, strict=True):
                    peak = self._peak_values[key][column]
                    scalar = float(value)
                    if scalar < peak[0]: peak[0], peak[1] = scalar, float(time)
                    if scalar > peak[2]: peak[2], peak[3] = scalar, float(time)
                    if abs(scalar) > abs(peak[4]) or np.isnan(peak[5]): peak[4], peak[5] = scalar, float(time)
        if self._writer is not None and store_history:
            self._writer.writerow([float(time)] + [float(x) for key in self.columns for x in values[key]])
            self._file.flush()

    def diagnostic(self, **values) -> None:
        if self.config.diagnostics:
            self.diagnostics.append(values)

    def finish(self, completed: bool = True, message: str = "") -> AnalysisResult:
        if self._file is not None:
            self._file.close()
        responses = {}
        if self.keep_history:
            responses = {
                key: np.vstack(rows) if rows else np.empty((0, len(self.columns[key])))
                for key, rows in self.history.items()
            }
            times = np.asarray(self.times)
        else:
            times = np.empty(0)
        peaks = {
            key: {
                column: Peak(values[0], values[1], values[2], values[3], values[4], values[5])
                for column, values in columns.items()
            }
            for key, columns in self._peak_values.items()
        } if self.keep_peak else {}
        return AnalysisResult(times, responses, self.columns, peaks, self.diagnostics, completed, message)
