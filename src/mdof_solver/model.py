from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import eigh

from .loading import LoadHistory
from .materials import Elastic, UniaxialMaterial
from .matrices import DampingMatrix, MassMatrix, StiffnessMatrix


_MATERIAL_METHODS = (
    "setTrialStrain", "getStrain", "getStress", "getTangent",
    "getInitialTangent", "commitState", "revertToLastCommit",
    "revertToStart", "getCopy",
)


@dataclass(frozen=True)
class ElementInfo:
    id: str
    template_tag: int
    weight: float
    dofs: tuple[int, ...]
    deformation_coefficients: tuple[float, ...]


@dataclass
class RuntimeElement:
    info: ElementInfo
    b: NDArray[np.float64]
    material: UniaxialMaterial

    def trial(self, u: NDArray[np.float64], v: NDArray[np.float64]) -> tuple[float, float]:
        self.material.revertToLastCommit()
        self.material.setTrialStrain(float(self.b @ u), float(self.b @ v))
        force = self.info.weight * float(self.material.getStress())
        tangent = self.info.weight * float(self.material.getTangent())
        if not np.isfinite(force) or not np.isfinite(tangent):
            raise FloatingPointError(f"Material for element {self.info.id} returned a non-finite value")
        return force, tangent


class RuntimeModel:
    def __init__(self, numeric_stiffness: NDArray[np.float64], eles: list[RuntimeElement]):
        self.numeric_stiffness = numeric_stiffness
        self.eles = eles

    def evaluate(
        self, u: NDArray[np.float64], v: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        restoring = self.numeric_stiffness @ u
        tangent = self.numeric_stiffness.copy()
        forces = np.empty(len(self.eles), dtype=float)
        for index, ele in enumerate(self.eles):
            force, kt = ele.trial(u, v)
            forces[index] = force
            restoring += ele.b * force
            tangent += kt * np.outer(ele.b, ele.b)
        return restoring, tangent, forces

    def commit(self) -> None:
        checkpoints = [ele.material.getCopy() for ele in self.eles]
        try:
            for ele in self.eles:
                ele.material.commitState()
        except Exception:
            for ele, checkpoint in zip(self.eles, checkpoints, strict=True):
                ele.material = checkpoint
            raise

    def revert(self) -> None:
        for ele in self.eles:
            ele.material.revertToLastCommit()

    def deformations(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray([ele.b @ u for ele in self.eles], dtype=float)


@dataclass(frozen=True)
class ModalResult:
    eigenvalues: NDArray[np.float64]
    circular_frequencies: NDArray[np.float64]
    frequencies: NDArray[np.float64]
    periods: NDArray[np.float64]
    mode_shapes: NDArray[np.float64]
    classification: tuple[str, ...]
    participation_factors: NDArray[np.float64] | None = None
    effective_modal_masses: NDArray[np.float64] | None = None
    effective_modal_mass_ratios: NDArray[np.float64] | None = None

    def save_npz(self, path: str | Path) -> None:
        payload = {
            "eigenvalues": self.eigenvalues,
            "circular_frequencies": self.circular_frequencies,
            "frequencies": self.frequencies,
            "periods": self.periods,
            "mode_shapes": self.mode_shapes,
            "classification": np.asarray(self.classification),
        }
        for name in ("participation_factors", "effective_modal_masses", "effective_modal_mass_ratios"):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        np.savez_compressed(path, **payload)

    def save_csv(self, path: str | Path) -> None:
        columns = [self.eigenvalues, self.circular_frequencies, self.frequencies, self.periods]
        header = ["eigenvalue", "circular_frequency", "frequency", "period"]
        for dof, values in enumerate(self.mode_shapes):
            header.append(f"mode_shape_dof_{dof}")
            columns.append(values)
        for name, value in (
            ("participation_factor", self.participation_factors),
            ("effective_modal_mass", self.effective_modal_masses),
            ("effective_modal_mass_ratio", self.effective_modal_mass_ratios),
        ):
            if value is not None:
                header.append(name)
                columns.append(value)
        numeric = np.column_stack(columns)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="") as stream:
            stream.write(",".join(header + ["classification"]) + "\n")
            for row, classification in zip(numeric, self.classification, strict=True):
                stream.write(",".join([*(f"{value:.17g}" for value in row), classification]) + "\n")


class System:
    def __init__(
        self,
        mass: MassMatrix | ArrayLike,
        damping: DampingMatrix | ArrayLike,
        stiffness: StiffnessMatrix | ArrayLike,
        load: LoadHistory,
    ):
        self.mass = mass if isinstance(mass, MassMatrix) else MassMatrix(mass)
        self.damping = damping if isinstance(damping, DampingMatrix) else DampingMatrix(damping)
        self.stiffness = stiffness if isinstance(stiffness, StiffnessMatrix) else StiffnessMatrix(stiffness)
        self.load = load
        n = self.mass.shape[0]
        if self.damping.shape != (n, n) or self.stiffness.shape != (n, n) or load.size != n:
            raise ValueError("Mass, damping, stiffness, and load must have the same number of DOFs")
        self._numeric_stiffness, definitions = self.stiffness.split()
        if not np.allclose(self._numeric_stiffness, self._numeric_stiffness.T, rtol=1e-10, atol=1e-12):
            raise ValueError("The numeric part of the stiffness matrix must be symmetric")
        self._definitions = self._compile_definitions(definitions)
        self.eles = tuple(info for info, _, _ in self._definitions)

    @property
    def ndof(self) -> int:
        return self.mass.shape[0]

    def _validate_material(self, material: Any) -> None:
        missing = [name for name in _MATERIAL_METHODS if not callable(getattr(material, name, None))]
        if missing:
            raise TypeError(
                f"Material {getattr(material, 'tag', material)!r} is missing methods: "
                f"{', '.join(missing)}"
            )
        tag = getattr(material, "tag", None)
        if isinstance(tag, bool) or not isinstance(tag, Integral):
            raise TypeError("Material tag must be an integer")

    def _compile_definitions(self, definitions):
        result = []
        element_number = 1
        n = self.ndof
        tolerance = 1e-10
        tags: dict[int, UniaxialMaterial] = {}
        for template, coefficient in definitions:
            self._validate_material(template)
            existing = tags.get(int(template.tag))
            if existing is not None and existing is not template:
                raise ValueError(f"Material tag {template.tag} is duplicated within the system")
            tags[int(template.tag)] = template
            scale = max(1.0, float(np.max(np.abs(coefficient))))
            tol = tolerance * scale
            if not np.allclose(coefficient, coefficient.T, rtol=1e-10, atol=tol):
                raise ValueError(f"Coefficient matrix for material template {template.tag!r} is not symmetric")
            if np.any(coefficient[np.triu_indices(n, 1)] > tol):
                raise ValueError(f"Off-diagonal coefficients for material template {template.tag!r} must be nonpositive")
            row_sums = coefficient.sum(axis=1)
            if np.any(row_sums < -tol):
                raise ValueError(f"Coefficient-matrix row sums for material template {template.tag!r} cannot be negative")
            reconstructed = np.zeros_like(coefficient)
            pieces: list[tuple[float, NDArray[np.float64], tuple[int, ...]]] = []
            for i, value in enumerate(row_sums):
                if value > tol:
                    b = np.zeros(n); b[i] = 1.0
                    pieces.append((float(value), b, (i,)))
                    reconstructed += value * np.outer(b, b)
            for i in range(n):
                for j in range(i + 1, n):
                    value = -coefficient[i, j]
                    if value > tol:
                        b = np.zeros(n); b[i], b[j] = -1.0, 1.0
                        pieces.append((float(value), b, (i, j)))
                        reconstructed += value * np.outer(b, b)
            if not np.allclose(coefficient, reconstructed, rtol=1e-9, atol=tol):
                raise ValueError(f"Material template {template.tag!r} cannot be decomposed into supported connections")
            for weight, b, dofs in pieces:
                info = ElementInfo(
                    id=f"E{element_number}", template_tag=template.tag, weight=weight,
                    dofs=dofs, deformation_coefficients=tuple(float(b[i]) for i in dofs),
                )
                result.append((info, b, template))
                element_number += 1
        return result

    def create_runtime(self) -> RuntimeModel:
        eles = []
        for info, b, template in self._definitions:
            material = template.getCopy()
            material.revertToStart()
            self._validate_material(material)
            eles.append(RuntimeElement(info, b.copy(), material))
        return RuntimeModel(self._numeric_stiffness.copy(), eles)

    def initial_stiffness(self) -> NDArray[np.float64]:
        matrix = self._numeric_stiffness.copy()
        for info, b, template in self._definitions:
            value = float(template.getInitialTangent())
            if not np.isfinite(value):
                raise ValueError(f"Material template {template.tag!r} returned a non-finite initial stiffness")
            matrix += info.weight * value * np.outer(b, b)
        return matrix

    def eigen(self, n_modes: int | None = None, influence: ArrayLike | None = None) -> ModalResult:
        stiffness = self.initial_stiffness()
        if not np.allclose(stiffness, stiffness.T, rtol=1e-10, atol=1e-12):
            raise ValueError("The initial stiffness matrix must be symmetric")
        values, vectors = eigh(stiffness, self.mass.values)
        if n_modes is not None:
            if n_modes <= 0 or n_modes > self.ndof:
                raise ValueError("n_modes must be between 1 and the number of DOFs")
            values, vectors = values[:n_modes], vectors[:, :n_modes]
        scale = max(1.0, float(np.max(np.abs(values))))
        tol = 1e-10 * scale
        classification = tuple("unstable" if x < -tol else "rigid" if abs(x) <= tol else "vibration" for x in values)
        omega = np.where(values > tol, np.sqrt(np.maximum(values, 0.0)), np.where(values < -tol, np.nan, 0.0))
        frequencies = omega / (2 * np.pi)
        periods = np.full_like(omega, np.inf)
        positive = omega > 0
        periods[positive] = 2 * np.pi / omega[positive]
        periods[np.isnan(omega)] = np.nan
        factors = masses = ratios = None
        if influence is not None:
            direction = np.asarray(influence, dtype=float)
            if direction.shape != (self.ndof,):
                raise ValueError("The influence vector length must equal the number of DOFs")
            factors = vectors.T @ self.mass.values @ direction
            masses = factors**2
            total = float(direction @ self.mass.values @ direction)
            ratios = masses / total if total > 0 else np.full_like(masses, np.nan)
        return ModalResult(values, omega, frequencies, periods, vectors, classification, factors, masses, ratios)

    def central_difference_limit(self) -> float:
        """Return the linear undamped reference limit 2 / omega_max."""
        omega = self.eigen().circular_frequencies
        finite_positive = omega[np.isfinite(omega) & (omega > 0)]
        return float("inf") if finite_positive.size == 0 else 2.0 / float(np.max(finite_positive))
