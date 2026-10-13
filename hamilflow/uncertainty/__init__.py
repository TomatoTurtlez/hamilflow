"""Uncertainty utilities for hamilflow.

Provides:
- ``read_deeph_hamiltonian``, ``write_deeph_hamiltonian``, ``average_predicted_hamiltonians``
- ``discover_structures``, ``link_ensemble_files``
- ``BandUncertaintyCalculator``
- ``BandUncertaintyPostProcessor``, ``UncertaintyCalibrator``
- ``MatrixUncertaintyCalculator``
- ``fit_rigid_shifts``, ``write_aligned_hamiltonian`` (rigid-shift alignment, see ``alignment``)
"""

from .hamiltonian_io import (
    read_deeph_hamiltonian,
    write_deeph_hamiltonian,
    average_predicted_hamiltonians,
)
from .ensemble_io import discover_structures, link_ensemble_files
from .band_uncertainty import BandUncertaintyCalculator
from .band_postprocess import BandUncertaintyPostProcessor, UncertaintyCalibrator
from .matrix_uncertainty import MatrixUncertaintyCalculator
from .alignment import fit_rigid_shifts, write_aligned_hamiltonian

__all__ = [
    "read_deeph_hamiltonian",
    "write_deeph_hamiltonian",
    "average_predicted_hamiltonians",
    "discover_structures",
    "link_ensemble_files",
    "BandUncertaintyCalculator",
    "BandUncertaintyPostProcessor",
    "UncertaintyCalibrator",
    "MatrixUncertaintyCalculator",
    "fit_rigid_shifts",
    "write_aligned_hamiltonian",
]
