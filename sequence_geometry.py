# === sequence_geometry.py ===
"""
Helix geometry and hydrophobicity utilities for peptide sequences.

This module places a sequence's residues on an idealized alpha-helix and
derives geometric/physicochemical quantities from that layout. It is used both
by restrictions (e.g. hydrophobic moment / index bounds) and by GenMethod
subclasses that operate on helix faces.

Helix model:
    Residues are arranged on a helix with a fixed rotation per residue
    (_THETA_DEG) and a fixed height step (_INCREMENT). Position i is
    (cos(i*theta), -sin(i*theta), i*increment), so the XY projection is the
    helical wheel and Z is the axial rise.

Main quantities:
    - compute_helix_positions: the 3D coordinates per residue.
    - compute_hm_vector / compute_hm_scalar: hydrophobic moment as a direction
      and as a magnitude, summing residue hydrophobicities weighted by XY position.
    - compute_hi: the (summed or averaged) hydrophobic index.
    - align_to_minus_y / align_to: rotate the wheel so the HM vector points at a
      chosen direction.
    - get_faces / compute_sections: partition the helical wheel into angular
      regions (faces / equal sectors).

Hydrophobicity values come from each Residue (res.hydrophobicity), which were
set from the scale chosen when the parent Sequence was built; these functions
do not take a scale argument of their own.
"""

import math
import numpy as np
from sequence import Sequence
from intervals import CircleInterval

# --- Helix geometry constants ---
_THETA_DEG: float = 100.0   # rotation per residue in degrees (ideal alpha-helix)
_INCREMENT: float = -1.0    # axial height increment per residue


# --- Geometry functions ----------------------------------------------------

def compute_helix_positions(seq: Sequence, translate: bool = False) -> np.ndarray:
    """
    Compute 3D helix coordinates for each residue using the fixed helix params.

    Position i is (cos(i*theta), -sin(i*theta), i*increment), with theta from
    _THETA_DEG and increment from _INCREMENT. When translate is True, the helix
    is centered at the origin by subtracting the mean position.

    Args:
        seq (Sequence): Sequence whose residues are laid out (only its length
            is used here).
        translate (bool): If True, center the coordinates at (0, 0, 0).

    Returns:
        np.ndarray: Array of shape (n_residues, 3) with one row per residue.
    """
    theta_rad = np.deg2rad(_THETA_DEG)
    positions = np.array([
        [
            np.cos(i * theta_rad),
            -np.sin(i * theta_rad),
            i * _INCREMENT,
        ]
        for i in range(len(seq))
    ])
    if translate:
        positions -= positions.mean(axis=0)
    return positions


def compute_hm_vector(seq: Sequence, positions: np.ndarray) -> np.ndarray:
    """
    Compute the hydrophobic moment direction as a normalized 2D vector.

    Sums each residue's XY position weighted by its hydrophobicity, then
    normalizes. Falls back to [1, 0] when the moment is zero (undefined
    direction).

    Args:
        seq (Sequence): Sequence providing residue hydrophobicities.
        positions (np.ndarray): Helix coordinates from compute_helix_positions.

    Returns:
        np.ndarray: Unit 2D vector [x, y] giving the HM direction.
    """
    hm = np.zeros(2)
    for res, pos in zip(seq.residues, positions):
        hm += res.hydrophobicity * pos[:2]
    norm = np.linalg.norm(hm)
    if norm == 0:
        return np.array([1.0, 0.0])
    return hm / norm


def compute_hm_scalar(seq: Sequence, positions: np.ndarray, average: bool = False) -> float:
    """
    Compute the hydrophobic moment magnitude (its scalar length).

    Sums each residue's XY position weighted by hydrophobicity and returns the
    norm of the resulting 2D vector, optionally divided by the number of
    residues. Should be computed before any in-place alignment that would
    rotate `positions`.

    Args:
        seq (Sequence): Sequence providing residue hydrophobicities.
        positions (np.ndarray): Helix coordinates from compute_helix_positions.
        average (bool): If True, divide the moment by the residue count.

    Returns:
        float: The hydrophobic moment magnitude, rounded to 4 decimals.
    """
    hm = np.zeros(2)
    for res, pos in zip(seq.residues, positions):
       hm += res.hydrophobicity * pos[:2]
    if average:
        hm /= len(seq.residues)
    hm = round(float(np.linalg.norm(hm)), 3)
    return hm 


def compute_hi(seq: Sequence, average: bool = False) -> float:
    """
    Compute the hydrophobic index: the sum (or mean) of residue hydrophobicities.

    Args:
        seq (Sequence): Sequence providing residue hydrophobicities.
        average (bool): If True, divide the sum by the residue count.

    Returns:
        float: The hydrophobic index, rounded to 4 decimals.
    """
    hi = 0
    for res in seq.residues:
       hi += res.hydrophobicity
    if average:
        hi /= len(seq.residues)
    hi = round(float(hi), 3)
    return hi


def align_to_minus_y(positions: np.ndarray, hm_vector: np.ndarray) -> np.ndarray:
    """
    Rotate positions in the XY plane so the HM vector points toward -Y.

    The Z coordinates are left unchanged. Convenience wrapper equivalent to
    align_to(positions, hm_vector, target=[0, -1]).

    Args:
        positions (np.ndarray): Helix coordinates to rotate.
        hm_vector (np.ndarray): Current HM direction (from compute_hm_vector).

    Returns:
        np.ndarray: A rotated copy of `positions`.
    """
    target = np.array([0.0, -1.0])
    hx, hy = hm_vector
    tx, ty = target
    angle = np.arctan2(ty, tx) - np.arctan2(hy, hx)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotated = positions.copy()
    rotated[:, 0] = positions[:, 0] * cos_a - positions[:, 1] * sin_a
    rotated[:, 1] = positions[:, 0] * sin_a + positions[:, 1] * cos_a
    return rotated


def align_to(positions: np.ndarray, hm_vector: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Rotate positions in the XY plane so the HM vector points toward `target`.

    The Z coordinates are left unchanged.

    Args:
        positions (np.ndarray): Helix coordinates to rotate.
        hm_vector (np.ndarray): Current HM direction (from compute_hm_vector).
        target (np.ndarray): Desired 2D direction, e.g. [0, -1] for -Y.

    Returns:
        np.ndarray: A rotated copy of `positions`.
    """
    hx, hy = hm_vector
    tx, ty = target
    angle = np.arctan2(ty, tx) - np.arctan2(hy, hx)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotated = positions.copy()
    rotated[:, 0] = positions[:, 0] * cos_a - positions[:, 1] * sin_a
    rotated[:, 1] = positions[:, 0] * sin_a + positions[:, 1] * cos_a
    return rotated


def get_faces(
    positions: np.ndarray,
    seq: Sequence,
    phi_deg: float,
    ref_angle: float = np.pi * 3/2
) -> tuple:
    """
    Split residues into two faces by angular proximity to a reference direction.

    Residues whose helical-wheel angle falls within +/- phi_deg/2 of `ref_angle`
    form the positive face; the rest form the negative face. The default
    ref_angle (3*pi/2, i.e. -Y) matches the HM direction after alignment.

    Args:
        positions (np.ndarray): Helix coordinates (XY used for the wheel angle).
        seq (Sequence): Sequence providing residue indices.
        phi_deg (float): Total slice width in degrees; phi_deg/2 on each side.
        ref_angle (float): Center of the slice, in radians.

    Returns:
        tuple: (positive_face, negative_face) as lists of residue indices.
    """
    border = np.deg2rad(phi_deg / 2)
    interval = CircleInterval(start=ref_angle-border, end=ref_angle+border, lclosed=False, rclosed=False)
    positive_face = []
    negative_face = []
    for res, pos in zip(seq.residues, positions):
        angle = np.mod(math.atan2(pos[1], pos[0]), 2*np.pi)
        if interval(angle):
            positive_face.append(res.index)
        else:
            negative_face.append(res.index)
    
    return positive_face, negative_face


def compute_sections(n: int) -> list:
    """
    Divide the full circle into n equal angular sections as CircleIntervals.

    Section 0 is centered (bisector) on -Y (-pi/2); sections are numbered
    counterclockwise. Each section is left-open / right-closed.

    Args:
        n (int): Number of equal sections.

    Returns:
        list: n CircleInterval objects covering the full circle.
    """
    segment_angle = 2 * math.pi / n
    bisector_0 = -math.pi / 2
    section_start = bisector_0 - segment_angle / 2
    intervals = []
    for i in range(n):
        start = section_start + i * segment_angle
        end = start + segment_angle
        intervals.append(
            CircleInterval(start=start, end=end, lclosed=False, rclosed=True)
        )
    return intervals


if __name__ == '__main__':
    pass
