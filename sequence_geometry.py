# === sequence_geometry.py ===
import math
import numpy as np
from sequence import Sequence
from intervals import CircleInterval

# --- Helix geometry constants ---
_THETA_DEG: float = 100.0   # rotation per residue in degrees
_INCREMENT: float = -1.0     # height increment per residue

# --- Geometry functions ----------------------------------------------------

def compute_helix_positions(seq: Sequence) -> np.ndarray:
    """
    Computes 3D helix positions for each residue using fixed helix parameters.
    Centers the helix at (0, 0, 0) by subtracting the mean position.
    Returns an array of shape (n_residues, 3).
    """
    theta_rad = np.deg2rad(_THETA_DEG)
    positions = np.array([
        [
            np.cos(i * theta_rad),
            np.sin(i * theta_rad),
            i * _INCREMENT,
        ]
        for i in range(len(seq))
    ])
    positions -= positions.mean(axis=0)
    return positions


def compute_hm_vector(seq: Sequence, positions: np.ndarray) -> np.ndarray:
    """
    Computes the hydrophobic moment vector from residue hydrophobicities
    and their XY positions. Returns a normalized 2D vector [x, y].
    """
    hm = np.zeros(2)
    for res, pos in zip(seq.residues, positions):
        hm += res.hydrophobicity * pos[:2]
    norm = np.linalg.norm(hm)
    if norm == 0:
        return np.array([1.0, 0.0])
    return hm / norm


def compute_hm_scalar(seq: Sequence, positions: np.ndarray) -> float:
    """
    Computes the hydrophobic moment scalar magnitude before alignment.
    Must be called before align_to_minus_y() to preserve the original vector.
    """
    hm = np.zeros(2)
    for res, pos in zip(seq.residues, positions):
        hm += res.hydrophobicity * pos[:2]
    return float(np.linalg.norm(hm))


def align_to_minus_y(positions: np.ndarray, hm_vector: np.ndarray) -> np.ndarray:
    """
    Rotates all positions in the XY plane so that the hydrophobic moment
    vector points toward -Y. Z coordinates are preserved unchanged.
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

def get_faces(
    positions: np.ndarray,
    seq: Sequence,
    phi_deg: float,
) -> tuple:
    """
    Splits residues into two faces based on their proximity to the hydrophobic
    moment vector direction (-Y after alignment).
    phi_deg is the total slice angle in degrees; phi/2 is applied on each side
    of the HM vector.
    Returns (positive_face, negative_face) as lists of residue indices.
    Reusable by GenMethod subclasses that work with helix geometry.
    """
    border = np.cos(np.deg2rad(phi_deg / 2))
    positive_face = [res.index for res, pos in zip(seq.residues, positions) if -pos[1] >= border]
    negative_face = [res.index for res, pos in zip(seq.residues, positions) if -pos[1] < border]
    return positive_face, negative_face


def compute_sections(n: int) -> list:
    """
    Divides the full circle (2pi) into n equal CircleInterval sections.
    Section 0 has its bisector pointing toward -Y (-pi/2).
    Sections are numbered counterclockwise.
    Returns a list of CircleInterval objects.
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
