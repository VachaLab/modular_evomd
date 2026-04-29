# === faces_mix.py ===
from __future__ import annotations
import logging
import random
import numpy as np
from typing import TYPE_CHECKING

from genmethod import GenMethod
from sequence import Sequence
from sequence_geometry import (
    compute_helix_positions,
    compute_hm_vector,
    align_to,
    get_faces,
)

if TYPE_CHECKING:
    from generator import Generator

logger = logging.getLogger(__name__)

# Allowed range for the slice angle in degrees.
_SLICE_ANGLE_MIN: float = 20.0
_SLICE_ANGLE_MAX: float = 340.0

# Minimum sequence length required for a meaningful helix face split.
_MIN_LENGTH: int = 18


class FacesMix(GenMethod):
    """
    Generates a child sequence by reconstructing one face of a parent
    using the spatially equivalent residues from the other parent.

    Both parents must have the same length and must contain at least 18
    residues, since shorter sequences do not complete a full helix turn
    and produce degenerate face assignments.

    Algorithm
    ---------
    1. One parent is chosen at random as the base (parent A) and the other
       as the donor (parent B).
    2. The helix positions of parent A are computed and aligned to its
       hydrophobic moment vector. The residues are split into a hydrophobic
       face and a hydrophilic face using the configured slice angle.
    3. One face of parent A is chosen at random to be replaced.
    4. The helix positions of parent B are computed and aligned to its own
       hydrophobic moment vector.
    5. Each vacant position in the child (positions belonging to the
       replaced face of parent A) is filled with the spatially closest
       available residue from parent B, measured as the 3D Euclidean
       distance between aligned helix positions. Each residue of parent B
       is used at most once.
    6. The conserved face of parent A is kept unchanged.

    Because both parents are independently aligned to their own hydrophobic
    moment vectors before the spatial search, residues are matched across
    structurally equivalent regions of the helix. Face mixing at the edges
    is minimal and does not affect the overall amphipathic character of the
    child.

    Parameters
    ----------
    slice_angle : float
        Total angular width of the hydrophobic face slice in degrees.
        Must be between 20 and 340. The slice is centered on the
        hydrophobic moment vector direction. Default is 180.
    """
    method_name = 'FacesMix'

    def __init__(self, slice_angle: float = 120.0) -> None:
        super().__init__()

        if not _SLICE_ANGLE_MIN <= slice_angle <= _SLICE_ANGLE_MAX:
            raise ValueError(
                f"FacesMix: slice_angle must be between {_SLICE_ANGLE_MIN} "
                f"and {_SLICE_ANGLE_MAX}, got {slice_angle}."
            )
        self.slice_angle: float = slice_angle

    def _aligned_positions(self, seq: Sequence) -> np.ndarray:
        """
        Returns the 3D helix positions for seq after aligning the
        hydrophobic moment vector to point toward -Y.
        """
        positions = compute_helix_positions(seq)
        hm_vector = compute_hm_vector(seq, positions)
        return align_to(positions, hm_vector, target=np.array([1., 0.]))

    def _as_sequence(self, seq: Sequence | str) -> Sequence:
        """
        Returns a Sequence object. Instantiates one only when the input
        is a plain string, avoiding redundant construction otherwise.
        """
        if isinstance(seq, Sequence):
            return seq
        return Sequence(str(seq))

    def generate(self, seq1: Sequence, seq2: Sequence) -> str:
        ornament = int((30 - len(self.method_name))/2)
        logger.debug(f"{'-' * ornament} {self.method_name} {'-' * ornament}")

        s1 = str(seq1)
        s2 = str(seq2)

        # Both parents must have the same length.
        if len(s1) != len(s2):
            raise ValueError(
                f"FacesMix: both parents must have the same length, "
                f"got {len(s1)} and {len(s2)}."
            )

        # Enforce minimum length for meaningful face assignment.
        if len(s1) < _MIN_LENGTH:
            raise ValueError(
                f"FacesMix: sequences must have at least {_MIN_LENGTH} "
                f"residues, got {len(s1)}."
            )

        # Sequence objects are required for helix geometry computation.
        parent_a = self._as_sequence(seq1)
        parent_b = self._as_sequence(seq2)

        # Randomly assign which parent is the base (A) and which is the
        # donor (B).
        if random.random() < 0.5:
            parent_a, parent_b = parent_a, parent_b
        else:
            parent_a, parent_b = parent_b, parent_a

        logger.debug(f"{parent_a} <- Parent 1")
        logger.debug(f"{parent_b} <- Parent 2")

        # Compute aligned helix positions for both parents.
        pos_a = self._aligned_positions(parent_a)
        pos_b = self._aligned_positions(parent_b)

        # Split parent A into hydrophobic and hydrophilic faces.
        hydrophobic_face, hydrophilic_face = get_faces(
            pos_a, parent_a, self.slice_angle, ref_angle=0
        )


        # Randomly select which face of parent A will be replaced.
        if random.random() < 0.5:
            vacant_indices = hydrophobic_face
            protoseq = [k.letter if k.index in hydrophilic_face else ' ' for k in parent_a.residues]
            logger.debug("Replacing hydrophobic face")
        else:
            vacant_indices = hydrophilic_face
            protoseq = [k.letter if k.index in hydrophobic_face else ' ' for k in parent_a.residues]
            logger.debug("Replacing hydrophilic face")

        logger.debug(f"{''.join(protoseq)} <- Base face")

        # Seed the child from parent A. Conserved positions are final.
        # Vacant positions will be overwritten from parent B.
        child: list[str] = list(str(parent_a))

        # Mutable pool of all parent B residues and their aligned positions.
        # Each entry is consumed once as it is assigned to a child position.
        donor_residues: list[str] = list(str(parent_b))
        donor_positions: list[np.ndarray] = list(pos_b)

        for idx in vacant_indices:
            if not donor_positions:
                # All donor residues have been consumed. Keep the parent A
                # residue at the remaining vacant positions.
                logger.debug(
                    f"FacesMix: donor pool exhausted at index {idx}, "
                    f"retaining parent A residue '{child[idx]}'."
                )
                continue

            # Find the donor residue whose aligned position is closest to
            # the vacant position in parent A.
            target = pos_a[idx]
            distances = [
                float(np.linalg.norm(target - dp))
                for dp in donor_positions
            ]
            closest = int(np.argmin(distances))

            logger.debug(
                f"FacesMix: index {idx} <- '{donor_residues[closest]}' "
                f"from donor (distance {distances[closest]:.3f})"
            )

            child[idx] = donor_residues[closest]

            # Remove the consumed residue and position from the pools.
            donor_residues.pop(closest)
            donor_positions.pop(closest)

        result = ''.join(child)
        logger.debug(
            f"FacesMix: '{parent_a}' + '{parent_b}' -> '{result}'"
        )
        logger.debug(f"{'-' * 30}")
        return result

    def __repr__(self) -> str:
        return f"FacesMix(slice_angle={self.slice_angle})"
        

if __name__ == '__main__':
    pass
