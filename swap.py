# === swap.py ===
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
    align_to_minus_y,
)

if TYPE_CHECKING:
    from generator import Generator

logger = logging.getLogger(__name__)


class Swap(GenMethod):
    """
    Generates a child sequence by replacing a contiguous fragment of one
    parent with spatially equivalent residues from the other parent.

    Both parents must have the same length. The child always has the same
    length as the parents.

    Algorithm
    ---------
    1. One parent is chosen at random as the receptor (A) and the other
       as the donor (B).
    2. The helix positions of both parents are computed and independently
       aligned to their own hydrophobic moment vectors.
    3. A contiguous fragment of parent A is selected at random. The fragment
       start index is drawn uniformly across the sequence and its length is
       drawn uniformly from 1 to max(1, len // 4).
    4. Each vacant position in the child (positions within the removed
       fragment) is filled with the spatially closest available residue
       from parent B, measured as the 3D Euclidean distance between aligned
       helix positions. Each residue of parent B is used at most once.
    5. All positions outside the fragment are kept from parent A unchanged.

    The spatial search over all residues of parent B (not just those at
    equivalent primary sequence positions) ensures that the structural
    context of the helix is respected when reconstructing the fragment,
    producing children that maintain the amphipathic character of the
    parents.

    This method does not require the amino acid pool.
    """

    def __init__(self) -> None:
        super().__init__()

    def _aligned_positions(self, seq: Sequence) -> np.ndarray:
        """
        Returns the 3D helix positions for seq after aligning the
        hydrophobic moment vector to point toward -Y.
        """
        positions = compute_helix_positions(seq)
        hm_vector = compute_hm_vector(seq, positions)
        return align_to_minus_y(positions, hm_vector)

    def _as_sequence(self, seq: Sequence | str) -> Sequence:
        """
        Returns a Sequence object. Instantiates one only when the input
        is a plain string, avoiding redundant construction otherwise.
        """
        if isinstance(seq, Sequence):
            return seq
        return Sequence(str(seq))

    def generate(self, seq1: Sequence, seq2: Sequence) -> str:
        s1 = str(seq1)
        s2 = str(seq2)
        length = len(s1)

        # Both parents must have the same length.
        if length != len(s2):
            raise ValueError(
                f"Swap: both parents must have the same length, "
                f"got {length} and {len(s2)}."
            )

        # Sequence objects are required for helix geometry computation.
        parent_a = self._as_sequence(seq1)
        parent_b = self._as_sequence(seq2)

        # Randomly assign receptor (A) and donor (B).
        if random.random() < 0.5:
            parent_a, parent_b = parent_a, parent_b

        logger.debug(f"Swap: receptor='{parent_a}' donor='{parent_b}'")

        # Compute aligned helix positions for both parents.
        pos_a = self._aligned_positions(parent_a)
        pos_b = self._aligned_positions(parent_b)

        # Select a random contiguous fragment from parent A.
        max_fragment = max(1, length // 4)
        fragment_len = random.randint(1, max_fragment)
        fragment_start = random.randint(0, length - fragment_len)
        vacant_indices = list(range(fragment_start, fragment_start + fragment_len))

        logger.debug(
            f"Swap: fragment [{fragment_start}:{fragment_start + fragment_len}] "
            f"(length {fragment_len})"
        )

        # Seed the child from parent A. Non-vacant positions are final.
        child: list[str] = list(str(parent_a))

        # Mutable pool of all parent B residues and their aligned positions.
        # Each entry is consumed once as it is assigned to a child position.
        donor_residues: list[str] = list(str(parent_b))
        donor_positions: list[np.ndarray] = list(pos_b)

        for idx in vacant_indices:
            if not donor_positions:
                # All donor residues have been consumed. Retain the parent A
                # residue at remaining vacant positions.
                logger.debug(
                    f"Swap: donor pool exhausted at index {idx}, "
                    f"retaining parent A residue '{child[idx]}'."
                )
                continue

            # Find the donor residue whose aligned position is spatially
            # closest to the vacant position in parent A.
            target = pos_a[idx]
            distances = [
                float(np.linalg.norm(target - dp))
                for dp in donor_positions
            ]
            closest = int(np.argmin(distances))

            logger.debug(
                f"Swap: index {idx} <- '{donor_residues[closest]}' "
                f"from donor (distance {distances[closest]:.3f})"
            )

            child[idx] = donor_residues[closest]

            # Remove the consumed residue and position from the pools.
            donor_residues.pop(closest)
            donor_positions.pop(closest)

        result = ''.join(child)
        logger.debug(f"Swap: '{parent_a}' + '{parent_b}' -> '{result}'")
        return result

    def __repr__(self) -> str:
        return "Swap()"
        