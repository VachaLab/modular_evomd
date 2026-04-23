# === directed_mutations.py ===
from __future__ import annotations
import logging
import random
from typing import TYPE_CHECKING

from genmethod import GenMethod
from scales import Scales

if TYPE_CHECKING:
    from sequence import Sequence

logger = logging.getLogger(__name__)

# Small epsilon to avoid division by zero when hydrophobicity values are identical.
_EPSILON: float = 1e-8

# Eisenberg hydrophobicity scale used by both methods as the reference.
_EISENBERG: dict[str, float] = Scales.hydrophobicity_scales['eisenberg']

# Group assignments from Scales, used by GroupMutation.
_AA_GROUP: dict[str, int] = Scales.aa_group


class GroupMutation(GenMethod):
    """
    Generates a mutant by replacing one residue with another from the same
    chemical group.

    Groups are defined in Scales.aa_group as integer labels:
        -3: negative (D, E)
        -2: amides   (N, Q, H)
        -1: polar    (S, T)
         0: non-polar thioether (C, M)
         1: aliphatic (A, G, I, L, P, V)
         2: aromatic  (F, W, Y)
         3: positive  (R, K)

    Algorithm
    ---------
    1. A position is selected at random from the sequence.
    2. The group of the current residue at that position is identified.
    3. All residues in the amino acid pool that belong to the same group
       and are not identical to the current residue are collected as
       candidates.
    4. One candidate is selected uniformly at random and placed at the
       mutated position.

    If no valid candidate exists in the pool for the selected group (e.g.
    a highly restricted pool that contains only one residue per group),
    the method logs a warning and returns the parent sequence unchanged.

    seq2 is always ignored. This method only requires seq1.
    """

    def generate(self, seq1: Sequence, seq2: Sequence) -> str:
        pool: str = self.generator.aa_pool
        seq_str: str = str(seq1)
        length: int = len(seq_str)

        # Select a random position to mutate.
        idx: int = random.randint(0, length - 1)
        current_aa: str = seq_str[idx]
        current_group: int = _AA_GROUP.get(current_aa, 0)

        logger.debug(
            f"GroupMutation: position {idx} '{current_aa}' "
            f"(group {current_group})"
        )

        # Collect candidates: same group, different residue, within pool.
        candidates: list[str] = [
            aa for aa in pool
            if _AA_GROUP.get(aa, 0) == current_group and aa != current_aa
        ]

        if not candidates:
            logger.warning(
                f"GroupMutation: no candidates found for residue '{current_aa}' "
                f"(group {current_group}) in pool '{pool}'. "
                f"Returning parent sequence unchanged."
            )
            return seq_str

        new_aa: str = random.choice(candidates)

        logger.debug(
            f"GroupMutation: '{current_aa}' -> '{new_aa}' at position {idx}"
        )

        return seq_str[:idx] + new_aa + seq_str[idx + 1:]

    def __repr__(self) -> str:
        return "GroupMutation()"


class HydrophobicityMutation(GenMethod):
    """
    Generates a mutant by replacing one residue with another selected
    proportionally by hydrophobicity similarity.

    Hydrophobicity values are taken from the Eisenberg scale defined in
    Scales.hydrophobicity_scales['eisenberg'].

    Algorithm
    ---------
    1. A position is selected at random from the sequence.
    2. The Eisenberg hydrophobicity of the current residue is read.
    3. For every residue in the amino acid pool that is not identical to
       the current residue, a weight is computed as:
           weight = 1 / (|h_current - h_candidate| + epsilon)
       where epsilon avoids division by zero for identical hydrophobicity
       values. Residues with similar hydrophobicity receive higher weights.
    4. One candidate is selected using weighted random sampling.

    If no valid candidate exists in the pool (pool contains only the
    current residue), the method logs a warning and returns the parent
    sequence unchanged.

    seq2 is always ignored. This method only requires seq1.
    """

    def generate(self, seq1: Sequence, seq2: Sequence) -> str:
        pool: str = self.generator.aa_pool
        seq_str: str = str(seq1)
        length: int = len(seq_str)

        # Select a random position to mutate.
        idx: int = random.randint(0, length - 1)
        current_aa: str = seq_str[idx]
        current_h: float = _EISENBERG.get(current_aa, 0.0)

        logger.debug(
            f"HydrophobicityMutation: position {idx} '{current_aa}' "
            f"(h={current_h:.3f})"
        )

        # Collect candidates: any residue in the pool except the current one.
        candidates: list[str] = [aa for aa in pool if aa != current_aa]

        if not candidates:
            logger.warning(
                f"HydrophobicityMutation: no candidates found for residue "
                f"'{current_aa}' in pool '{pool}'. "
                f"Returning parent sequence unchanged."
            )
            return seq_str

        # Compute similarity weights: higher weight for closer hydrophobicity.
        weights: list[float] = [
            1.0 / (abs(current_h - _EISENBERG.get(aa, 0.0)) + _EPSILON)
            for aa in candidates
        ]

        new_aa: str = random.choices(candidates, weights=weights, k=1)[0]

        logger.debug(
            f"HydrophobicityMutation: '{current_aa}' (h={current_h:.3f}) -> "
            f"'{new_aa}' (h={_EISENBERG.get(new_aa, 0.0):.3f}) "
            f"at position {idx}"
        )

        return seq_str[:idx] + new_aa + seq_str[idx + 1:]

    def __repr__(self) -> str:
        return "HydrophobicityMutation()"
        