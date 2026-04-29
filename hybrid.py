# === hybrid.py ===
from __future__ import annotations
import logging
import random
from typing import TYPE_CHECKING

from genmethod import GenMethod

if TYPE_CHECKING:
    from sequence import Sequence

logger = logging.getLogger(__name__)


class Hybrid(GenMethod):
    """
    Generates a child sequence by combining fragments from two parent sequences.

    Both parents are split at a crossover point and the leading fragment of
    seq1 is joined with the trailing fragment of seq2. When the parents have
    equal length the child length is preserved. When the parents differ in
    length the crossover point is determined by a relative position (fraction
    of each sequence), so the proportional structure of both parents is
    respected and the child length can vary between calls.

    The crossover fraction is drawn uniformly from (0, 1) on each call,
    excluding the endpoints to ensure that both parents always contribute
    at least one residue to the child.

    Parameters
    ----------
    None. All shared resources are accessed through self.generator when
    needed. This method does not require the amino acid pool.
    """
    method_name = 'HYBRID'

    def generate(self, seq1: Sequence, seq2: Sequence) -> str:
        ornament = int((30 - len(self.method_name))/2)
        logger.debug(f"{'-' * ornament} {self.method_name} {'-' * ornament}")
        s1 = str(seq1)
        s2 = str(seq2)
        len1 = len(s1)
        len2 = len(s2)

        # Draw a crossover fraction strictly between 0 and 1 so that both
        # parents always contribute at least one residue.
        fraction = random.uniform(0.0, 1.0)

        # Map the fraction to an absolute cut index in each parent.
        # round() is used so that the cut index scales proportionally with
        # sequence length. Clamping ensures the index stays within bounds
        # even after rounding at the extremes.
        cut1 = max(1, min(round(fraction * len1), len1 - 1))
        cut2 = max(1, min(round(fraction * len2), len2 - 1))

        child = s1[cut1:] + s2[:cut2]

        logger.debug(f"Fraction={fraction:.3f}")
        logger.debug(f"{s1} <- Parent 1")
        logger.debug(f"{s2} <- Parent 2")
        logger.debug(f"{s1[cut1:]}{' '*len(s2[:cut2])}")
        logger.debug(f"{' '*len(s1[cut1:])}{s2[:cut2]}")
        logger.debug(f"{child} <- Child")
        logger.debug(f"{'-' * 30}")

        return child

if __name__ == '__main__':
    pass
