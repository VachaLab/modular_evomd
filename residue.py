# === residue.py ===
"""
Residue: a single amino acid within a peptide sequence.

A Residue bundles the physicochemical properties of one amino acid, looked up
from the Scales tables by its one-letter code: charge, volume, mass,
hydrophobicity (on a chosen scale), and a group classification. It also records
its position (index) within the parent sequence.

Residue holds only per-residue scalar properties. Geometry and any spatial
computation (helix positions, hydrophobic moment, etc.) are handled externally
by the GenMethod subclasses, not here.
"""

import logging
from scales import Scales

logger = logging.getLogger(__name__)


class Residue:
    """
    Represents a single amino acid residue within a peptide sequence.

    Stores physicochemical properties looked up from Scales, indexed by the
    residue's position. Geometry and spatial calculations are handled
    externally by GenMethod subclasses.
    """

    def __init__(self, letter: str, index: int = 0, scale: str = 'eisenberg') -> None:
        """
        Build a Residue from its one-letter code and position.

        Args:
            letter (str): One-letter amino acid code (used as the Scales key).
            index (int): Position of the residue within the parent sequence.
            scale (str): Hydrophobicity scale name used to look up the
                residue's hydrophobicity value.
        """
        self.letter: str = letter
        self.index: int = index
        self.charge: float = Scales.aa_charges[self.letter]                       # net charge of the residue
        self.volume: float = Scales.aa_volumes[self.letter]                       # side-chain volume
        self.mass: float = Scales.aa_masses[self.letter]                          # residue mass
        self.hydrophobicity: float = Scales.hydrophobicity_scales[scale][self.letter]  # hydrophobicity on the chosen scale
        self.group: int = Scales.aa_group[self.letter]                            # signed chemical-group id (-3 neg, -2 amide, -1 polar, 0 non-polar, 1 aliphatic, 2 aromatic, 3 positive)

    def __str__(self) -> str:
        """Return the one-letter code."""
        return self.letter

    def __repr__(self) -> str:
        """Return an unambiguous representation with letter and index."""
        return f"Residue(letter={self.letter!r}, index={self.index})"


if __name__ == '__main__':
    pass
