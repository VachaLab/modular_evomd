# === residue.py ===
import logging
from scales import Scales

logger = logging.getLogger(__name__)


class Residue:
    """
    Represents a single amino acid residue within a peptide sequence.
    Stores physicochemical properties indexed by position.
    Geometry and spatial calculations are handled externally by GenMethod subclasses.
    """

    def __init__(self, letter: str, index: int = 0, scale: str = 'eisenberg') -> None:
        self.letter: str = letter
        self.index: int = index
        self.charge: float = Scales.aa_charges[self.letter]
        self.volume: float = Scales.aa_volumes[self.letter]
        self.mass: float = Scales.aa_masses[self.letter]
        self.hydrophobicity: float = Scales.hydrophobicity_scales[scale][self.letter]
        self.group: int = Scales.aa_group[self.letter]

    def __str__(self) -> str:
        return self.letter

    def __repr__(self) -> str:
        return f"Residue(letter={self.letter!r}, index={self.index})"


if __name__ == '__main__':
    pass
    