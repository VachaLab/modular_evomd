# === genmethod.py ===
from __future__ import annotations
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from generator import Generator
    from sequence import Sequence

logger = logging.getLogger(__name__)


class GenMethod:
    """
    Abstract base class for sequence generation methods.

    Subclasses implement generate() to produce a new candidate sequence
    string from two parent Sequence objects. The parent objects are always
    provided by Evolver via Generator, even when a method does not use
    both of them.

    Access to shared resources (amino acid pool, peptide length) is provided
    through a reference to the parent Generator, injected automatically when
    the method is registered. Methods that do not need these resources can
    ignore the reference entirely.

    Subclasses that require physicochemical properties of the parents should
    access them directly from the Sequence objects (e.g. seq.residues,
    seq.charge). Subclasses that only need the sequence string should use
    str(seq) to avoid unnecessary object construction overhead.
    """
    method_name: str = 'EmptyMethod'
    expected_parents: int = 2

    def __init__(self) -> None:
        # Reference to the Generator that owns this method.
        # Injected automatically by Generator.register_method().
        # Provides access to aa_pool, peptide_len, and other shared state.
        self.generator: Generator | None = None

    def generate(self, seq1: Sequence, seq2: Sequence, verbose=False) -> str:
        """
        Produces a candidate sequence string from two parent Sequence objects.

        Both parents are always provided. Methods that use only one parent
        may ignore seq2. Methods that generate sequences from scratch (e.g.
        Random) may ignore both parents.

        Must return a plain string of amino acid single-letter codes.
        Validation is handled by Generator after this method returns.

        Raises NotImplementedError if not overridden by a subclass.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement generate()."
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


if __name__ == '__main__':
    pass