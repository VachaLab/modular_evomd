# === genmethod.py ===
"""
GenMethod: abstract base class for sequence generation methods.

A GenMethod produces one candidate sequence string per call. Concrete methods
(Swap, Hybrid, FacesMix, Pattern, Flanks, etc.) subclass GenMethod and
implement generate(). The Generator owns one or more GenMethod instances,
selects among them, and validates whatever string they return.

Contract for subclasses:
  - generate(seq1, seq2, verbose) returns a plain amino-acid string. Validation
    against restrictions is the Generator's job, not the method's.
  - Two parents are always passed in by the Generator, even when a method uses
    fewer. `expected_parents` declares how many a method actually needs, so the
    Generator can filter out methods it cannot satisfy with the parents at hand
    (e.g. a 2-parent method when only one individual is available).
  - Shared resources (amino acid pool, target length) are reached through
    self.generator, a back-reference the Generator injects on registration.
    Methods that do not need them can ignore it.

Subclasses needing physicochemical properties should read them straight from
the Sequence objects (seq.residues, seq.charge, ...). Those needing only the
letters should use str(seq) to avoid building extra objects.
"""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.generator import Generator
    from ..core.sequence import Sequence

logger = logging.getLogger(__name__)


class GenMethod:
    """
    Abstract base class for sequence generation methods.

    Subclasses implement generate() to produce a new candidate sequence string
    from two parent Sequence objects. The parents are always provided by Evolver
    via Generator, even when a method does not use both.

    Access to shared resources (amino acid pool, peptide length) is provided
    through a reference to the parent Generator, injected automatically when the
    method is registered. Methods that do not need these resources can ignore
    the reference entirely.

    Subclasses that require physicochemical properties of the parents should
    access them directly from the Sequence objects (e.g. seq.residues,
    seq.charge). Subclasses that only need the sequence string should use
    str(seq) to avoid unnecessary object construction overhead.

    Class attributes:
        method_name (str): Short label used in logs and verbose output.
        expected_parents (int): Number of parents the method requires; the
            Generator uses it to filter methods when fewer parents are available.
    """

    method_name: str = 'EmptyMethod'
    expected_parents: int = 2

    def __init__(self) -> None:
        # Back-reference to the owning Generator, injected by
        # Generator._register_method(). Provides access to aa_pool,
        # peptide_len, and other shared state. None until registered.
        self.generator: Generator | None = None

    def generate(self, seq1: Sequence, seq2: Sequence, verbose=False) -> str:
        """
        Produce a candidate sequence string from two parent Sequence objects.

        Both parents are always provided. Methods that use only one parent may
        ignore seq2; methods that build from scratch may ignore both. Must
        return a plain string of amino-acid single-letter codes; validation is
        handled by the Generator after this method returns.

        Args:
            seq1 (Sequence): First parent.
            seq2 (Sequence): Second parent.
            verbose (bool): If True, print a human-readable trace of the step.

        Returns:
            str: The generated candidate sequence.

        Raises:
            NotImplementedError: If a subclass does not override this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement generate()."
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


if __name__ == '__main__':
    pass
