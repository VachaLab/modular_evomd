# === restriction.py ===
"""
Restriction: pluggable sequence-validity constraints for the Generator.

A Restriction encapsulates one rule a candidate sequence must satisfy. The
Generator holds a list of them and, inside its generation loop, accepts a
candidate only when every restriction's test() returns True.

The module provides the abstract base class plus a set of built-in
restrictions:
  - CompositionRestriction: bounds the count of residues from a given set.
  - HdistributionRestriction: bounds the hydrophobicity alternation index.
  - PatternRestriction: rejects forbidden substrings / regex patterns.
  - ChargeRestriction: bounds the net charge.
  - HindexRestriction: bounds the hydrophobic index.
  - HmomentRestriction: bounds the hydrophobic moment.
  - ForbiddenSequence: rejects exact full-sequence matches.

Each test() updates self.message with a human-readable outcome; the Generator
only emits it at DEBUG level, so there is no overhead in normal operation.
Restrictions that need physicochemical values import Scales (or
sequence_geometry) lazily to avoid circular imports at module load time.
"""

from __future__ import annotations
from .sequence import Sequence
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class Restriction:
    """
    Abstract base class for sequence validity constraints.

    Subclasses implement test() to evaluate whether a candidate sequence
    string meets a specific requirement. Generator tests all registered
    Restriction instances inside the generation loop and only accepts a
    candidate when every restriction passes.

    The message attribute is updated on every test() call and is available
    for logging purposes. It is only emitted at DEBUG level by Generator,
    so there is no output overhead in normal operation.
    """

    def __init__(self) -> None:
        # Updated by test() on every call with a human-readable description
        # of the outcome. Populated even when the sequence passes, allowing
        # callers to log acceptance reasons if needed.
        self.message: str = ''

    def test(self, seq: str, verbose=False) -> bool:
        """
        Evaluate whether seq satisfies this restriction.

        Returns True if the sequence is accepted, False if rejected, and
        updates self.message with a description of the outcome.

        Raises:
            NotImplementedError: If not overridden by a subclass.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement test()."
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def _as_sequence(self, seq: Sequence | str, h_scale: str = 'eisenberg') -> Sequence:
        """
        Return a Sequence object for the input.

        Instantiates a new Sequence only when given a plain string, avoiding
        redundant construction when a Sequence is already provided.
        """
        if isinstance(seq, Sequence):
            return seq
        return Sequence(str(seq), h_scale=h_scale)


# ---------------------------------------------------------------------------
# Built-in restrictions
# ---------------------------------------------------------------------------

class CompositionRestriction(Restriction):
    """
    Accepts sequences that contain at least a minimum and at most a maximum
    count of residues belonging to a specified set.

    Useful for enforcing constraints such as "at least 2 positive residues"
    or "no more than 4 hydrophobic residues".

    Parameters
    ----------
    residues : str
        String of single-letter amino acid codes defining the target set.
        Case-insensitive.
    min : int
        Minimum number of residues from the set required. Default 0.
    max : int | None
        Maximum number of residues from the set allowed. None means no
        upper bound.
    """

    def __init__(
        self,
        residues: str,
        min: int = 0,
        max: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._residues: frozenset[str] = frozenset(residues.upper())
        self._min: int = min
        self._max: Optional[int] = max

        if self._max is not None and self._min > self._max:
            raise ValueError(
                f"CompositionRestriction: min ({self._min}) must be "
                f"<= max ({self._max})."
            )

    def test(self, seq: str, verbose=False) -> bool:
        """Return True if the count of target residues is within [min, max]."""
        count = sum(1 for aa in seq.upper() if aa in self._residues)
        above_min = count >= self._min
        below_max = self._max is None or count <= self._max

        passed = above_min and below_max
        bound_str = (
            f"[{self._min}, {self._max}]"
            if self._max is not None
            else f"[{self._min}, ∞)"
        )
        if passed:
            message = (
                f"count of {''.join(sorted(self._residues))} is {count}, "
                f"within {bound_str}"
            )
        else:
            message = (
                f"count of {''.join(sorted(self._residues))} is {count}, "
                f"outside {bound_str}"
            )

        if verbose:
            print(f"{message}. Pass: {passed}")
        return passed

    def __repr__(self) -> str:
        return (
            f"CompositionRestriction(residues='{''.join(sorted(self._residues))}', "
            f"min={self._min}, max={self._max})"
        )


class HdistributionRestriction(Restriction):
    """
    Accepts sequences whose hydrophobicity distribution (alternation index)
    falls within a specified range.

    The index is computed as the mean of the absolute hydrophobicity
    differences between adjacent residues, using the selected scale (Eisenberg
    by default). Residues absent from the scale are treated as neutral (0.0)
    and a warning is logged.

    Parameters
    ----------
    min : float | None
        Minimum accepted alternation index (inclusive). None means no lower bound.
    max : float | None
        Maximum accepted alternation index (inclusive). None means no upper bound.
    h_scale : str
        Name of the hydrophobicity scale used for the per-residue values.
    """

    def __init__(
        self,
        min: Optional[float] = None,
        max: Optional[float] = None,
        h_scale: str = 'eisenberg',
    ) -> None:
        super().__init__()

        if min is None and max is None:
            raise ValueError(
                "HdistributionRestriction requires at least one of "
                "'min' or 'max'."
            )
        self._min: Optional[float] = min
        self._max: Optional[float] = max

        if (
            self._min is not None
            and self._max is not None
            and self._min > self._max
        ):
            raise ValueError(
                f"HdistributionRestriction: min ({self._min}) must be "
                f"<= max ({self._max})."
            )
        
        self.h_scale = h_scale

        # Import here to avoid circular dependency at module level.
        from .scales import Scales
        self._hi_table: dict[str, float] = Scales.hydrophobicity_scales[self.h_scale]

    def test(self, seq: str, verbose=False) -> bool:
        """Return True if the mean adjacent-residue hydrophobicity difference is within [min, max]."""
        # 1. Map the sequence to numeric values, handling unknown residues.
        valores = []
        for aa in seq.upper():
            if aa not in self._hi_table:
                logger.warning(
                    f"HdistributionRestriction: unrecognized residue '{aa}' treated as neutral (0.0)."
                )
            valores.append(self._hi_table.get(aa, 0.0))

        # 2. Compute the alternation (distribution) index.
        if len(valores) < 2:
            dist_index = 0.0  # 0 or 1 residue -> no adjacent difference
        else:
            diferencias = [abs(valores[i+1] - valores[i]) for i in range(len(valores)-1)]
            dist_index = sum(diferencias) / len(diferencias)

        # 3. Evaluate against the min/max bounds.
        above_min = self._min is None or dist_index >= self._min
        below_max = self._max is None or dist_index <= self._max

        passed = above_min and below_max

        # 4. Report when verbose.
        if verbose:
            bound_str = f"[{self._min}, {self._max}]"
            print(f"Hdistribution restriction: {bound_str} Current: {dist_index:.4f} = Pass: {passed}")

        return passed

    def __repr__(self) -> str:
        return (
            f"HdistributionRestriction(min={self._min}, "
            f"max={self._max})"
        )


class PatternRestriction(Restriction):
    """
    Rejects sequences that contain any of the specified forbidden substrings
    or regular expression patterns.

    Each entry in patterns is first tried as a plain substring. Entries
    that contain regex metacharacters are compiled and matched as regular
    expressions.

    Parameters
    ----------
    patterns : list[str]
        Forbidden substrings or regex patterns. Case-insensitive matching
        is applied to all entries.
    """

    def __init__(self, patterns: list[str]) -> None:
        super().__init__()
        self._plain: list[str] = []
        self._regex: list[re.Pattern] = []

        # Separate plain strings from regex patterns for efficiency.
        _meta = set(r'\.^$*+?{}[]|()')
        for p in patterns:
            if any(c in _meta for c in p):
                self._regex.append(re.compile(p, re.IGNORECASE))
            else:
                self._plain.append(p.upper())

    def test(self, seq: str, verbose=False) -> bool:
        """Return True only if no forbidden substring or regex pattern is present."""
        seq_upper = seq.upper()

        passed = True
        message = "no forbidden patterns found"

        # Check plain substrings first.
        for pattern in self._plain:
            if pattern in seq_upper:
                message = f"forbidden substring '{pattern}' found in '{seq}'"
                passed = False

        # Check regex patterns.
        for rx in self._regex:
            match = rx.search(seq_upper)
            if match:
                message = (
                    f"forbidden pattern '{rx.pattern}' matched at "
                    f"position {match.start()} in '{seq}'"
                )
                passed = False

        if verbose:
            print(f"{message}. Pass: {passed}")
        return passed

    def __repr__(self) -> str:
        plain = self._plain
        regex = [rx.pattern for rx in self._regex]
        return f"PatternRestriction(patterns={plain + regex})"


class ChargeRestriction(Restriction):
    """
    Accepts sequences whose net charge falls within a specified range.

    Charge is computed as the sum of per-residue charges using the values
    defined in Scales.aa_charges. Residues not present in the scale are
    treated as neutral (charge 0.0) and a warning is logged.

    Parameters
    ----------
    min : float | None
        Minimum accepted net charge (inclusive). None means no lower bound.
    max : float | None
        Maximum accepted net charge (inclusive). None means no upper bound.
    """

    def __init__(
        self,
        min: Optional[float] = None,
        max: Optional[float] = None,
    ) -> None:
        super().__init__()

        if min is None and max is None:
            raise ValueError(
                "ChargeRestriction requires at least one of "
                "'min' or 'max'."
            )
        self._min: Optional[float] = min
        self._max: Optional[float] = max

        if (
            self._min is not None
            and self._max is not None
            and self._min > self._max
        ):
            raise ValueError(
                f"ChargeRestriction: min ({self._min}) must be "
                f"<= max ({self._max})."
            )

        # Import here to avoid circular dependency at module level.
        from .scales import Scales
        self._charge_table: dict[str, float] = Scales.aa_charges

    def test(self, seq: str, verbose=False) -> bool:
        """Return True if the summed net charge is within [min, max]."""
        charge = 0.0
        for aa in seq.upper():
            if aa not in self._charge_table:
                logger.warning(
                    f"ChargeRestriction: unrecognized residue '{aa}' treated as neutral."
                )
            charge += self._charge_table.get(aa, 0.0)

        above_min = self._min is None or charge >= self._min
        below_max = self._max is None or charge <= self._max

        passed = above_min and below_max

        if verbose:
            bound_str = f"[{self._min}, {self._max}]"
            print(f"Charge restriction: {bound_str} Current: {charge} = Pass: {passed}")

        return passed

    def __repr__(self) -> str:
        return (
            f"ChargeRestriction(min_charge={self._min}, "
            f"max_charge={self._max})"
        )

        
class HindexRestriction(Restriction):
    """
    Accepts sequences whose hydrophobic index falls within a specified range.

    The hydrophobic index is computed by sequence_geometry.compute_hi over a
    Sequence built from the candidate, using the per-residue hydrophobicity
    values of the selected scale.

    Parameters
    ----------
    min : float | None
        Minimum accepted hydrophobic index (inclusive). None means no lower bound.
    max : float | None
        Maximum accepted hydrophobic index (inclusive). None means no upper bound.
    h_scale : str
        Name of the hydrophobicity scale used for the per-residue values.
    """

    def __init__(
        self,
        min: Optional[float] = None,
        max: Optional[float] = None,
        h_scale: str = 'eisenberg',
        average: bool = False,
    ) -> None:
        super().__init__()

        if min is None and max is None:
            raise ValueError(
                "HindexRestriction requires at least one of "
                "'min' or 'max'."
            )
        self._min: Optional[float] = min
        self._max: Optional[float] = max

        if (
            self._min is not None
            and self._max is not None
            and self._min > self._max
        ):
            raise ValueError(
                f"HindexRestriction: min ({self._min}) must be "
                f"<= max ({self._max})."
            )

        self.h_scale = h_scale
        self.average = average

        # Import here to avoid circular dependency at module level.
        from .scales import Scales
        self._hi_table: dict[str, float] = Scales.hydrophobicity_scales[self.h_scale]

    def test(self, seq: str, verbose=False) -> bool:
        """Return True if the computed hydrophobic index is within [min, max]."""
        from .sequence_geometry import compute_hi
        test_seq = Sequence(seq, h_scale=self.h_scale,)
        hindex = compute_hi(test_seq, average=self.average)
        above_min = self._min is None or hindex >= self._min
        below_max = self._max is None or hindex <= self._max

        passed = above_min and below_max

        if verbose:
            bound_str = f"[{self._min}, {self._max}]"
            print(f"Hi restriction: {bound_str} Current: {hindex} = Pass: {passed}")
        
        return passed

    def __repr__(self) -> str:
        return (
            f"HindexRestriction(min={self._min}, "
            f"max={self._max})"
        )

        
class HmomentRestriction(Restriction):
    """
    Accepts sequences whose hydrophobic moment falls within a specified range.

    The hydrophobic moment is computed as described in
    Faraday Symp. Chem. Soc., 1982, 17, 109-120, via
    sequence_geometry.compute_hm_scalar over the helix positions of the
    candidate sequence.

    Parameters
    ----------
    min : float | None
        Minimum accepted hydrophobic moment (inclusive). None means no lower bound.
    max : float | None
        Maximum accepted hydrophobic moment (inclusive). None means no upper bound.
    h_scale : str
        Name of the hydrophobicity scale used for the per-residue values.
    """

    def __init__(
        self,
        min: Optional[float] = None,
        max: Optional[float] = None,
        h_scale: str = 'eisenberg',
        average: bool = False,
    ) -> None:
        super().__init__()

        if min is None and max is None:
            raise ValueError(
                "HmomentRestriction requires at least one of "
                "'min' or 'max'."
            )
        self._min: Optional[float] = min
        self._max: Optional[float] = max

        if (
            self._min is not None
            and self._max is not None
            and self._min > self._max
        ):
            raise ValueError(
                f"HmomentRestriction: min ({self._min}) must be "
                f"<= max ({self._max})."
            )

        self.h_scale = h_scale
        self.average = average

        # Import here to avoid circular dependency at module level.
        from .scales import Scales
        self._hi_table: dict[str, float] = Scales.hydrophobicity_scales[self.h_scale]

    def test(self, seq: str, verbose=False) -> bool:
        """Return True if the computed hydrophobic moment is within [min, max]."""
        from .sequence_geometry import compute_helix_positions, compute_hm_scalar
        seq = self._as_sequence(seq, h_scale=self.h_scale)
        
        positions = compute_helix_positions(seq, translate=False)
        hm_scalar = compute_hm_scalar(seq, positions, average=self.average)

        above_min = self._min is None or hm_scalar >= self._min
        below_max = self._max is None or hm_scalar <= self._max

        passed = above_min and below_max

        if verbose:
            bound_str = f"[{self._min}, {self._max}]"
            print(f"Hm restriction: {bound_str} Current: {hm_scalar} = Pass: {passed}")

        return passed

    def __repr__(self) -> str:
        return (
            f"HmomentRestriction(min={self._min}, "
            f"max={self._max})"
        )


class ForbiddenSequence(Restriction):
    """
    Rejects sequences that exactly match any sequence in a forbidden set.

    Unlike PatternRestriction, which matches substrings or regex patterns,
    this restriction compares the whole candidate against each forbidden
    entry. It is intended for user-supplied excluded_sequences: specific
    peptides that must never appear during evolution.

    Matching is case-insensitive and based on full-string equality.

    Parameters
    ----------
    sequences : list[str]
        Forbidden sequences. Each candidate equal to one of these (ignoring
        case) is rejected.
    """

    def __init__(self, sequences: list[str]) -> None:
        super().__init__()
        # Store as an uppercase set for O(1) membership tests.
        self._forbidden: frozenset[str] = frozenset(
            str(s).upper() for s in sequences
        )

    def test(self, seq: str, verbose=False) -> bool:
        """Return True only if the full candidate is not in the forbidden set."""
        candidate = str(seq).upper()
        passed = candidate not in self._forbidden

        if passed:
            message = f"'{seq}' is not in the forbidden set"
        else:
            message = f"'{seq}' is a forbidden sequence"

        self.message = message
        if verbose:
            print(f"{message}. Pass: {passed}")
        return passed

    def __repr__(self) -> str:
        return f"ForbiddenSequence(n={len(self._forbidden)})"

if __name__ == '__main__':
    pass
