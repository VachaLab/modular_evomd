# === restriction.py ===
from __future__ import annotations
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

    def test(self, seq: str) -> bool:
        """
        Evaluates whether seq satisfies this restriction.

        Returns True if the sequence is accepted, False if rejected.
        Updates self.message with a description of the outcome.

        Raises NotImplementedError if not overridden by a subclass.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement test()."
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# Built-in restrictions
# ---------------------------------------------------------------------------

class LengthRestriction(Restriction):
    """
    Accepts sequences whose length falls within a specified range.

    Accepts either a fixed length or a min/max range. When a fixed length
    is provided, only sequences of exactly that length are accepted.
    When a range is provided, sequences whose length falls within
    [min_len, max_len] (inclusive on both ends) are accepted.

    Parameters
    ----------
    length : int | None
        Fixed required length. When provided, min_len and max_len are
        ignored.
    min_len : int | None
        Minimum accepted length (inclusive). Used only when length is None.
    max_len : int | None
        Maximum accepted length (inclusive). Used only when length is None.
    """

    def __init__(
        self,
        length: Optional[int] = None,
        min_len: Optional[int] = None,
        max_len: Optional[int] = None,
    ) -> None:
        super().__init__()

        if length is not None:
            # Fixed length mode: both min and max are set to the same value.
            self._min: int = length
            self._max: int = length
        elif min_len is not None or max_len is not None:
            # Range mode: either bound may be open-ended.
            self._min = min_len if min_len is not None else 0
            self._max = max_len if max_len is not None else int(1e9)
        else:
            raise ValueError(
                "LengthRestriction requires either 'length' or at least one "
                "of 'min_len' / 'max_len'."
            )

        if self._min > self._max:
            raise ValueError(
                f"LengthRestriction: min_len ({self._min}) must be <= "
                f"max_len ({self._max})."
            )

    def test(self, seq: str) -> bool:
        n = len(seq)
        passed = self._min <= n <= self._max
        if passed:
            self.message = f"length {n} is within [{self._min}, {self._max}]"
        else:
            self.message = (
                f"length {n} is outside [{self._min}, {self._max}]"
            )
        return passed

    def __repr__(self) -> str:
        if self._min == self._max:
            return f"LengthRestriction(length={self._min})"
        return f"LengthRestriction(min_len={self._min}, max_len={self._max})"


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
    min_count : int
        Minimum number of residues from the set required. Default 0.
    max_count : int | None
        Maximum number of residues from the set allowed. None means no
        upper bound.
    """

    def __init__(
        self,
        residues: str,
        min_count: int = 0,
        max_count: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._residues: frozenset[str] = frozenset(residues.upper())
        self._min: int = min_count
        self._max: Optional[int] = max_count

        if self._max is not None and self._min > self._max:
            raise ValueError(
                f"CompositionRestriction: min_count ({self._min}) must be "
                f"<= max_count ({self._max})."
            )

    def test(self, seq: str) -> bool:
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
            self.message = (
                f"count of {{''.join(sorted(self._residues))}} is {count}, "
                f"within {bound_str}"
            )
        else:
            self.message = (
                f"count of {''.join(sorted(self._residues))} is {count}, "
                f"outside {bound_str}"
            )
        return passed

    def __repr__(self) -> str:
        return (
            f"CompositionRestriction(residues='{''.join(sorted(self._residues))}', "
            f"min_count={self._min}, max_count={self._max})"
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

    def test(self, seq: str) -> bool:
        seq_upper = seq.upper()

        # Check plain substrings first.
        for pattern in self._plain:
            if pattern in seq_upper:
                self.message = f"forbidden substring '{pattern}' found in '{seq}'"
                return False

        # Check regex patterns.
        for rx in self._regex:
            match = rx.search(seq_upper)
            if match:
                self.message = (
                    f"forbidden pattern '{rx.pattern}' matched at "
                    f"position {match.start()} in '{seq}'"
                )
                return False

        self.message = "no forbidden patterns found"
        return True

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
    min_charge : float | None
        Minimum accepted net charge (inclusive). None means no lower bound.
    max_charge : float | None
        Maximum accepted net charge (inclusive). None means no upper bound.
    """

    def __init__(
        self,
        min_charge: Optional[float] = None,
        max_charge: Optional[float] = None,
    ) -> None:
        super().__init__()

        if min_charge is None and max_charge is None:
            raise ValueError(
                "ChargeRestriction requires at least one of "
                "'min_charge' or 'max_charge'."
            )
        self._min: Optional[float] = min_charge
        self._max: Optional[float] = max_charge

        if (
            self._min is not None
            and self._max is not None
            and self._min > self._max
        ):
            raise ValueError(
                f"ChargeRestriction: min_charge ({self._min}) must be "
                f"<= max_charge ({self._max})."
            )

        # Import here to avoid circular dependency at module level.
        from scales import Scales
        self._charge_table: dict[str, float] = Scales.aa_charges

    def test(self, seq: str) -> bool:
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
        bound_str = f"[{self._min}, {self._max}]"
        if passed:
            self.message = f"charge {charge} is within {bound_str}"
        else:
            self.message = f"charge {charge} is outside {bound_str}"
        return passed

    def __repr__(self) -> str:
        return (
            f"ChargeRestriction(min_charge={self._min}, "
            f"max_charge={self._max})"
        )

        