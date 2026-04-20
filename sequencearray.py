# === sequencearray.py ===
import logging
from collections.abc import MutableSequence
from typing import Any, List, Optional
from sequence import Sequence

logger = logging.getLogger(__name__)


class SequenceArray(MutableSequence):
    """
    A validated list-like container for Sequence objects.
    Used by Evolver to manage peptide populations.
    Only accepts Sequence instances; raises TypeError otherwise.
    """

    def __init__(self, sequences: Optional[List[Sequence]] = None) -> None:
        invalid = [s for s in sequences if not isinstance(s, Sequence)] if sequences else []
        if invalid:
            raise TypeError(f"All elements must be Sequence instances. Invalid: {invalid}")
        self._sequences: List[Sequence] = list(sequences) if sequences else []

    # Abstract method implementations ---------------------------------------

    def __getitem__(self, index):
        return self._sequences[index]

    def __setitem__(self, index, value: Sequence) -> None:
        if not isinstance(value, Sequence):
            raise TypeError(f"Expected Sequence instance, got {type(value).__name__}")
        self._sequences[index] = value

    def __delitem__(self, index) -> None:
        del self._sequences[index]

    def __len__(self) -> int:
        return len(self._sequences)

    def insert(self, index: int, value: Sequence) -> None:
        if not isinstance(value, Sequence):
            raise TypeError(f"Expected Sequence instance, got {type(value).__name__}")
        self._sequences.insert(index, value)

    # Overridden MutableSequence methods ------------------------------------

    def append(self, value: Sequence) -> None:
        """Appends a Sequence to the array. Raises TypeError if value is not a Sequence."""
        if not isinstance(value, Sequence):
            raise TypeError(f"Expected Sequence instance, got {type(value).__name__}")
        self._sequences.append(value)

    # String representation -------------------------------------------------

    def __str__(self) -> str:
        return f"SequenceArray({[str(s) for s in self._sequences]})"

    def __repr__(self) -> str:
        return f"SequenceArray(n={len(self._sequences)})"

    # Population utilities --------------------------------------------------

    def _to_str(self, seq: "str | Sequence") -> str:
        """Normalizes a sequence identifier to an uppercase string."""
        return str(seq).upper()

    def exists(self, seq: str) -> bool:
        """
        Returns True if a sequence string is already present in the array.
        Comparison is case-insensitive and normalized to uppercase.
        """
        seq = self._to_str(seq)
        seq = seq.upper()
        return any(str(s) == seq for s in self._sequences)

    def extract(self, seq: str) -> Sequence:
        """
        Finds a Sequence by its string representation, removes it from the array,
        and returns it. Only the first match is extracted.
        Raises ValueError if the sequence is not found.
        """
        seq = self._to_str(seq)
        seq = seq.upper()
        for i, s in enumerate(self._sequences):
            if str(s) == seq:
                return self._sequences.pop(i)
        raise ValueError(f"Sequence '{seq}' not found in SequenceArray.")

    def filter_by(self, attribute: str, value: Any) -> "SequenceArray":
        """
        Returns a new SequenceArray containing only sequences where
        the given attribute matches the specified value.
        Raises AttributeError if the attribute does not exist on Sequence.
        """
        if self._sequences and not hasattr(self._sequences[0], attribute):
            raise AttributeError(f"Sequence has no attribute '{attribute}'.")
        filtered = [s for s in self._sequences if getattr(s, attribute) == value]
        return SequenceArray(sequences=filtered)

    def sort_by_fitness(self, reverse: bool = True) -> None:
        """
        Sorts the array in-place by mean fitness.
        Sequences without fitness values (None) are always placed at the end.
        reverse=True sorts from highest to lowest fitness (maximize).
        reverse=False sorts from lowest to highest fitness (minimize).
        """
        def sort_key(s: Sequence):
            f = s.fitness
            if f is None:
                # Places None values at the end regardless of sort direction
                return float('-inf') if reverse else float('inf')
            return f

        self._sequences.sort(key=sort_key, reverse=reverse)


if __name__ == '__main__':
    pass
    