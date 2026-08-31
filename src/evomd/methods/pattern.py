# === pattern.py ===
from __future__ import annotations
import logging
import random
from typing import TYPE_CHECKING

from .genmethod import GenMethod

if TYPE_CHECKING:
    from ..core.sequence import Sequence

logger = logging.getLogger(__name__)


class Pattern(GenMethod):
    """
    Generates a sequence from a fixed template pattern, varying only the
    positions marked with the free-position character ('-').

    This method has two modes, selected automatically by whether a parent
    is supplied:

    - No parent (seq1 is None): the full sequence is built from scratch.
      Fixed positions in the pattern are copied verbatim and every free
      position is drawn from its own option list. This is the first-fill
      behaviour and is unaffected by max_mutations.
    - With a parent (seq1 is given): the parent sequence is taken as the
      starting point and between 1 and max_mutations distinct free positions
      are mutated, each drawing its new residue from its own option list.
      All other positions (fixed and the untouched free ones) keep the
      parent's residues.

    Because expected_parents is 0, the method is eligible both as the
    Generator initial method (called with no parents) and as a regular
    working method (called with one parent).

    The options are given as a list of lists, one inner list per free
    position, in left-to-right order:

        pattern  = 'AY---L-V---KK'
        options  = [['A','M'], ['K'], ['R','D'], ['?'],
                    ['S','T'], ['C','M'], ['?'], ['E','D']]

    The special token '?' expands to all residues in the amino acid pool
    (Instructor.mut_aa, exposed here through Generator.aa_pool).

    `options` is optional. When it is omitted, shorter than the number of
    free positions, or contains an empty inner list, the affected positions
    default to the full amino acid pool (equivalent to '?'). This pool
    resolution is deferred to generation time, so it always reflects the
    current Generator.aa_pool.

    Optional per-option weights bias the random choice at each free
    position. When omitted, all options at a position are equally likely.

    Parameters
    ----------
    pattern : str
        Template string. The character given by `free_char` (default '-')
        marks positions to be filled; all other characters are fixed.
    options : list[list[str]] | None
        One list of candidate residues per free position. Use ['?'] (or an
        empty list, or simply provide fewer lists than free positions) to
        allow the full amino acid pool at that position. Defaults to the
        full pool at every free position when omitted.
    weights : list[list[float]] | None
        Optional selection weights matching the shape of `options`. When a
        position uses the full pool ('?', empty, or missing), a weights entry
        for it must either be omitted (None) or match the expanded pool length.
    max_mutations : int | None
        Upper bound on how many free positions are mutated per call in the
        with-parent mode. The actual count is drawn uniformly from
        1..max_mutations on each call. Values are clamped to the number of
        free positions; None or values <= 0 fall back to 1. Has no effect on
        the no-parent (template) mode. Defaults to 1.
    free_char : str
        Character marking free positions. Defaults to '-'.
    """
    method_name: str = 'PATTERN'
    expected_parents: int = 0

    def __init__(
        self,
        pattern: str,
        options: list[list[str]] | None = None,
        weights: list[list[float]] | None = None,
        max_mutations: int | None = 1,
        free_char: str = '-',
    ) -> None:
        super().__init__()
        self.pattern: str = pattern
        self.free_char: str = free_char
        self.free_indices: list[int] = [
            i for i, c in enumerate(pattern) if c == free_char
        ]

        # Options default to the full amino acid pool when missing or empty.
        # An empty/short `options` is padded with the '?' token, and any empty
        # inner list is replaced by ['?'], deferring pool resolution to
        # generation time (so it reflects the current Generator.aa_pool).
        if options is None:
            options = []
        if len(options) > len(self.free_indices):
            raise ValueError(
                f"Pattern: number of option lists ({len(options)}) exceeds "
                f"number of free positions ({len(self.free_indices)}) in the "
                f"pattern '{pattern}'."
            )
        self.options: list[list[str]] = [
            (options[n] if n < len(options) and options[n] else ['?'])
            for n in range(len(self.free_indices))
        ]

        # Normalize weights to one entry per free position (None = uniform).
        if weights is None:
            weights = []
        if len(weights) > len(self.free_indices):
            raise ValueError(
                f"Pattern: number of weight lists ({len(weights)}) exceeds "
                f"number of free positions ({len(self.free_indices)})."
            )
        self.weights: list[list[float] | None] = [
            (weights[n] if n < len(weights) and weights[n] else None)
            for n in range(len(self.free_indices))
        ]

        # Upper bound for mutations per call (with-parent mode only).
        # Clamp to the available free positions; None/<=0 falls back to 1.
        if max_mutations is None or max_mutations <= 0:
            max_mutations = 1
        self.max_mutations: int = min(max_mutations, max(1, len(self.free_indices)))

    def _resolve_options(self, raw_options: list[str]) -> list[str]:
        """
        Expands the '?' token into the full amino acid pool. A list that is
        exactly ['?'] becomes the whole pool; otherwise options are returned
        as given. The pool is read from the Generator back-reference so that
        it always reflects the current Instructor.mut_aa.
        """
        if raw_options == ['?']:
            if self.generator is None:
                raise RuntimeError(
                    "Pattern: '?' option requires a Generator reference to "
                    "resolve the amino acid pool, but none was injected."
                )
            return list(self.generator.aa_pool)
        return raw_options

    def _choose_residue(self, n: int) -> str:
        """
        Draws one residue for the n-th free position, expanding '?' to the
        pool and applying optional per-position weights. Guards against a
        weights/options length mismatch after '?' expansion.
        """
        choices = self._resolve_options(self.options[n])
        weights = self.weights[n]
        if weights is not None and len(weights) != len(choices):
            raise ValueError(
                f"Pattern: weights at free position {n} have length "
                f"{len(weights)} but there are {len(choices)} options."
            )
        return random.choices(choices, weights=weights, k=1)[0]

    def _generate_from_template(self, verbose=False) -> str:
        """Builds a full sequence from the pattern (no-parent mode)."""
        child: list[str] = list(self.pattern)
        for n, idx in enumerate(self.free_indices):
            child[idx] = self._choose_residue(n)
        result = ''.join(child)

        if verbose:
            ornament = int((30 - len(self.method_name)) / 2)
            print(f"{'-' * ornament} {self.method_name} {'-' * ornament}")
            print(f"{self.pattern} <- Pattern (template mode)")
            marker = ''.join('^' if i in self.free_indices else ' '
                             for i in range(len(self.pattern)))
            print(f"{marker} <- Free positions")
            print(f"{result} <- Child")
            print(f"{'-' * 30}")
        return result

    def _mutate_parent(self, seq1: Sequence, verbose=False) -> str:
        """
        Mutates between 1 and self.max_mutations distinct random free
        positions of the parent, each drawing its new residue from that
        position's option list. All other positions keep the parent's
        residues. Only positions in self.free_indices are eligible.
        """
        parent_str = str(seq1)

        if not self.free_indices:
            # No free positions to mutate; return the parent unchanged.
            logger.debug(
                "Pattern: parent received but pattern has no free positions; "
                "returning parent unchanged."
            )
            return parent_str

        child: list[str] = list(parent_str)

        # Draw how many positions to mutate this call, then pick that many
        # distinct free-position slots (so each uses its own option list).
        n_mut = random.randint(1, self.max_mutations)
        chosen_slots = random.sample(range(len(self.free_indices)), k=n_mut)

        mutated_indices = []
        for n in chosen_slots:
            idx = self.free_indices[n]
            child[idx] = self._choose_residue(n)
            mutated_indices.append(idx)

        result = ''.join(child)

        if verbose:
            ornament = int((30 - len(self.method_name)) / 2)
            print(f"{'-' * ornament} {self.method_name} {'-' * ornament}")
            print(f"{parent_str} <- Parent (mutation mode)")
            marker = ''.join('^' if i in mutated_indices else ' '
                             for i in range(len(parent_str)))
            print(f"{marker} <- Mutated positions ({n_mut})")
            print(f"{result} <- Child")
            print(f"{'-' * 30}")
        return result

    def generate(self, seq1: Sequence = None, seq2: Sequence = None, verbose=False) -> str:
        # Mode is selected by parent availability. seq2 is always ignored.
        if seq1 is None:
            return self._generate_from_template(verbose=verbose)
        return self._mutate_parent(seq1, verbose=verbose)

    def __repr__(self) -> str:
        return (
            f"Pattern(pattern='{self.pattern}', "
            f"free_positions={len(self.free_indices)}, "
            f"max_mutations={self.max_mutations})"
        )


if __name__ == '__main__':
    pass
    