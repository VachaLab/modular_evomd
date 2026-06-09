# === flanks.py ===
from __future__ import annotations
import logging
import random
from typing import TYPE_CHECKING

from genmethod import GenMethod
from sequence import Sequence

if TYPE_CHECKING:
    from generator import Generator

logger = logging.getLogger(__name__)


class Flanks(GenMethod):
    """
    Wrapper method that holds two constant, immutable flanking fragments and
    restricts all evolution to the central core of the peptide.

    The full peptide is always laid out as:

        n_flank + core + c_flank

    with len(core) == peptide_len - len(n_flank) - len(c_flank). The flanks
    are fixed for the whole run; only the core is generated or modified.

    Modes (selected by parent availability):

    - No parent (seq1 is None): a random core is drawn from the amino acid
      pool and the flanks are attached. This makes Flanks usable as the
      Generator initial method.
    - With parents (seq1 given, seq2 optional): the flanks are stripped from
      both parents, each core is wrapped as a Sequence, and one of the
      wrapped inner methods (e.g. Swap, Hybrid, FacesMix) is applied to the
      cores. The flanks are then re-attached to the produced core.

    Flanks acts as a wrapper: it does not reimplement swap/hybrid/faces, it
    delegates to GenMethod instances passed at construction. Because the
    inner methods receive only the core wrapped as a Sequence, any helix
    geometry they use (Swap, FacesMix) is computed over the isolated core,
    not over the full peptide. Purely linear methods (Hybrid) are unaffected.

    Because expected_parents is 0, the method is eligible as the Generator
    initial method and is skipped by the Generator extra-mutation step, so
    the flanks are never touched by that mutation.

    Parameters
    ----------
    n_flank : str | None
        Constant N-terminal fragment. None or '' means no N-flank.
    c_flank : str | None
        Constant C-terminal fragment. None or '' means no C-flank.
    methods : list[GenMethod] | GenMethod | None
        Inner generation methods applied to the core when parents are
        available. Defaults to an empty list (core-only random/mutation).
    weights : list[float] | None
        Selection weights for the inner methods. Must match methods length.
        Uniform by default.
    """
    method_name: str = 'FLANKS'
    expected_parents: int = 0

    def __init__(
        self,
        n_flank: str | None = None,
        c_flank: str | None = None,
        methods: list[GenMethod] | GenMethod | None = None,
        weights: list[float] | None = None,
        extra_mutation: bool = False,
        extra_mutation_prob: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_flank: str = (n_flank or '').upper()
        self.c_flank: str = (c_flank or '').upper()

        # Normalize inner methods into a list.
        if methods is None:
            methods = []
        elif isinstance(methods, GenMethod):
            methods = [methods]
        self.methods: list[GenMethod] = methods

        # Normalize weights to match the inner methods.
        if weights is None:
            weights = [1.0] * len(self.methods)
        if len(weights) != len(self.methods):
            raise ValueError(
                f"Flanks: number of weights ({len(weights)}) must match "
                f"number of inner methods ({len(self.methods)})."
            )
        self.weights: list[float] = weights

        # Extra mutation configuration.
        self.extra_mutation: bool = extra_mutation
        if not 0.0 <= extra_mutation_prob <= 1.0:
            raise ValueError("extra_mutation_prob must be between 0.0 and 1.0.")
        self.extra_mutation_prob: float = extra_mutation_prob

    # Generator injection -----------------------------------------------------

    def _register_inner(self) -> None:
        """
        Injects the Generator back-reference into every inner method so they
        can access aa_pool / peptide_len. Called lazily on first generate(),
        since self.generator is only available after Flanks itself is
        registered by the Generator.
        """
        for method in self.methods:
            method.generator = self.generator

    # Length helpers ----------------------------------------------------------

    def _peptide_len(self) -> int:
        """Resolves the configured full-peptide length from the Generator."""
        if self.generator is None:
            raise RuntimeError(
                "Flanks requires a Generator reference to resolve peptide_len."
            )
        length = self.generator.peptide_len
        if isinstance(length, (tuple, list)):
            # Variable length: draw once for this call.
            length = random.randint(length[0], length[1])
        if length is None:
            raise ValueError("Flanks requires Generator.peptide_len to be set.")
        return length

    def _core_len(self, peptide_len: int) -> int:
        """Computes and validates the core length for a given peptide length."""
        core_len = peptide_len - len(self.n_flank) - len(self.c_flank)
        if core_len <= 0:
            raise ValueError(
                f"Flanks: flanks ('{self.n_flank}' + '{self.c_flank}', "
                f"total {len(self.n_flank) + len(self.c_flank)}) leave no room "
                f"for a core in a peptide of length {peptide_len}."
            )
        return core_len

    def _strip_flanks(self, seq: GenMethod | Sequence | str) -> str:
        """Returns the core substring of a full-length parent."""
        s = str(seq)
        start = len(self.n_flank)
        end = len(s) - len(self.c_flank)
        return s[start:end]

    def _attach_flanks(self, core: str) -> str:
        """Re-attaches the constant flanks around a core string."""
        return self.n_flank + core + self.c_flank

    # Core builders -----------------------------------------------------------

    def _random_core(self, core_len: int) -> str:
        """Builds a random core of the required length from the pool."""
        pool = self.generator.aa_pool
        return ''.join(random.choices(pool, k=core_len))

    def _mutate_core(self, core: str, verbose: bool = True) -> str:
        """Single-position random mutation of the core (fallback path)."""
        if random.random() < self.extra_mutation_prob:
            pool = self.generator.aa_pool
            idx = random.randint(0, len(core) - 1)
            new_aa = random.choice(pool)
            if verbose:
                print("Inner point mutation on core")
                print(f"{core}")
                print(f"{' '*idx}{new_aa}{' '*(len(core) - idx - 1)} <- mutation")
            return core[:idx] + new_aa + core[idx + 1:]
        return core

    def _select_inner(self, available_parents: int):
        """
        Selects one inner method whose expected_parents requirement is met
        by the number of available parents. Returns None when no inner method
        is eligible.
        """
        eligible = [
            (m, w) for m, w in zip(self.methods, self.weights)
            if getattr(m, 'expected_parents', 2) <= available_parents
        ]
        if not eligible:
            return None
        inner_methods, inner_weights = zip(*eligible)
        return random.choices(inner_methods, weights=inner_weights, k=1)[0]

    # Generation --------------------------------------------------------------

    def generate(self, seq1: Sequence = None, seq2: Sequence = None, verbose=False) -> str:
        if verbose:
            ornament = int((30 - len(self.method_name)) / 2)
            print(f"{'-' * ornament} {self.method_name} {'-' * ornament}")
            
        # Ensure inner methods can see the Generator (lazy, idempotent).
        self._register_inner()

        peptide_len = self._peptide_len()
        core_len = self._core_len(peptide_len)

        # No-parent mode: random core + flanks.
        if seq1 is None and seq2 is None:
            core = self._random_core(core_len)
            result = self._attach_flanks(core)
            if verbose:
                print(f"n_flank='{self.n_flank}'  c_flank='{self.c_flank}'  mode=RADOM")
                print(f"{result} <- Child")
            return result

        # With-parent mode: strip flanks, operate on cores, re-attach.
        available_parents = sum(p is not None for p in (seq1, seq2))
        core1 = self._strip_flanks(seq1)
        core2 = self._strip_flanks(seq2) if seq2 is not None else None

        inner = self._select_inner(available_parents)
        
        if verbose:
            print(f"n_flank='{self.n_flank}'  c_flank='{self.c_flank}'  mode={inner.method_name}")
        
        core_seq1 = Sequence(core1)
        core_seq2 = Sequence(core2)
        if verbose:
            print(f"{core1} <- Core parent 1")
            print(f"{core2} <- Core parent 2")

        new_core = inner.generate(core_seq1, core_seq2, verbose=False)
        if self.extra_mutation:
            new_core = self._mutate_core(new_core, verbose=verbose)
        method_label = inner.method_name

        # Defensive: inner methods preserve length, but guard against drift.
        if len(new_core) != core_len:
            logger.warning(
                f"Flanks: inner method '{method_label}' returned a core of "
                f"length {len(new_core)} (expected {core_len}); using it as-is."
            )

        result = self._attach_flanks(new_core)
        if verbose:
            print(f"{new_core} <- new core")
            print(f"{result} <- Child")
            print(f"{'-' * 30}")
        return result

    def __repr__(self) -> str:
        return (
            f"Flanks(n_flank='{self.n_flank}', c_flank='{self.c_flank}', "
            f"methods={self.methods})"
        )


if __name__ == '__main__':
    pass
