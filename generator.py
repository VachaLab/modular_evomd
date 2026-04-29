# === generator.py ===
from __future__ import annotations
import logging
import random
from typing import TYPE_CHECKING, Optional

from genmethod import GenMethod
from scales import Scales

if TYPE_CHECKING:
    from sequence import Sequence
    from restriction import Restriction

logger = logging.getLogger(__name__)

# Default amino acid pool: the 20 standard proteinogenic amino acids.
_DEFAULT_AA_POOL: str = ''.join(sorted(Scales.aa_charges.keys()))


class _RandomInitial(GenMethod):
    """
    Built-in method for generating sequences from scratch during the first
    population fill.

    Both parents are ignored. A full-length sequence is built by drawing
    residues uniformly at random from the Generator amino acid pool.
    The target length is resolved from Generator.peptide_len on each call.
    """

    def generate(self, seq1: Sequence, seq2: Sequence) -> str:
        pool = self.generator.aa_pool
        length = self.generator.peptide_len

        # Resolve target length from the generator configuration.
        if isinstance(length, tuple):
            length = random.randint(length[0], length[1])
        if length is None:
            raise ValueError(
                "_RandomInitial requires Generator.peptide_len to be set."
            )

        return ''.join(random.choices(pool, k=length))


class _RandomMutation(GenMethod):
    """
    Built-in single-position point mutation method.

    Used exclusively as the extra mutation step applied after the main
    generation method. Takes seq1, selects one position at random, and
    replaces it with a residue drawn from the Generator amino acid pool.
    seq2 is always ignored.

    Always mutates exactly one position. Never rebuilds the sequence
    from scratch.
    """

    def generate(self, seq1: Sequence, seq2: Sequence) -> str:
        pool = self.generator.aa_pool
        seq_str = str(seq1)
        length = len(seq_str)

        if length == 0:
            raise ValueError("_RandomMutation received an empty sequence.")

        # Replace exactly one randomly chosen position.
        idx = random.randint(0, length - 1)
        return seq_str[:idx] + random.choice(pool) + seq_str[idx + 1:]


class Generator:
    """
    Orchestrates the creation of candidate sequences for Evolver.

    Generator holds one or more GenMethod instances and selects among them
    on each generation call. It enforces sequence validity through an optional
    list of Restriction objects and applies an optional extra point mutation
    after the main generation step.

    Resources shared with GenMethod instances (aa_pool, peptide_len) are
    accessed by those instances through a back-reference to this Generator,
    injected automatically when a method is registered.

    Parameters
    ----------
    methods : list[GenMethod] | GenMethod | None
        One or more generation methods used during evolution. When multiple
        methods are provided, one is selected randomly on each call according
        to the corresponding weights. Defaults to _RandomInitial if not provided.
    weights : list[float] | None
        Selection weights for each method in methods. Must match the length
        of methods. All weights default to 1.0 (uniform selection).
    initial_method : GenMethod | None
        Method used exclusively for the first population. Defaults to
        _RandomInitial when not provided.
    aa_pool : str | None
        String of single-letter amino acid codes available for generation.
        Defaults to all 20 standard proteinogenic amino acids.
    peptide_len : int | tuple[int, int] | None
        Target sequence length. An integer fixes the length. A tuple
        (min, max) allows variable-length sequences drawn uniformly from
        that range. None disables length control entirely.
    extra_mutation : bool
        Whether to apply a random single-position mutation after the main
        generation step.
    extra_mutation_prob : float
        Probability of applying the extra mutation on any given call.
        Only relevant when extra_mutation=True. Must be in [0.0, 1.0].
    restrictions : list[Restriction] | None
        Sequence validity constraints tested inside the generation loop.
        A candidate is accepted only when all restrictions pass.
    max_attempts : int
        Maximum number of generation attempts per call before raising
        a RuntimeError. Prevents infinite loops under very strict restrictions.
    """

    def __init__(
        self,
        methods: list[GenMethod] | GenMethod | None = None,
        weights: list[float] | None = None,
        initial_method: GenMethod | None = None,
        aa_pool: str | None = None,
        peptide_len: int | tuple[int, int] | None = None,
        extra_mutation: bool = False,
        extra_mutation_prob: float = 0.2,
        restrictions: list[Restriction] | None = None,
        max_attempts: int = 10000,
    ) -> None:

        # Amino acid pool shared with all registered GenMethod instances.
        self.aa_pool: str = aa_pool if aa_pool is not None else _DEFAULT_AA_POOL

        # Target sequence length: int for fixed, tuple for variable range.
        self.peptide_len: int | tuple[int, int] | None = peptide_len

        # Validate and normalize methods into a list.
        if methods is None:
            methods = [_RandomInitial()]
        elif isinstance(methods, GenMethod):
            methods = [methods]
        self.methods: list[GenMethod] = methods

        # Validate and normalize weights.
        if weights is None:
            weights = [1.0] * len(self.methods)
        if len(weights) != len(self.methods):
            raise ValueError(
                f"Length of weights ({len(weights)}) must match "
                f"length of methods ({len(self.methods)})."
            )
        self.weights: list[float] = weights

        # Method used for the first population only.
        self.initial_method: GenMethod = (
            initial_method if initial_method is not None else _RandomInitial()
        )

        # Extra mutation configuration.
        self.extra_mutation: bool = extra_mutation
        if not 0.0 <= extra_mutation_prob <= 1.0:
            raise ValueError("extra_mutation_prob must be between 0.0 and 1.0.")
        self.extra_mutation_prob: float = extra_mutation_prob

        # Built-in point mutator reused exclusively for the extra mutation step.
        self._point_mutator: _RandomMutation = _RandomMutation()

        # Sequence validity restrictions evaluated inside the generation loop.
        self.restrictions: list[Restriction] = restrictions if restrictions is not None else []

        # Maximum loop iterations before raising an error.
        self.max_attempts: int = max_attempts

        # Inject generator reference into all registered methods.
        self._register_method(self.initial_method)
        for method in self.methods:
            self._register_method(method)
        self._register_method(self._point_mutator)

    # Registration ----------------------------------------------------------

    def _register_method(self, method: GenMethod) -> None:
        """Injects a back-reference to this Generator into a GenMethod instance."""
        method.generator = self

    def add_method(
        self,
        method: GenMethod,
        weight: float = 1.0,
    ) -> None:
        """
        Registers an additional generation method after construction.
        Injects the Generator reference automatically.
        """
        self._register_method(method)
        self.methods.append(method)
        self.weights.append(weight)
        logger.info(f"Generator: method {method} added with weight {weight}.")

    def add_restriction(self, restriction: Restriction) -> None:
        """Registers an additional Restriction after construction."""
        self.restrictions.append(restriction)
        logger.info(f"Generator: restriction {restriction} added.")

    # Length resolution -----------------------------------------------------

    def _resolve_length(self) -> int | None:
        """
        Returns the target sequence length for the current call.
        Draws uniformly from the range when peptide_len is a tuple.
        Returns None when no length constraint is configured.
        """
        if self.peptide_len is None:
            return None
        if isinstance(self.peptide_len, tuple):
            return random.randint(self.peptide_len[0], self.peptide_len[1])
        return self.peptide_len

    # Validation ------------------------------------------------------------

    def _passes_restrictions(self, seq: str) -> bool:
        """
        Returns True when seq satisfies all registered Restriction instances.
        Logs the failure reason at DEBUG level when a restriction is not met.
        """
        for restriction in self.restrictions:
            if not restriction.test(seq):
                logger.debug(
                    f"Generator: sequence '{seq}' rejected by "
                    f"{restriction.__class__.__name__}: {restriction.message}"
                )
                return False
        return True

    # Core generation -------------------------------------------------------

    def _select_method(self) -> GenMethod:
        """Selects one GenMethod from self.methods according to self.weights."""
        return random.choices(self.methods, weights=self.weights, k=1)[0]

    def _apply_extra_mutation(self, seq: str) -> str:
        """
        Applies a single-position random mutation with probability
        extra_mutation_prob. Returns the sequence unchanged if the
        mutation is not applied.

        Passes the candidate string wrapped as a minimal object to
        _RandomMutation, which only reads str(seq1) and len(seq1).
        """
        if random.random() < self.extra_mutation_prob:
            mutated = self._point_mutator.generate(
                _StrAdapter(seq),
                _StrAdapter(seq),
            )
            logger.debug(
                f"Generator: extra mutation applied: '{seq}' -> '{mutated}'"
            )
            return mutated
        return seq

    def generate(self, seq1: Sequence, seq2: Sequence) -> str:
        """
        Produces a valid candidate sequence string from two parent Sequence
        objects.

        Selects a GenMethod on each attempt, applies optional extra mutation,
        and validates against all registered Restriction instances. Loops
        until a valid candidate is found or max_attempts is exceeded.

        Parameters
        ----------
        seq1 : Sequence
            First parent, always provided by Evolver.
        seq2 : Sequence
            Second parent, always provided by Evolver. May be the same
            object as seq1 when the population has only one individual.

        Returns
        -------
        str
            A candidate sequence string that passes all restrictions.

        Raises
        ------
        RuntimeError
            When no valid sequence is found within max_attempts iterations.
        """
        for attempt in range(1, self.max_attempts + 1):
            # Method selection is inside the loop so that diverse methods
            # are tried when restrictions are strict.
            method = self._select_method()
            logger.debug(
                f"Generator: attempt {attempt}, method {method}"
            )

            candidate = method.generate(seq1, seq2)

            if self.extra_mutation:
                candidate = self._apply_extra_mutation(candidate)

            if self._passes_restrictions(candidate):
                logger.debug(
                    f"Generator: valid sequence found after {attempt} attempt(s): '{candidate}'"
                )
                return candidate

        raise RuntimeError(
            f"Generator: could not produce a valid sequence after "
            f"{self.max_attempts} attempts. Check that restrictions are not "
            f"too strict for the configured methods and amino acid pool."
        )

    def generate_initial(self, seq1: Sequence, seq2: Sequence) -> str:
        """
        Produces a valid candidate sequence string using the initial method.

        Used by Evolver exclusively during the first population fill.
        Follows the same validation loop as generate().
        """
        for attempt in range(1, self.max_attempts + 1):
            candidate = self.initial_method.generate(seq1, seq2)

            if self.extra_mutation:
                candidate = self._apply_extra_mutation(candidate)

            if self._passes_restrictions(candidate):
                logger.debug(
                    f"Generator: initial sequence found after {attempt} attempt(s): '{candidate}'"
                )
                return candidate

        raise RuntimeError(
            f"Generator: could not produce a valid initial sequence after "
            f"{self.max_attempts} attempts. Check that restrictions are not "
            f"too strict for the initial method and amino acid pool."
        )

    def __repr__(self) -> str:
        return (
            f"Generator(methods={self.methods}, weights={self.weights}, "
            f"aa_pool='{self.aa_pool}', peptide_len={self.peptide_len}, "
            f"extra_mutation={self.extra_mutation})"
        )


# ---------------------------------------------------------------------------
# Internal adapter
# ---------------------------------------------------------------------------

class _StrAdapter:
    """
    Minimal string wrapper used internally by Generator._apply_extra_mutation().

    Allows passing a plain string candidate to _RandomMutation.generate()
    without constructing a full Sequence object. _RandomMutation only reads
    str(seq1) and len(seq1), so this adapter is sufficient.
    """

    def __init__(self, seq: str) -> None:
        self.sequence: str = seq

    def __str__(self) -> str:
        return self.sequence

    def __len__(self) -> int:
        return len(self.sequence)

    def __getitem__(self, index):
        return self.sequence[index]


if __name__ == '__main__':
    pass