# === restriction.py ===
from __future__ import annotations
from sequence import Sequence
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

    def _as_sequence(self, seq: Sequence | str) -> Sequence:
        """
        Returns a Sequence object. Instantiates one only when the input
        is a plain string, avoiding redundant construction otherwise.
        """
        if isinstance(seq, Sequence):
            return seq
        return Sequence(str(seq))


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
    min_count : int
        Minimum number of residues from the set required. Default 0.
    max_count : int | None
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
    Acepta secuencias cuya distribución de hidrofobicidad (índice de alternancia)
    cae dentro de un rango especificado.

    El índice se calcula como el promedio de las diferencias absolutas de 
    hidrofobicidad entre residuos adyacentes usando la escala de Eisenberg.
    Residuos no presentes en la escala se tratan como neutros (0.0) y se 
    registra una advertencia.

    Parameters
    ----------
    min : float | None
        Índice de alternancia mínimo aceptado (inclusivo). None significa sin límite inferior.
    max : float | None
        Índice de alternancia máximo aceptado (inclusivo). None significa sin límite superior.
    """

    def __init__(
        self,
        min: Optional[float] = None,
        max: Optional[float] = None,
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

        # Import here to avoid circular dependency at module level.
        from scales import Scales
        self._hi_table: dict[str, float] = Scales.hydrophobicity_scales["eisenberg"]

    def test(self, seq: str, verbose=False) -> bool:
        # 1. Convertir la secuencia a valores numéricos y manejar residuos desconocidos
        valores = []
        for aa in seq.upper():
            if aa not in self._hi_table:
                logger.warning(
                    f"HdistributionRestriction: unrecognized residue '{aa}' treated as neutral (0.0)."
                )
            valores.append(self._hi_table.get(aa, 0.0))

        # 2. Calcular el índice de distribución (alternancia)
        if len(valores) < 2:
            dist_index = 0.0  # Si la secuencia tiene 0 o 1 aminoácido, la diferencia es 0
        else:
            diferencias = [abs(valores[i+1] - valores[i]) for i in range(len(valores)-1)]
            dist_index = sum(diferencias) / len(diferencias)

        # 3. Evaluar contra los límites min y max
        above_min = self._min is None or dist_index >= self._min
        below_max = self._max is None or dist_index <= self._max

        passed = above_min and below_max

        # 4. Imprimir resultados si es verbose
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
        from scales import Scales
        self._charge_table: dict[str, float] = Scales.aa_charges

    def test(self, seq: str, verbose=False) -> bool:
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
    Accepts sequences whose net charge falls within a specified range.

    Hydrophobic index is computed as the sum of per-residue hydrophobic moments using the values
    defined in Scales.hydrophobicity_scales["eisenberg"]. Residues not present in the scale are
    treated as zero (0.0) and a warning is logged.

    Parameters
    ----------
    min : float | None
        Minimum accepted hydrophobic index (inclusive). None means no lower bound.
    max : float | None
        Maximum accepted hydrophobic index (inclusive). None means no upper bound.
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
                f"HindexRestriction: min ({self._min}) must be "
                f"<= max ({self._max})."
            )

        # Import here to avoid circular dependency at module level.
        from scales import Scales
        self._hi_table: dict[str, float] = Scales.hydrophobicity_scales["eisenberg"]

    def test(self, seq: str, verbose=False) -> bool:
        test_seq = Sequence(seq)
        hindex = round(test_seq.hydrophobic_index, 4)
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

    Hydrophobic moment is computed as described in Faraday Symp. Chem. Soc., 1982, 17,109-120.

    Parameters
    ----------
    min : float | None
        Minimum accepted hydrophobic index (inclusive). None means no lower bound.
    max : float | None
        Maximum accepted hydrophobic index (inclusive). None means no upper bound.
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
        from scales import Scales
        self._hi_table: dict[str, float] = Scales.hydrophobicity_scales["eisenberg"]

    def test(self, seq: str, verbose=False) -> bool:
        from sequence_geometry import compute_helix_positions, compute_hm_scalar
        seq = self._as_sequence(seq)
        
        positions = compute_helix_positions(seq, translate=False)
        hm_scalar = compute_hm_scalar(seq, positions)

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
