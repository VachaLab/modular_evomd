# === sequence.py ===
"""
Sequence: the individual unit optimized by Evo-MD.

A Sequence represents one peptide. It bundles:
  - identity: the residue string, the generation it was created in, and the
    hydrophobicity scale used for its residues;
  - per-residue objects (Residue) and summary physicochemical properties
    (hydrophobic index, net charge, terminal charges);
  - a fitness history (fitness_list) exposed as a mean via the `fitness`
    property;
  - simulation state flags and counters used by the Evolver and Manager to
    track where the sequence is in the construct/calculate/check/analyze cycle;
  - diagnostic labels: `is_elite` (stuck in the top for several consecutive
    generations) and `is_preferent` (regenerated and discarded repeatedly,
    hinting at low parent variability).

Sequences are unique objects: a given peptide exists only once across all of
the Evolver's lists, and is moved between lists rather than duplicated.
Equality and hashing are based on the residue string, so a Sequence compares
equal to another Sequence (or a plain str) with the same letters.

Geometry (helix positions, hydrophobic moment, etc.) is computed externally by
the GenMethod subclasses, not stored here.
"""

import logging
import random
from typing import List, Optional
from residue import Residue
from scales import Scales
from utils import ResidueError

logger = logging.getLogger(__name__)

# Valid amino acid letters derived from Scales
_VALID_AA: set = set(Scales.aa_charges.keys())


class Sequence:
    """
    Represents a peptide sequence as the individual unit to be optimized.

    Stores identity, physicochemical summary properties, fitness history, and
    simulation state. Geometry is handled externally by GenMethod subclasses.
    """

    def __init__(self, seq: str, generation: int = 0, h_scale: str = 'eisenberg') -> None:
        """
        Build and validate a Sequence.

        The residue string is validated (non-empty, all letters known) and the
        hydrophobicity scale is checked before any property is computed. Summary
        properties are derived once at construction from the per-residue values.

        Args:
            seq (str): Peptide residue string (case-insensitive; stored upper-cased).
            generation (int): Generation in which the sequence was created.
            h_scale (str): Hydrophobicity scale used for the residues.

        Raises:
            ResidueError: If the string is empty or contains unknown residues.
            ValueError: If the hydrophobicity scale is unknown.
        """
        # Validate residue letters before creating the object
        if not seq:
            raise ResidueError("Sequence cannot be empty.")
        invalid = [aa for aa in seq if aa.upper() not in _VALID_AA]
        if invalid:
            raise ResidueError(f"Unrecognized residue(s) in sequence '{seq}': {invalid}")
        if h_scale not in Scales.hydrophobicity_scales:
            raise ValueError(f"Unknown hydrophobicity scale: '{h_scale}'. "
                             f"Available: {list(Scales.hydrophobicity_scales.keys())}")

        # Core identity
        self.sequence: str = seq.upper()
        self.generation: int = generation
        self.hydrophobic_scale: str = h_scale

        # Residue objects (one per letter, carrying position and properties)
        self.residues: List[Residue] = [
            Residue(letter=aa, index=idx, scale=self.hydrophobic_scale)
            for idx, aa in enumerate(self.sequence)
        ]

        # Physicochemical summary properties (derived once from the residues)
        self.hydrophobic_index: float = round(
            sum(r.hydrophobicity for r in self.residues), 3
        )
        self.charge: float = sum(r.charge for r in self.residues)
        self.n_ter_charge: float = self.residues[0].charge   # charge of the N-terminal residue
        self.c_ter_charge: float = self.residues[-1].charge  # charge of the C-terminal residue

        # Fitness history: one entry per evaluation; mean exposed via `fitness`
        self.fitness_list: List[float] = []

        # Simulation state flags
        self.is_elite: bool = False             # diagnostic: stuck in the top for consecutive generations
        self.is_preferent: bool = False         # diagnostic: regenerated/discarded repeatedly
        self.is_discarded: bool = False         # currently in the discarded list
        self.is_reinserted: bool = False        # has been reinserted at least once
        self.is_top: bool = False               # within the top (elite-ratio) fraction this generation
        self.is_constructed: bool = False       # simulation system has been built
        self.is_just_constructed: bool = False  # built in the current step, awaiting calculation
        self.is_running: bool = False           # simulation in progress
        self.is_waiting_analysis: bool = False  # simulation finished, fitness not yet computed
        self.is_failed: bool = False            # simulation failed

        # Directory state
        self.has_directory: bool = False
        self.directory: Optional[str] = None       # base directory for this sequence
        self.last_iter_dir: Optional[str] = None   # directory of the latest iteration

        # Counters
        self.simulation_attempts: int = 0
        self.completed_simulations: int = 0
        self.failed_simulations: int = 0
        self.reinsertions: int = 0          # times regenerated after being discarded
        self.times_elite: int = 0           # times the elite tag has been (re)confirmed
        self.consecutive_top: int = 0       # consecutive generations in the top section

        # Position in sorted population
        self.current_index: Optional[int] = None  # index in the last sorted ranking

    # Special methods -------------------------------------------------------
    # Sequences behave like their residue string for comparison, hashing,
    # iteration, indexing and concatenation, so they can be used directly in
    # membership tests and string operations.

    def __str__(self) -> str:
        return self.sequence

    def __repr__(self) -> str:
        return self.sequence

    def __len__(self) -> int:
        return len(self.sequence)

    def __iter__(self):
        return iter(self.sequence)

    def __getitem__(self, index):
        return self.sequence[index]

    def __add__(self, other):
        if isinstance(other, str):
            return str(self) + other
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, str):
            return other + str(self)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.sequence)

    def __eq__(self, other) -> bool:
        # Equal to another Sequence or a plain str with the same letters.
        if isinstance(other, Sequence):
            return self.sequence == other.sequence
        if isinstance(other, str):
            return self.sequence == other
        return NotImplemented

    def __contains__(self, item) -> bool:
        return item in self.sequence

    def __reversed__(self):
        return reversed(self.sequence)

    # Sequence utilities ----------------------------------------------------

    def randomize(self) -> str:
        """Return a new string with this sequence's residues shuffled randomly."""
        as_list = list(self.sequence)
        random.shuffle(as_list)
        return ''.join(as_list)

    # Fitness and penalty ---------------------------------------------------

    @property
    def fitness(self) -> Optional[float]:
        """Mean of all recorded fitness evaluations, or None if none recorded yet."""
        if not self.fitness_list:
            return None
        return sum(self.fitness_list) / len(self.fitness_list)
    
    def get_iterations(self) -> int:
        """Return the number of fitness evaluations recorded."""
        return len(self.fitness_list)

    # Residue queries -------------------------------------------------------

    def get_charged_res(self, charge: str = 'both', letters: bool = False) -> list:
        """
        Return the charged residues of the requested sign.

        Args:
            charge (str): Which residues to return: 'positive', 'negative', or
                'both' (any non-zero charge).
            letters (bool): If True, return residue letters; otherwise return
                their positional indices.

        Returns:
            list: Indices (default) or letters of the matching residues.

        Raises:
            ValueError: If `charge` is not one of the accepted filters.
        """
        testers = {
            'positive': lambda x: x > 0,
            'negative': lambda x: x < 0,
            'both':     lambda x: x != 0,
        }
        if charge not in testers:
            raise ValueError(f"Invalid charge filter '{charge}'. Use 'positive', 'negative', or 'both'.")
        tester = testers[charge]
        if letters:
            return [r.letter for r in self.residues if tester(r.charge)]
        return [r.index for r in self.residues if tester(r.charge)]

    def check_consecutive_aa(self, max_rep: int) -> bool:
        """
        Report whether any residue repeats consecutively too many times.

        Args:
            max_rep (int): Threshold; the check triggers at this many identical
                residues in a row.

        Returns:
            bool: True if some amino acid appears `max_rep` or more times
            consecutively, else False.
        """
        if len(self.sequence) < max_rep:
            return False
        repetitions = 1
        for i in range(1, len(self.sequence)):
            if self.sequence[i] == self.sequence[i - 1]:
                repetitions += 1
                if repetitions >= max_rep:
                    return True
            else:
                repetitions = 1
        return False

    # State management ------------------------------------------------------

    def check_elite(self, iterations_elite: int = 3) -> bool:
        """
        Update the consecutive-top streak and (re)assign the elite tag.

        Elite is a purely diagnostic label: a sequence becomes elite once it has
        stayed in the top section for `iterations_elite` consecutive
        generations. When the streak breaks (not in the top this generation),
        the counter and the tag are reset. Elite does NOT affect list
        distribution or parent selection.

        Intended to be called once per sequence per sort, with `is_top` already
        set for the current generation.

        Args:
            iterations_elite (int): Consecutive top generations required to earn
                the tag.

        Returns:
            bool: True if the sequence is elite after this update, else False.
        """
        if self.is_top:
            self.consecutive_top += 1
        else:
            # streak broken --> reset both counter and label
            self.consecutive_top = 0
            self.is_elite = False
            return False

        if self.consecutive_top >= iterations_elite:
            if not self.is_elite:
                logger.info(f"Sequence {self.sequence} is elite "
                            f"({self.consecutive_top} consecutive top generations)")
            self.is_elite = True
            self.times_elite += 1
            return True

        self.is_elite = False
        return False
    
    def check_reinsertion(self, iterations_preferent: int = 3) -> None:
        """
        Record a reinsertion and (possibly) assign the preferent tag.

        Called when a sequence reappears as a child after having been discarded,
        which only happens when avoid_reinsertion is False. Increments the
        reinsertion counter and marks the sequence as reinserted (and no longer
        discarded). Once reinsertions reach the threshold, the sequence is
        flagged as preferent.

        Preferent is a purely diagnostic label: repeatedly regenerating the same
        sequence suggests the parents lack enough variability.

        Args:
            iterations_preferent (int): Reinsertion count at which the preferent
                tag is assigned.
        """
        self.reinsertions += 1
        self.is_discarded = False
        self.is_reinserted = True
        logger.info(f"Sequence {self.sequence} was reinserted ({self.reinsertions})")
        if self.reinsertions >= iterations_preferent:
            self.is_preferent = True
            logger.warning(f"Sequence {self.sequence} is preferent "
                           f"({self.reinsertions} reinsertions) --> parents may lack variability")


if __name__ == '__main__':
    pass
