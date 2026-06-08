# === sequence.py ===
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
    Stores identity, physicochemical summary properties, fitness history,
    and simulation state. Geometry is handled externally by GenMethod subclasses.
    """

    def __init__(self, seq: str, generation: int = 0, h_scale: str = 'eisenberg') -> None:
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

        # Residue objects
        self.residues: List[Residue] = [
            Residue(letter=aa, index=idx, scale=self.hydrophobic_scale)
            for idx, aa in enumerate(self.sequence)
        ]

        # Physicochemical summary properties
        self.hydrophobic_index: float = round(
            sum(r.hydrophobicity for r in self.residues), 3
        )
        self.charge: float = sum(r.charge for r in self.residues)
        self.n_ter_charge: float = self.residues[0].charge
        self.c_ter_charge: float = self.residues[-1].charge

        # Fitness history
        self.fitness_list: List[float] = []

        # Simulation state flags
        self.is_elite: bool = False
        self.is_preferent: bool = False
        self.is_discarded: bool = False
        self.is_reinserted: bool = False
        self.is_top: bool = False
        self.is_constructed: bool = False
        self.is_just_constructed: bool = False
        self.is_running: bool = False
        self.is_waiting_analysis: bool = False
        self.is_failed: bool = False

        # Directory state
        self.has_directory: bool = False
        self.directory: Optional[str] = None
        self.last_iter_dir: Optional[str] = None

        # Counters
        self.simulation_attempts: int = 0
        self.completed_simulations: int = 0
        self.failed_simulations: int = 0
        self.reinsertions: int = 0
        self.times_elite: int = 0
        self.consecutive_top: int = 0   # consecutive generations in the top section

        # Position in sorted population
        self.current_index: Optional[int] = None

    # Special methods -------------------------------------------------------

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
        """Returns a new sequence string with residues shuffled randomly."""
        as_list = list(self.sequence)
        random.shuffle(as_list)
        return ''.join(as_list)

    # Fitness and penalty ---------------------------------------------------

    @property
    def fitness(self) -> Optional[float]:
        if not self.fitness_list:
            return None
        return sum(self.fitness_list) / len(self.fitness_list)
    
    def get_iterations(self) -> int:
        """Returns the number of fitness evaluations recorded."""
        return len(self.fitness_list)

    # Residue queries -------------------------------------------------------

    def get_charged_res(self, charge: str = 'both', letters: bool = False) -> list:
        """
        Returns indices (or letters) of residues with the specified charge type.
        charge: 'positive', 'negative', or 'both'
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
        Returns True if any amino acid appears consecutively max_rep or more times.
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
        Updates the consecutive-top counter and assigns the elite tag.

        Elite is a diagnostic label: a sequence becomes elite when it has
        stayed in the top section for `iterations_elite` consecutive
        generations. It does NOT affect list distribution or parent selection.
        Returns True if elite status is currently set.
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
        Called when a sequence reappears as a child after having been
        discarded (only happens when avoid_reinsertion == False).
        Increments the reinsertion counter and assigns the preferent tag
        when the threshold is reached.

        Preferent is a diagnostic label: it signals that parents lack enough
        variability, since the same sequence keeps being regenerated.
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
