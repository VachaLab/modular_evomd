"""
evomd: a molecular-simulation-driven evolutionary engine for peptide
(amino acid) sequence optimization.

Usable both as a set of console scripts (evo-md, peptide-viewer,
sequence-logo) and as an importable library, e.g.:

    import evomd

    evo = evomd.create_evolver(instructions)  # path to a YAML file, or raw YAML text
    evo.show_evolver()
    evo.start(fast_cycle=True)
    evo.plot_evolution()
"""

from .core.sequence import Sequence
from .core.residue import Residue
from .core.scales import Scales
from .core.evolver import Evolver
from .core.instructor import Instructor
from .core.manager import Manager
from .core.generator import Generator

__version__ = "0.1.0"

__all__ = [
    "Sequence",
    "Residue",
    "Scales",
    "Evolver",
    "Instructor",
    "Manager",
    "Generator",
    "create_evolver",
]


def create_evolver(instructions: str, fast_cycle: bool = False, verbose: bool = False) -> Evolver:
    """
    Build a new Evolver from YAML instructions.

    This is the library entry point equivalent to evo-md.py's
    --create-evolver action: it builds the Instructor (and, through it, the
    Generator) from the given instructions, wraps it in a new Evolver, and
    saves an initial checkpoint before any sequence is populated.

    Args:
        instructions (str): Either a path to an existing YAML instruction
            file, or a string containing the YAML content directly (e.g.
            built inline in a notebook with io.StringIO-compatible text).
        fast_cycle (bool): Initial fast_cycle flag on the returned Evolver.
        verbose (bool): Initial verbose flag on the returned Evolver.

    Returns:
        Evolver: A newly created Evolver, not yet populated or started.
            Call evo.start()/evo.restart() to run the evolution.
    """
    instructor = Instructor(instructions)
    evo = Evolver(instructor, fast_cycle=fast_cycle, verbose=verbose)
    evo.save_pkl()
    return evo
