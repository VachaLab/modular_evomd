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
from .core import utils as _utils

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
    "create_evolver_from_report",
    "load_evolver",
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


def evolver_from_report(
    instructions: str, report_path: str, fast_cycle: bool = False, verbose: bool = False
) -> Evolver:
    """
    Build a new Evolver from YAML instructions, seeded from a CSV report.

    Equivalent to `evo-md --create-evolver --read-report <report_path>`: same
    as create_evolver(), but instead of starting from an empty population,
    the Evolver's sequences are initialized from a CSV report with the
    format produced by Evolver.report_sequences() / `evo-md --report-sequences`
    (columns: sequence, generation, fitness).

    Args:
        instructions (str): Either a path to an existing YAML instruction
            file, or a string containing the YAML content directly.
        report_path (str): Path to the CSV report to read sequences from.
        fast_cycle (bool): Initial fast_cycle flag on the returned Evolver.
        verbose (bool): Initial verbose flag on the returned Evolver.

    Returns:
        Evolver: A newly created Evolver, seeded from the report. Call
            evo.start()/evo.restart() to continue the evolution from there.

    Raises:
        FileNotFoundError: If report_path does not exist.
    """
    instructor = Instructor(instructions)
    evo = Evolver(instructor, fast_cycle=fast_cycle, verbose=verbose)
    evo.read_report(report_path)
    evo.save_pkl()
    return evo


def load_evolver(path: str = "evolver.pkl") -> Evolver:
    """
    Load a previously saved Evolver from a pickle file.

    A thin, validated wrapper around pickle loading -- avoids having to
    `import pickle` directly in a notebook and gives a clear error if the
    path is wrong or the file doesn't contain an Evolver.

    Args:
        path (str): Path to the .pkl file (defaults to 'evolver.pkl', the
            default name used by create_evolver()/evo-md.py in the current
            working directory).

    Returns:
        Evolver: The loaded Evolver, with all of its state (sequences,
            generation count, runnable/started flags, etc.) exactly as it
            was when last saved. Call evo.restart() to resume it, or just
            inspect it with evo.show_evolver(), evo.plot_evolution(), etc.

    Raises:
        FileNotFoundError: If path does not exist.
        TypeError: If the file exists but does not contain an Evolver.
    """
    if not _utils.exists(path):
        raise FileNotFoundError(f"No such Evolver pickle file: '{path}'")
    evo = _utils.read_pkl(path)
    if not isinstance(evo, Evolver):
        raise TypeError(
            f"'{path}' was loaded but does not contain an Evolver "
            f"(got {type(evo).__name__} instead)."
        )
    return evo
