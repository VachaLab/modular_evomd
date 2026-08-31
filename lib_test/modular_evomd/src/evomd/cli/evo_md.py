# === evo-md.py ===
"""
Evo-MD command-line entry point.

This is the top-level script that runs Evo-MD. It reads the command-line
arguments (see parserlib.get_arguments), builds or loads an Evolver, and
dispatches to exactly one action based on which flag was selected.

Workflow overview:
    1. setup_logging() configures global logging.
    2. main() parses the arguments and runs the matching action.
    3. Most actions obtain an Evolver via get_evolver(), do their work, and
       exit. --start/--restart delegate the full evolutionary loop to
       Evolver.start()/Evolver.restart(), so the CLI and the notebook/library
       API (evomd.create_evolver(), evo.start(), evo.restart()) share the
       exact same loop implementation.

Action dispatch:
    main() uses an if/elif chain over the action flags and exits at the end of
    each branch, so only the first matching action runs per invocation.

Persistence:
    The Evolver is saved as a pickle (evolver.pkl by default). Intermediate
    saves are skipped when --fast-cycle is set, trading crash-resistance for
    speed. The in-memory Evolver is the single source of truth for the whole
    --start/--restart run; it is not reloaded from disk mid-loop.
"""

#--------------------------------------------
# Global logging configuration
import logging

def setup_logging():
    """Configure root logging: INFO level with a 'LEVEL: message' format."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
#--------------------------------------------
from .parserlib import get_arguments
from ..core.instructor import Instructor
from ..core.evolver import Evolver
from ..core import utils


def get_evolver(args, skip_new=False):
    """
    Build a new Evolver or load an existing one, according to the arguments.

    Resolution order:
        1. If --file is given, it exists, and skip_new is False -> create a new
           session from the YAML instruction file and save it.
        2. Else if --evopkl is given and exists -> load that pickle.
        3. Else if the default 'evolver.pkl' exists -> load it.
        4. Otherwise -> log an error and exit(1).

    Setting recover_enabled (for --restart) is not this function's job: it
    is set by Evolver.restart() itself, right before it runs the loop.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        skip_new (bool): If True, never create a new session from --file
            (forces loading an existing Evolver). Used by inspection actions
            that must not overwrite the current session.

    Returns:
        Evolver: The created or loaded Evolver instance.
    """
    logger = logging.getLogger(__name__)
    evo_pre = 'evolver.pkl'  # default pickle name looked up in the working dir
    evo = None

    if args.file and utils.exists(args.file) and not skip_new:
        if utils.exists(evo_pre):
            logger.warning("evolver.pkl file found in current directory.")
            logger.warning("A new evolver.pkl file will be created. Do you want to continue? (y/n): ")
            reply = input()
            if reply.lower()[0] != 'y':
                exit(0)
        # New session created from the YAML instruction file
        inst = Instructor(args.file)
        evo = Evolver(inst)
        evo.save_pkl()
    elif args.evopkl and utils.exists(args.evopkl):
        # Load a pickle whose name is not the default 'evolver.pkl'
        evo = utils.read_pkl(args.evopkl)
    elif utils.exists(evo_pre):
        # Fall back to the default pickle in the working directory
        evo = utils.read_pkl(evo_pre)
    else:
        logger.error('Evo-MD: Failed to create or read Evolver')
        exit(1)

    return evo


def main():
    """
    Parse the arguments and run the single matching Evo-MD action.

    Dispatches on the action flags defined in parserlib. Each branch performs
    its action and exits, so only one action runs per invocation. See the
    module docstring for the overall workflow.
    """
    logger = logging.getLogger(__name__)
    args = get_arguments()

    # Dispatch on the selected action (first match wins, then exit).
    if args.show_defaults:
        # --show-defaults: print Instructor defaults (no session needed).
        logger.info('Showing Instructor default configuration . . .')
        print(Instructor(args.file))
        exit(0)

    elif args.show_current:
        # --show-current: print the loaded session's Instructor configuration.
        evo = get_evolver(args, skip_new=True)
        logger.info('Showing Instructor configuration . . .')
        # evo.instructor.show_configuration()
        print(evo.instructor)
        exit(0)

    elif args.show_evolver:
        # --show-evolver: print the Evolver summary. --top-list overrides how
        # many top sequences are shown.
        evo = get_evolver(args, skip_new=True)
        if args.top_list:
            evo.instructor.top_list = int(args.top_list)
        logger.info('Showing Evolver . . .')
        evo.show_evolver()
        exit(0)

    elif args.report_sequences:
        # --report-sequences: write all sequences to sequences_report.csv.
        evo = get_evolver(args, skip_new=True)
        logger.info('Creating Evolver report . . .')
        evo.report_sequences()
        exit(0)

    elif args.create_evolver:
        # --create-evolver: build the Evolver, populate it (or initialize from
        # a CSV report via --read-report), show it, and exit without running
        # the evolution loop.
        evo = get_evolver(args)
        # activate console output if requested
        if args.verbose:
            evo.verbose = True
        else:
            evo.verbose = False
        # propagate the fast-cycle flag to the Evolver
        evo.fast_cycle = args.fast_cycle
        if args.read_report:
            # initialize from an existing report instead of populating from scratch
            evo.read_report(args.read_report)
        else:
            # populate sequences from scratch
            evo.populate()
        evo.save_pkl()
        logger.info('Showing Evolver . . .')
        evo.show_evolver()
        exit(0)

    elif args.populate_previous:
        # --populate-previous: populate from results of previous simulations
        # found in the simulation directory, then save and exit.
        evo = get_evolver(args)
        evo.read_directories()
        evo.save_pkl()
        exit(0)
    
    elif args.from_backup:
        # --populate-previous: populate from results of previous simulations
        # found in the simulation directory, then save and exit.
        evo = get_evolver(args)
        evo.read_previous()
        evo.save_pkl()
        exit(0)

    elif args.stop_evolver:
        # --stop-evolver: mark the Evolver as intentionally stopped so a
        # later --start/--restart on the same pkl does not resume it by
        # accident, then save and exit.
        evo = get_evolver(args, skip_new=True)
        logger.info('Stopping evolver . . .')
        evo.stop_evolver()
        exit(0)

    elif args.start:
        # --start / --restart: run the full evolution loop. --restart implies
        # --start (see parserlib), so this branch handles both; which one
        # actually ran is only distinguished by whether recovery was engaged.
        evo = get_evolver(args)

        # activate console output if requested
        if args.verbose:
            evo.verbose = True
        else:
            evo.verbose = False

        if args.restart:
            evo.restart(fast_cycle=args.fast_cycle)
        else:
            evo.start(fast_cycle=args.fast_cycle)

        evo.show_evolver()
        exit(0)

    elif args.show_lists:
        # --show-lists: print every Evolver sequence list with per-list counts
        # and a grand total, then exit.
        evo = get_evolver(args)
        logger.info('Showing sequences\n')

        total = 0

        # current sequences
        print('Evolver.sequences')
        i = 0
        for seq in evo.sequences:
            print(f'{seq}')
            i += 1
        print(f'--> {i} sequences found\n')
        total += i

        # parent sequences
        print('Evolver.parent_sequences')
        i = 0
        for seq in evo.parent_sequences:
            print(f'{seq}')
            i += 1
        print(f'--> {i} sequences found\n')
        total += i

        # discarded sequences
        print('Evolver.discarded_sequences')
        i = 0
        for seq in evo.discarded_sequences:
            print(f'{seq}')
            i += 1
        print(f'--> {i} sequences found\n')
        total += i

        # excluded (forbidden) sequences
        print('Evolver.excluded_sequences')
        i = 0
        for seq in evo.excluded_sequences:
            print(f'{seq}')
            i += 1
        print(f'--> {i} sequences found\n')
        total += i

        # grand total across all lists
        print(f'Total: {total} sequences')
        exit(0)

    elif args.plot_evolution:
        # --plot-evolution: plot fitness across generations. Modifiers:
        # --show-std (std-dev overlay), --show-kids (per-individual scatter),
        # --plot-name (output filename).
        evo = get_evolver(args)
        evo.plot_evolution(show_std=args.show_std, show_kids=args.show_kids, name=args.plot_name)
        exit(0)

    elif args.last_generation:
        # --last-generation: roll back to the last fully completed generation,
        # save the updated state, and exit.
        evo = get_evolver(args, skip_new=True)
        logger.info('Going back to the last completed generation . . .')
        evo.revert_last_generation()
        evo.save_pkl()  # persist the rolled-back state
        evo.show_evolver()
        exit(0)


if __name__ == '__main__':
    setup_logging()
    main()
