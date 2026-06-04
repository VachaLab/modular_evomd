# === evo-md.py ===
"""
Command-line entry point for the evolutionary optimizer (evo-md).

This script ties together the three main pieces of the project:

    * Instructor : reads the input YAML and holds the configuration.
    * Evolver    : the optimizer itself; keeps the population of Sequences,
                   runs the evolutionary cycle and stores its state in a
                   pickle file (``evolver.pkl``).
    * Manager    : (used internally by the Evolver) handles directories,
                   simulations and result analysis.

Typical workflow
-----------------
    1. Create the evolver object from a YAML file (saved as ``evolver.pkl``)::

           python evo-md.py --create-evolver --file input.yaml

    2. Start the optimization (runs until a stop condition is met)::

           python evo-md.py --start

The behaviour is selected by the command-line arguments parsed in
``parserlib.get_arguments``; ``main`` simply dispatches on those flags.
There are three ways to stop a run: kill the process (state is kept in
``evolver.pkl``), set ``max_generations`` in the YAML, or set
``target_fitness`` in the YAML. The last two are handled in
``Evolver.check_termination``.
"""
#--------------------------------------------
# Global logging configuration
import logging


def setup_logging() -> None:
    """Configure the root logger (level INFO, ``LEVEL: message`` format)."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
#--------------------------------------------
from parserlib import get_arguments
from instructor import Instructor
from evolver import Evolver
import utils


def iterate_evolver(evo: Evolver, fast_cycle: bool = False) -> None:
    """Run one evolutionary iteration and persist the state.

    Wraps ``Evolver.iterate`` (construct -> calculate -> check -> analyze)
    and saves the pickle afterwards unless ``fast_cycle`` is enabled.

    Parameters
    ----------
    evo : Evolver
        The evolver instance to advance.
    fast_cycle : bool
        If True, skip the intermediate ``save_pkl`` call (faster, but the
        on-disk state is only written at the very end of the run).
    """
    evo.iterate()
    if not fast_cycle:
        evo.save_pkl()  # save again


def handling_evolver(evo: Evolver, fast_cycle: bool = False) -> None:
    """Sort the population and refill it for the next generation.

    Calls ``sort_sequences`` (rank by fitness and split into elite /
    parents / discarded) and then ``populate`` (generate the offspring
    that make up the next generation), saving the pickle after each step
    unless ``fast_cycle`` is enabled.

    Parameters
    ----------
    evo : Evolver
        The evolver instance to handle.
    fast_cycle : bool
        If True, skip the intermediate ``save_pkl`` calls.
    """
    evo.sort_sequences()
    if not fast_cycle:
        evo.save_pkl()  # save after sorting
    evo.populate()
    if not fast_cycle:
        evo.save_pkl()  # save again


def get_evolver(args, skip_new: bool = False, internal: bool = False) -> Evolver:
    """Load an existing Evolver or create a new one, depending on ``args``.

    Resolution order:
        1. If ``--file`` points to an existing YAML and ``skip_new`` is
           False: build a brand-new Evolver from that configuration and
           save it as ``evolver.pkl``.
        2. Else, if ``--evopkl`` points to an existing pickle: load it.
        3. Else, if ``evolver.pkl`` exists in the current directory: load it.
        4. Otherwise: log an error and exit.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    skip_new : bool
        If True, never create a new Evolver from the YAML (used by actions
        that must operate on an already-existing object, e.g. ``--show-evolver``).
    internal : bool
        If True, this is an internal reload inside the main loop; in that
        case the recover flag is NOT re-enabled (recovery must only run on
        the first iteration of a resumed session).

    Returns
    -------
    Evolver
        The loaded or freshly created evolver instance.
    """
    logger = logging.getLogger(__name__)
    evo_pre = 'evolver.pkl'
    evo = None

    if args.file and utils.exists(args.file) and not skip_new:
        # New session
        inst = Instructor(args.file)
        evo = Evolver(inst)
        evo.save_pkl()
    elif args.evopkl and utils.exists(args.evopkl):
        evo = utils.read_pkl(args.evopkl)
    elif utils.exists(evo_pre):
        evo = utils.read_pkl(evo_pre)
    else:
        logger.error('Evo-MD: Failed to create or read Evolver')
        exit(1)

    # Assign recover flag only if resuming from existing object
    if args.restart and not internal:
        evo.recover_enabled = True

    return evo

def main() -> None:
    """Dispatch to the right action based on the command-line arguments.

    Each ``elif`` branch corresponds to one mutually exclusive action flag
    defined in ``parserlib``. Most branches load or create an Evolver, do
    their job, and call ``exit``. The ``--start`` branch is the main one:
    it runs the evolutionary loop until a termination criterion is met.
    """

    logger = logging.getLogger(__name__)
    args = get_arguments()

    # what do I have to do?
    if args.show_defaults:
        # Print the default Instructor configuration and exit (no evolver needed).
        logger.info('Showing Instructor default configutation . . .')
        print(Instructor(args.file))
        exit(0)

    elif args.show_current:
        # Print the configuration stored in the existing evolver.
        evo = get_evolver(args, skip_new=True)
        logger.info('Showing Instructor configutation . . .')
        # evo.instructor.show_configuration()
        print(evo.instructor)
        exit(0)

    elif args.change_method:
        # Interactively edit optimization-method parameters on an existing evolver.
        # Load the latest Evolver object
        evo = get_evolver(args, skip_new=True)
        logger.info('Showing Evolver . . .')
        print(evo.instructor)

        # Interactive method modification
        evo.instructor.interactive_change_method(evo.generations)

        # Save updated state
        logger.info('Saving new configuration . . .')
        evo.save_pkl()
        exit(0)

    elif args.show_evolver:
        # Print the full evolver state (population summary + top sequences).
        evo = get_evolver(args, skip_new=True)
        if args.top_list:
            evo.instructor.top_list = int(args.top_list)
        logger.info('Showing Evolver . . .')
        print(evo)
        exit(0)

    elif args.report_sequences:
        # Dump every sequence to sequences_report.csv and exit.
        evo = get_evolver(args, skip_new=True)
        logger.info('Creating Evolver report . . .')
        evo.report_sequences()
        exit(0)

    elif args.create_evolver:
        # Build evolver.pkl from the YAML, optionally seeding it from a report,
        # then show it and exit (does NOT start the optimization).
        # It just creates evolver and exit
        evo = get_evolver(args)
        # set fast_cycle
        evo.fast_cycle = args.fast_cycle
        if args.read_report:
            # initialize from an existing report instead of populating from scratch
            evo.read_report(args.read_report)
        else:
            # populate sequences
            evo.populate()
        evo.save_pkl()
        logger.info('Showing Evolver . . .')
        print(evo)
        exit(0)
    
    elif args.populate_previous:
        # Seed the population from simulations already present on disk.
        evo = get_evolver(args)
        evo.read_previous()
        evo.save_pkl()
        exit()
    
    elif args.stop_evolver:
        # Flag the evolver to stop after the current iteration finishes.
        # changes evo.runnable to false
        evo = get_evolver(args, skip_new=True)
        logger.info('Stopping evolver . . .')
        evo.runnable = False
        evo.save_pkl()
        exit(0)

    elif args.start:
        # ---- Main action: run the evolutionary loop ----
        evo = get_evolver(args)

        # set fast_cycle
        evo.fast_cycle = args.fast_cycle

        # populate sequences
        evo.populate()
        evo.started = True
        if not args.fast_cycle:
            evo.save_pkl()

        # Loop until check_termination() (or an external --stop) clears runnable.
        while evo.runnable:
            # --- reload evolver ---
            # Reload from disk each cycle so external edits (e.g. --stop,
            # --insert-sequence) are picked up; skipped in fast_cycle mode.
            if not args.fast_cycle:
                evo = get_evolver(args, internal=True)

            # --- iterate ---
            iterate_evolver(evo, fast_cycle=args.fast_cycle)

            # --- sort and repopulate ---
            handling_evolver(evo, fast_cycle=args.fast_cycle)

            # --- back up sequences ---
            if not args.fast_cycle:
                evo.sequence_backup()

            # --- check stopping criteria ---
            evo.check_termination()

        evo.save_pkl()
        print(evo)
        exit(0)

    elif args.insert_sequence:
        # Queue a sequence to be injected into the next generation.
        evo = get_evolver(args)
        # insert sequence into Evolver.to_include list
        try:
            evo.to_include.append(args.insert_sequence)
        except:
            evo.to_include = [args.insert_sequence]
        exit(0)
    
    elif args.show_lists:
        # Print every sequence grouped by the list it currently belongs to.
        evo = get_evolver(args)
        # show sequences
        logger.info('Showing sequences\n')
        logger.info('Evolver.sequences')
        total = 0
        i = 0
        for seq in evo.sequences:
            logger.info(f'{seq}')
            i += 1
        logger.info(f'--> {i} sequences found\n')
        total += i
        # parents
        logger.info('Evolver.parent_sequences')
        i = 0
        for seq in evo.parent_sequences:
            logger.info(f'{seq}')
            i += 1
        logger.info(f'--> {i} sequences found\n')
        total += i
        # discarded
        logger.info('Evolver.discarded_sequences')
        i = 0
        for seq in evo.discarded_sequences:
            logger.info(f'{seq}')
            i += 1
        logger.info(f'--> {i} sequences found\n')
        total += i
        # excluded sequences
        logger.info('Evolver.excluded_sequences')
        i = 0
        for seq in evo.excluded_sequences:
            logger.info(f'{seq}')
            i += 1
        logger.info(f'--> {i} sequences found\n')
        total += i
        # total
        logger.info(f'Total: {total} sequences')
        exit(0)
    
    elif args.plot_evolution:
        # Plot fitness across generations and save evolution.png.
        evo = get_evolver(args)
        # plot_evolution
        evo.plot_evolution(show_std=args.show_std, show_kids=args.show_kids)
        exit(0)

    elif args.last_generation:
        # Roll the population back to the last fully-evaluated generation
        # and repopulate from there (useful after an interrupted run).
        import math
        evo = get_evolver(args)
        # list with sequences evaluated
        # NOTE: Sequence exposes fitness as a property, not get_mean_fitness();
        # this call will raise AttributeError. Kept as-is per "do not change
        # the structure"; replace with `k.fitness` when fixing.  # FIXME
        evo.discarded_sequences = [k for k in evo.discarded_sequences if k.get_mean_fitness() is not None and not math.isnan(k.get_mean_fitness())]
        evo.parent_sequences = []
        evo.sequences = []
        evo.generations = max(set([k.generation for k in evo.discarded_sequences]))
        print(f"Las completed generation: {evo.generations}")
        print(f"Completed sequences: {len(evo.discarded_sequences)}")
        evo.sort_sequences()
        evo.save_pkl()  # save after sorting
        evo.populate()
        evo.save_pkl()  # save again
        print(evo)

    elif args.test:
        """
        This sections is used to include testing code
        """
        # Scratch branch for ad-hoc testing; not part of the normal workflow.
        evo = get_evolver(args)
        evo.sequence_backup()
        exit(0)

if __name__ == '__main__':
    setup_logging()
    main()
