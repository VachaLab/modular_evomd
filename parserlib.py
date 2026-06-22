# === parserlib.py ===
"""
Command-line argument parsing for Evo-MD.

This module defines a single entry point, `get_arguments()`, which builds the
argparse parser used by `evo-md.py` and returns the parsed namespace.

The arguments fall into two broad groups:

1. Action selectors (mutually exclusive in practice): exactly one of these is
   expected to be chosen per invocation, since `evo-md.py` dispatches on the
   first matching action and then exits. Examples: --show-defaults,
   --show-current, --show-evolver, --report-sequences, --create-evolver,
   --populate-previous, --stop-evolver, --start, --show-lists,
   --plot-evolution, --last-generation.

2. Modifiers: flags and values that tune the behaviour of an action without
   selecting one on their own. Examples: --file, --evopkl, --top-list,
   --read-report, --fast-cycle, --noverbose, and the plotting modifiers
   (--show-std, --show-kids, --plot-name).

Input sources:
    Most actions need an Evolver. It is obtained either by creating a new one
    from a YAML instruction file (--file) or by loading a previously saved
    binary (--evopkl, or the default 'evolver.pkl' in the working directory).

Note on --restart:
    Passing --restart implies --start; this is enforced at the end of
    get_arguments() so the start workflow runs in recovery mode.
"""

import argparse


def get_arguments() -> argparse.Namespace:
    """
    Build the command-line parser and return the parsed arguments.

    Reads ``sys.argv`` and parses it into an ``argparse.Namespace``. As a
    side effect, ``--restart`` forces ``--start`` to True so that a recovered
    session reuses the normal start workflow.

    Returns:
        argparse.Namespace: Parsed command-line arguments, with one attribute
        per option (e.g. ``args.file``, ``args.start``, ``args.fast_cycle``).
    """
    parser = argparse.ArgumentParser(
        description='Command-line interface for the Evo-MD-FE-DB agent.'
    )

    # --- input sources -------------------------------------------------------
    # YAML instruction file used to create a brand-new Evolver/Instructor.
    # When provided (and an action allows new sessions), it takes precedence
    # over loading an existing pickle.
    parser.add_argument(
        '-f', '--file',
        help='YAML file containing the evolution instructions. Used to create '
             'a new session.'
    )

    # Path to a previously saved Evolver pickle. Only needed when the binary
    # is not named 'evolver.pkl' (the default looked up in the working dir).
    parser.add_argument(
        '-evopkl', '--evopkl',
        help='Binary (pickle) Evolver file previously created. Only required '
             'if the pkl file is not named evolver.pkl.'
    )

    # --- configuration inspection (actions) ----------------------------------
    # Print the Instructor default configuration (no session needed) and exit.
    parser.add_argument(
        '-sd', '--show-defaults',
        action='store_true',
        help='Display the Instructor default configuration and exit.'
    )

    # Print the Instructor configuration of the current/loaded session and exit.
    parser.add_argument(
        '-sc', '--show-current',
        action='store_true',
        help='Display the current Instructor configuration and exit.'
    )

    # --- evolver inspection (actions) ----------------------------------------
    # Print a summary of the Evolver state (top sequences, counts) and exit.
    parser.add_argument(
        '-se', '--show-evolver',
        action='store_true',
        help='Display Evolver information and exit.'
    )

    # Modifier for --show-evolver: how many top sequences to list.
    # Expects an integer; default None keeps the Evolver's configured value.
    parser.add_argument(
        '-tl', '--top-list',
        default=None,
        help='Number of sequences to show when --show-evolver is used. '
             'Must be an integer. Modifier for --show-evolver.'
    )

    # Write every sequence (sequence, generation, fitness) to a CSV file
    # (sequences_report.csv) and exit.
    parser.add_argument(
        '-rs', '--report-sequences',
        action='store_true',
        help='Write all sequences to a CSV report (sequences_report.csv) and exit.'
    )

    # --- create evolver (action) ---------------------------------------------
    # Create the Evolver, populate (or read a report), show it, and exit
    # without starting the evolution loop.
    parser.add_argument(
        '-ce', '--create-evolver',
        action='store_true',
        help='Create the Evolver, populate it, display it, and exit without '
             'running the evolution loop.'
    )

    # Modifier for --create-evolver: initialize from an existing CSV report
    # instead of populating from scratch. The CSV must use the columns
    # produced by --report-sequences: sequence,generation,fitness.
    parser.add_argument(
        '-rr', '--read-report',
        default=None,
        help='CSV report (columns: sequence,generation,fitness) used to '
             'initialize the Evolver. Modifier for --create-evolver.'
    )

    # --- run control (actions) -----------------------------------------------
    # Request a graceful stop: sets the Evolver as non-runnable so the loop
    # exits after finishing the current iteration, then saves and exits.
    parser.add_argument(
        '-stop', '--stop-evolver',
        action='store_true',
        help='Stop the Evolver gracefully after the current iteration finishes.'
    )

    # Start the evolution loop. Populates, marks the Evolver as started, and
    # iterates until a termination criterion is met or a stop is requested.
    parser.add_argument(
        '-start', '--start',
        action='store_true',
        help='Start the Evo-MD evolution loop.'
    )

    # Resume an interrupted iteration. Implies --start (enforced below) and
    # enables recovery so already-completed steps of the last iteration are
    # skipped. If the last iteration finished cleanly, use --start instead.
    parser.add_argument(
        '-rstart', '--restart',
        action='store_true',
        help='Resume from an interrupted iteration (implies --start). If the '
             'last iteration finished, use --start instead.'
    )

    # Populate the Evolver from previously computed simulations found in the
    # simulation directory (requires an analyzer), then save and exit.
    parser.add_argument(
        '-pp', '--populate-previous',
        action='store_true',
        help='Populate using results from previous simulations in the '
             'simulation directory, then exit.'
    )

    # Populate the Evolver from json backup files found in the
    # simulation directory, then save and exit.
    parser.add_argument(
        '-bk', '--from-backup',
        action='store_true',
        help='Populate using backup jason files in the '
             'simulation directory, then exit.'
    )

    # --- inspection of sequence lists (action) -------------------------------
    # Print the contents of every Evolver list (current, parents, discarded,
    # excluded) with per-list counts and a grand total, then exit.
    parser.add_argument(
        '-sl', '--show-lists',
        action='store_true',
        help='Display all Evolver sequence lists (current, parents, '
             'discarded, excluded) with counts, and exit.'
    )

    # --- plotting (action + modifiers) ---------------------------------------
    # Plot the fitness evolution across generations and exit.
    parser.add_argument(
        '-pe', '--plot-evolution',
        action='store_true',
        help='Plot the fitness evolution across generations and exit.'
    )

    # Modifier for --plot-evolution: overlay the standard deviation per
    # generation on a secondary axis.
    parser.add_argument(
        '-pstd', '--show-std',
        action='store_true',
        help='Overlay per-generation standard deviation on the plot. '
             'Modifier for --plot-evolution.'
    )

    # Modifier for --plot-evolution: scatter every individual ("kid")
    # fitness value per generation.
    parser.add_argument(
        '-pkids', '--show-kids',
        action='store_true',
        help='Scatter individual sequence fitness values per generation. '
             'Modifier for --plot-evolution.'
    )

    # Modifier for --plot-evolution: output image filename.
    parser.add_argument(
        '-pname', '--plot-name',
        default='evolution.png',
        help='Output filename for the evolution plot (default: evolution.png). '
             'Modifier for --plot-evolution.'
    )

    # --- going back (action) -------------------------------------------------
    # Roll back to the last fully completed generation: drop sequences from
    # later generations, reset the generation counter, re-sort, save and exit.
    parser.add_argument(
        '-lg', '--last-generation',
        action='store_true',
        help='Roll back to the last completed generation, then save and exit.'
    )

    # --- global modifiers ----------------------------------------------------
    # Skip intermediate pickle saves; the Evolver is written to disk only at
    # the end of the run. Faster, but less crash-resistant.
    parser.add_argument(
        '-fc', '--fast-cycle',
        action='store_true',
        help='Save the pkl file only at the end of the run (skips '
             'intermediate saves). Faster but less crash-resistant.'
    )

    # Reduce Evolver console output during the run.
    parser.add_argument(
        '-nvb', '--noverbose',
        action='store_true',
        help='Reduce Evolver console output.'
    )

    args = parser.parse_args()

    # --restart always runs through the start workflow, in recovery mode.
    if args.restart:
        args.start = True

    return args


if __name__ == '__main__':
    pass
    