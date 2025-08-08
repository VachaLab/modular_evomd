# === parserlib.py ===
import argparse


def get_arguments() -> argparse.Namespace:
    """
    Reads command-line arguments and parses to an argparse object.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Command-line interface for the Evo-MD-FE-DB agent.')

    # Input file with sequences and additional options
    parser.add_argument(
        '-f', '--file',
        help='YAML file containing instructions.'
    )

    # Binary input file 
    parser.add_argument(
        '-evopkl', '--evopkl',
        help='Binary evolutionary file previously created. Needed if the pkl file name is not evolver.pkl'
    )

    # Show default configuration in Instructor
    parser.add_argument(
        '-sd', '--show-defaults',
        action='store_true',
        help='Display Instructor default configuration and exit.'
    )

    # Show current configuration in Instructor
    parser.add_argument(
        '-sc', '--show-current',
        action='store_true',
        help='Display Instructor current configuration and exit.'
    )

    # Show  Evolver
    parser.add_argument(
        '-se', '--show-evolver',
        action='store_true',
        help='Display evolver information and exit.'
    )

    # Report all sequences
    parser.add_argument(
        '-rs', '--report-sequences',
        action='store_true',
        help='Report sequences in a CSV file.'
    )

    # just create evolver
    parser.add_argument(
        '-ce', '--create-evolver',
        action='store_true',
        help='Creates evolver, shows evolver and exit.'
    )

    # stop the evolver job after finishing the current iteration
    parser.add_argument(
        '-stop', '--stop-evolver',
        action='store_true',
        help='Stops evolver after finishing the current iteration.'
    )

    # Agent actions
    parser.add_argument(
        '-start', '--start',
        help='Start the evo-md process.',
        action='store_true'
    )
    
    # Recover interrupted session
    parser.add_argument(
        '-rstart', '--restart',
        action='store_true',
        help='Attempt to continue from an interrupted iteration.'
    )

    parser.add_argument(
        '-cm', '--change-method',
        action='store_true',
        help='Interactively change optimization method parameters.'
    )

    parser.add_argument(
        '-pp', '--populate-previous',
        action='store_true',
        help='Populate using information from previous simulations in the simulation directory.'
    )

    parser.add_argument(
        '-in', '--insert-sequence',
        help='Insert a sequence and exit.'
    )

    parser.add_argument(
        '-sl', '--show-lists',
        action='store_true',
        help='Display all the sequences by list in evolver.'
    )

    # test
    parser.add_argument(
        '-test', '--test',
        help='Testing in evo-md.py script.',
        action='store_true'
    )
    
    args = parser.parse_args()

    # If restart is True, force start to be True
    if args.restart:
        args.start = True

    return args


if __name__ == '__main__':
    pass
