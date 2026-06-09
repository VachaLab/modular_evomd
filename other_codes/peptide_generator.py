# === peptide_generator.py ===
#--------------------------------------------
# Global logging configuration
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s: %(message)s'
    )
#--------------------------------------------
from parserlib import get_arguments
from instructor import Instructor
from evolver import Evolver
import utils

def get_evolver(args, skip_new=False, internal=False):
    logger = logging.getLogger(__name__)
    evo_pre = 'evolver.pkl'
    evo = None

    if args.file and utils.exists(args.file) and not skip_new:
        # New session
        inst = Instructor(args.file)
        evo = Evolver(inst)
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

def main():
    """
    Execute the evo-md according to the arguments in parser.
    """

    args = get_arguments()

    # Create evolver and read first sequences
    evo = get_evolver(args)
    back_population = evo.instructor.population
    evo.instructor.population = 100000
    evo.sequences = []
    evo.first_sequences()
    evo.instructor.population = back_population

    # set started = True
    evo.started = True

    # move first sequences to evo.parent_sequences
    evo.parent_sequences = [k for k in evo.sequences]
    back_len_parents = len(evo.parent_sequences)
    evo.sequences = []
    print(evo.instructor)
    
    # populate sequences
    evo.populate()

    # report
    print(f'{len(evo.sequences)} sequences generated from {back_len_parents} previously given')
    print('Showing sequences')
    for seq in evo.sequences:
        print(seq)
    exit(0)

if __name__ == '__main__':
    setup_logging()
    main()
