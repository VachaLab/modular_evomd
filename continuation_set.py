# === test.py ===
from instructor import Instructor
from evolver import Evolver
import utils
from manager import Manager

#--------------------------------------------
# Global logging configuration
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s: %(message)s'
    )
#--------------------------------------------

def get_evolver(args, skip_new=False, internal=False):
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

def main():
    logger = logging.getLogger(__name__)
    generations = 26
    evo = 'evolver.pkl'
    evo = utils.read_pkl(evo)
    evo.generations = generations
    evo.started = True
    evo.parent_sequences = [k for k in evo.sequences]
    evo.sequences = []
    evo.populate()
    evo.discarded_sequences = []
    evo.parent_sequences = []
    evo.save_pkl()
    print('Showing Evolver . . .')
    print(evo)
    exit(0)

if __name__ == '__main__':
    setup_logging()
    main()

    
