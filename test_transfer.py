# === test.py ===
from instructor import Instructor
from evolver import Evolver
import utils
from manager import Manager

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
    evo_pre = '/storage/brno12-cerit/projects/softmatter.old/tanguma/evo-md/ecoli_popc_random/evolver.pkl'
    evo_pre = utils.read_pkl(evo_pre)
    # print(evo_pre)
    all_pre_seq = evo_pre.sequences + evo_pre.parent_sequences + evo_pre.discarded_sequences
    print([k.get_mean_fitness() for k in all_pre_seq[:32]])
    new_evo = utils.read_pkl('evolver.pkl')
    new_evo.discarded_sequences = [k for k in all_pre_seq[:32]]
    print(new_evo)
    new_evo.save_pkl()

if __name__ == '__main__':
    main()

    
