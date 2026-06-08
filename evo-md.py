# === evo-md.py ===
#--------------------------------------------
# Global logging configuration
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
#--------------------------------------------
from parserlib import get_arguments
from instructor import Instructor
from evolver import Evolver
import utils


def iterate_evolver(evo, fast_cycle=False):
    evo.iterate()
    if not fast_cycle:
        evo.save_pkl()  # save again

def handling_evolver(evo, fast_cycle=False):
    evo.sort_sequences()
    if not fast_cycle:
        evo.save_pkl()  # save after sorting
    evo.populate()
    if not fast_cycle:
        evo.save_pkl()  # save again

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
    """
    Execute the evo-md according to the arguments in parser.
    """

    logger = logging.getLogger(__name__)
    args = get_arguments()

    # what do I have to do?
    if args.show_defaults:
        logger.info('Showing Instructor default configutation . . .')
        print(Instructor(args.file))
        exit(0)

    elif args.show_current:
        evo = get_evolver(args, skip_new=True)
        logger.info('Showing Instructor configutation . . .')
        # evo.instructor.show_configuration()
        print(evo.instructor)
        exit(0)

    elif args.show_evolver:
        evo = get_evolver(args, skip_new=True)
        if args.top_list:
            evo.instructor.top_list = int(args.top_list)
        logger.info('Showing Evolver . . .')
        print(evo)
        exit(0)

    elif args.report_sequences:
        evo = get_evolver(args, skip_new=True)
        logger.info('Creating Evolver report . . .')
        evo.report_sequences()
        exit(0)

    elif args.create_evolver:
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
        evo = get_evolver(args)
        evo.read_previous()
        evo.save_pkl()
        exit()
    
    elif args.stop_evolver:
        # changes evo.runnable to false
        evo = get_evolver(args, skip_new=True)
        logger.info('Stopping evolver . . .')
        evo.runnable = False
        evo.save_pkl()
        exit(0)

    elif args.start:
        evo = get_evolver(args)

        # set fast_cycle
        evo.fast_cycle = args.fast_cycle
        # verbose
        if args.noverbose:
            evo.verbose = False

        # populate sequences
        evo.populate()
        evo.started = True
        if not args.fast_cycle:
            evo.save_pkl()

        while evo.runnable:
            # --- reload evolver ---
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
    
    elif args.show_lists:
        evo = get_evolver(args)
        # show sequences
        logger.info('Showing sequences\n')
        print('Evolver.sequences')
        total = 0
        i = 0
        for seq in evo.sequences:
            print(f'{seq}')
            i += 1
        print(f'--> {i} sequences found\n')
        total += i
        # parents
        print('Evolver.parent_sequences')
        i = 0
        for seq in evo.parent_sequences:
            print(f'{seq}')
            i += 1
        print(f'--> {i} sequences found\n')
        total += i
        # discarded
        print('Evolver.discarded_sequences')
        i = 0
        for seq in evo.discarded_sequences:
            print(f'{seq}')
            i += 1
        print(f'--> {i} sequences found\n')
        total += i
        # excluded sequences
        print('Evolver.excluded_sequences')
        i = 0
        for seq in evo.excluded_sequences:
            print(f'{seq}')
            i += 1
        print(f'--> {i} sequences found\n')
        total += i
        # total
        print(f'Total: {total} sequences')
        exit(0)
    
    elif args.plot_evolution:
        evo = get_evolver(args)
        # plot_evolution
        evo.plot_evolution(show_std=args.show_std, show_kids=args.show_kids, name=args.plot_name)
        exit(0)

    elif args.last_generation:
        evo = get_evolver(args, skip_new=True)
        logger.info('Going back to the last completed generation . . .')
        evo.revert_last_generation()
        evo.save_pkl()  # save updated state
        print(evo)
        exit(0)

if __name__ == '__main__':
    setup_logging()
    main()
