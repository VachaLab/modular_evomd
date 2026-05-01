from generator import Generator
from hybrid import Hybrid
from swap import Swap
from faces_mix import FacesMix
from directed_mutations import GroupMutation, HydrophobicityMutation
from sequence import Sequence
import random
from restriction import HmomentRestriction, HindexRestriction
from peptide_viewer import plot_sequence

#--------------------------------------------
# Global logging configuration
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
#--------------------------------------------

if __name__ == '__main__':
    setup_logging()

    method1 = Swap()
    method2 = FacesMix()
    restriction1 = HindexRestriction(min=None, max=-7)
    restriction2 = HmomentRestriction(min=5.5, max=None)
    rest_list = [restriction2, restriction1, ]
    gen = Generator(methods=[method1, method2], peptide_len=20, extra_mutation=True, extra_mutation_prob=1, restrictions=rest_list)

    seq1 = gen.generate(verbose=True)
    seq2 = gen.generate(verbose=True)
    seq3 = gen.generate(seq1=seq1, seq2=seq2, verbose=True)

    # plot_sequence(seq3, print_hm=True, show_sections=True)

    print(seq1)
    print(seq2)
    print(seq3)
