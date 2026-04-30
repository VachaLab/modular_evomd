from generator import Generator
from hybrid import Hybrid
from swap import Swap
from faces_mix import FacesMix
from directed_mutations import GroupMutation, HydrophobicityMutation
from sequence import Sequence
import random
from restriction import HmomentRestriction, HindexRestriction

#--------------------------------------------
# Global logging configuration
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s: %(message)s'
    )
#--------------------------------------------

if __name__ == '__main__':
    setup_logging()

    method = Swap()
    restriction1 = HindexRestriction(min=None, max=-7)
    restriction2 = HmomentRestriction(min=5.5, max=None)
    rest_list = [restriction1, restriction2]
    gen = Generator(methods=method, peptide_len=20, extra_mutation=True, extra_mutation_prob=1, restrictions=rest_list)

    seq1 = gen.generate()
    seq2 = gen.generate()
    seq3 = gen.generate(seq1=seq1, seq2=seq2)

    print(seq1)
    print(seq2)
    print(seq3)
