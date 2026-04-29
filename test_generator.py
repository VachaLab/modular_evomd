from generator import Generator
from hybrid import Hybrid
from swap import Swap
from faces_mix import FacesMix
from sequence import Sequence

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

    seq1 = Sequence('GSRKTEREFDQNSQKYSRNFSG')
    seq2 = Sequence('YHTEANKNTRKMQRSTQKMRRY')

    method = Swap()
    gen = Generator(methods=method)
    
    new_seq = method.generate(seq1=seq1, seq2=seq2)

    print(seq1)
    print(seq2)
    print(new_seq)
