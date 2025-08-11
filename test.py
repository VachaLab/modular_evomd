# === test.py ===

from instructor import Instructor
from instruction_validators import *
from sequence import Sequence

if __name__ == '__main__':
    seq = 'AAYNMIVNWLQKLRMIFMIFLHILS'
    seq = Sequence(seq)
    print(seq.sequence)

