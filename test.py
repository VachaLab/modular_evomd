# === test.py ===
import utils
from sequence import Sequence
from sequencearray import SequenceArray


def main():
    sequence = Sequence('AAA')
    seq_2 = Sequence('LLLLLL')
    array = SequenceArray([sequence, seq_2])
    
    print(array)
    print(array.extract(sequence))
    print(array)

if __name__ == '__main__':
    main()

    
