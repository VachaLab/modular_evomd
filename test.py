# === test.py ===
from intervals import CircleInterval
from sequence import Sequence
from sequencearray import SequenceArray


def main():
    sequence = Sequence('AAA')
    seq_2 = Sequence('LLLLLL')
    array = SequenceArray([sequence, seq_2])
    
    print(array)
    print(array.extract(sequence))
    print(array)

    interval = CircleInterval(start=-20, end=20, rclosed=True, lclosed=True, degrees=True)
    print(interval)
    for num in range(20):
        val = num * 21
        print(val, interval(val))

if __name__ == '__main__':
    main()

    
