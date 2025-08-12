# === test.py ===

from sequence import Sequence

if __name__ == '__main__':
    seq = 'LHFKEKYAHGMALASRKNLSKI'
    seq = Sequence(seq)
    
    print(seq.hydrophobic_scale)

