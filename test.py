# === test.py ===

from sequence import Sequence
from sequencearray import SequenceArray

if __name__ == '__main__':
    sequences = [
        'R'*11 + 'I'*11,
        'LMKRMLMQQKRLGRQQHKAIET',
        'LHFKEKYAHGMALASRKNLSKI',
        'LKKYKEHARAGLHIANFLSKMS',
        'QKLSRAIAKGKDNLKEYKLNMS',
    ]

    sarray = SequenceArray(sequences=[Sequence(k) for k in sequences])
    
    for seq in sarray:
        print(seq, '---')
        print(seq.charge, seq.hydrophobic_moment, seq.hydrophobic_index, seq.hdistribution)
        print('---')
    