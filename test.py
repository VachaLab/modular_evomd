# === test.py ===

from sequence import Sequence

if __name__ == '__main__':
    seq = 'LHFKEKYAHGMALASRKNLSKI'
    seq = Sequence(seq)
    
    print(seq.hydrophobic_index, seq.hydrophobic_moment)

    seq = 'LKKYKEHARAGLHIANFLSKMS'
    seq = Sequence(seq)
    
    print(seq.hydrophobic_index, seq.hydrophobic_moment)

    seq = 'QKLSRAIAKGKDNLKEYKLNMS'
    seq = Sequence(seq)
    
    print(seq.hydrophobic_index, seq.hydrophobic_moment)
