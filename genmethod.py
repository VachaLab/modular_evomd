from sequence import Sequence

class GenMethod:
    def __init__(self, mut_aa: str, peptide_len: int, **kwargs):
        self.mut_aa = mut_aa
        self.peptide_len = peptide_len

    def generate(self, seq1: Sequence = None, seq2: Sequence = None) -> str:
        raise NotImplementedError