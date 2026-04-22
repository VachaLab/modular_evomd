from sequence import Sequence

class GenMethod:
    def __init__(self) -> None:
        pass
    
    def _to_str(self, seq: "str | Sequence") -> str:
        """Normalizes a sequence identifier to an uppercase string."""
        return str(seq).upper()
    
    def generate(self, seq1: Sequence = None, seq2: Sequence = None) -> str:
        
        return 