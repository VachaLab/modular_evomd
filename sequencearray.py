# === sequencearray.py ===
from collections.abc import MutableSequence
from typing import Any, Type, Optional, Sequence, Dict, List
from sequence import Sequence
import logging

logger = logging.getLogger(__name__)


class SeqArrayField:
    _registry: Dict[str, "SeqArrayField"] = {}

    def __init__(self):
        self.type_ = list
        self.default = []
        self.subtype_ = Sequence
    
    def __set_name__(self, owner, name):
        self.name = name
        owner._schema[name] = self
        SeqArrayField._registry[name] = self

    def __get__(self, instance, owner):
        if instance is None:
            return self  # Was ist Introspektion?
        return instance.__dict__.get(self.name, self.default)

    def __set__(self, instance, value):
        if not isinstance(value, self.type_):
            self._warn(value)
            instance.__dict__[self.name] = self.default
            return
        true_list = [isinstance(k, self.subtype_) if len(value) > 0 else True for k in value]
        if not all(true_list):
            self._warn(value)
            instance.__dict__[self.name] = self.default
            return
        instance.__dict__[self.name] = self.type_(value)
    
    def _warn(self, value):
        logger.warning(f"Invalid value '{value}' for '{self.name}' --> Using default: {self.default}")



class SequenceArray(MutableSequence):
    _schema: Dict[str, SeqArrayField] = {}

    _sequences = SeqArrayField()

    def __init__(self, sequences: List[Sequence] = None) -> None:
        if sequences is None:
            setattr(self, '_sequences', self._schema['_sequences'].default)
        try:
            setattr(self, '_sequences', sequences)
        except TypeError as e:
            logging.warning(f'Default value in {"_sequences"}', e, self._schema['_sequences'].default)
            setattr(self, '_sequences', self._schema['_sequences'].default)
    
    # --- special methods ---
    def __str__(self):
        return f'{self._sequences}'
    
    def __iter__(self):
        return iter(self._sequences)
    
    def __getitem__(self, index):
        return self._sequences[index]
    
    def __setitem__(self, index, value):
        self._sequences[index] = value
    
    def __delitem__(self, index):
        del self._sequences[index]
    
    def __len__(self):
        return len(self._sequences)
    
    def insert(self, index, value):
        self._sequences.insert(index, value)
    
    def append(self, value):
        self._sequences.append(value)
    
    # --- alignment ---
    def align_hm(self):
        pass


if __name__ == '__main__':
    pass
