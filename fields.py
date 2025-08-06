# === fields.py ===
import logging
from typing import Any, Type, Optional, Sequence, Dict

logger = logging.getLogger(__name__)

class Instruction:
    _registry: Dict[str, "Instruction"] = {}

    def __init__(
            self, 
            default: Any,
            type_: Type = None,
            subtype: Optional[Type] = None, 
            choices: Optional[Sequence[Any]] = None,
            subchoices: Optional[Any] = None,
            range: Optional[Sequence[Any]] = None,
            ):
        self.default = default
        self.type_ = type_
        self.subtype = subtype
        self.choices = choices
        self.subchoices = subchoices
        self.range = range

    def __set_name__(self, owner, name):
        self.name = name
        owner._schema[name] = self
        Instruction._registry[name] = self

    def __get__(self, instance, owner):
        return instance.__dict__.get(self.name, self.default)

    def __set__(self, instance, value):
        # Try trype conversion
        if not self._is_type(value):
            instance.__dict__[self.name] = self.default
            return
        value = self.type_(value)

        # check options
        if not self._is_inchoices(value):
            instance.__dict__[self.name] = self.default
            return
        
        instance.__dict__[self.name] = value
    
    def _is_type(self, value):
        # Intentar conversión al tipo principal
        try:
            value = self.type_(value)
            return True
        except (ValueError, TypeError) as e:
            self._raise_warning(value)
            return False

    def _is_inchoices(self, value):
        # validate choices
        if self.choices is not None and value not in self.choices:
            self._raise_warning(value)
            return False
        return True
    
    def _is_inrange(self, value):
        if self.range is not None and not (self.range[0] <= value <= self.range[1]):
            self._raise_warning(value)
            return False
        return True
    
    def _raise_warning(self, value) -> None:
        logging.warning(f"'{value}' is not a valid value for '{self.name}' --> Using default value: {self.default}")


class InstructionOdds(Instruction):
    def __set__(self, instance, value):
        # Try trype conversion
        if not self._is_type(value):
            instance.__dict__[self.name] = self.default
            return
        value = self.type_(value)

        # check range
        if not (0. <= value <= 1.):
            instance.__dict__[self.name] = self.default
            return
        
        instance.__dict__[self.name] = value
    

if __name__ == '__main__':
    pass
