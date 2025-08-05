# === fields.py ===
import logging
from typing import Any, Type, Optional, Sequence, Dict

logger = logging.getLogger(__name__)

class InstructionField:
    _registry: Dict[str, "InstructionField"] = {}

    def __init__(
            self, type_: Type, default: Any, 
            subtype: Optional[Type] = None, 
            choices: Optional[Sequence[Any]] = None,
            subchoices: Optional[Any] = None,
            range: Optional[Sequence[Any]] = None,
            ):
        self.type_ = type_
        self.default = default
        self.subtype = subtype
        self.choices = choices
        self.subchoices = subchoices
        self.range = range

    def __set_name__(self, owner, name):
        self.name = name
        owner._schema[name] = self
        InstructionField._registry[name] = self

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
            logging.warning(f"Instruction '{self.name}' is not {self.type_.__name__} --> Using default value: {self.default}")
            return False

    def _is_inchoices(self, value):
        # validate choices
        if self.choices is not None and value not in self.choices:
            logging.warning(f"'{value}' is not an option for '{self.name}' --> Using default value: {self.default}")
            return False
        return True


class InstructionInRange(InstructionField):
    def ___set__(self, instance, value):
        # Try trype conversion
        if not self._is_type(value):
            instance.__dict__[self.name] = self.default
            return
        value = self.type_(value)

        # check range
        if not self._is_inrange(value):
            instance.__dict__[self.name] = self.default
            return
        
        instance.__dict__[self.name] = value
    
    def _is_inrange(self, value):
        if self.range is not None and not (self.range[0] <= value <= self.range[1]):
            logging.warning(f"'{value}' is not in the range for '{self.name}' --> Using default value: {self.default}")
            return False
        return True

class InstructionList(InstructionField):
    def ___set__(self, instance, value):
        pass

class InstructionProbability(InstructionField):
    def ___set__(self, instance, value):
        # Try trype conversion
        if not self._is_type(value):
            instance.__dict__[self.name] = self.default
            return
        value = self.type_(value)

        # check range
        if not 0. <= value <= 1.:
            instance.__dict__[self.name] = self.default
            return

        instance.__dict__[self.name] = value

class InstructionChoices(InstructionField):
    def ___set__(self, instance, value):
        pass

if __name__ == '__main__':
    pass
