# === fields.py ===
import logging
from typing import Any, Type, Optional, Sequence, Dict, List
from .instruction_validators import *

logger = logging.getLogger(__name__)

class Instruction:
    _registry: Dict[str, "Instruction"] = {}

    def __init__(
            self, 
            type_: Type,
            default: Any,
            subtype: Optional[Type] = None, 
            choices: Optional[Sequence[Any]] = None,
            subchoices: Optional[Any] = None,
            range: Optional[Sequence[Any]] = None,
            ):
        # --- mandatory
        self.type_ = type_
        self.default = default
        # --- optional
        self.subtype = subtype
        self.choices = choices
        self.range = range

        self.validators: List[Validator] = [TypeValidator(type_)]
        if self.subtype:
            self.validators.append(TypeValidator(subtype))
        if choices:
            self.validators.append(ChoicesValidator(choices))
        if range:
            self.validators.append(RangeValidator(range[0], range[1]))

    def __set_name__(self, owner, name):
        self.name = name
        owner._schema[name] = self
        Instruction._registry[name] = self

    def __get__(self, instance, owner):
        if instance is None:
            return self  # introspeccion?
        return instance.__dict__.get(self.name, self.default)

    def __set__(self, instance, value):
        for validator in self.validators:
            if not validator.validate(value):
                self._warn(value)
                instance.__dict__[self.name] = self.default
                return
        instance.__dict__[self.name] = self.type_(value)
    
    def _warn(self, value):
        logger.warning(f"Invalid value '{value}' for '{self.name}' --> Using default: {self.default}")


class Instruction01(Instruction):  # for probabilities range=[0., 1.]
    def __init__(self, default: float):
        super().__init__(float, default, range=[0.0, 1.0])


if __name__ == '__main__':
    pass
