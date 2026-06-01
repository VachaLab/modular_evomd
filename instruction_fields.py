# === instruction_fields.py ===
import logging
from typing import Any, Type, Optional, Sequence, Dict, List, Callable
from instruction_validators import *

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
            normalize: Optional[Callable[[Any], Any]] = None,   # <-- nuevo
            choice_list: bool = False,   # <-- nuevo
            ):
        # --- mandatory
        self.type_ = type_
        self.default = default
        # --- optional
        self.subtype = subtype
        self.choices = choices
        self.subchoices = subchoices
        self.range = range
        self.normalize = normalize   # <-- nuevo

        self.validators: List[Validator] = [TypeValidator(self.type_)]

        if self.subtype:
            self.validators.append(SubtypeValidator(self.subtype))
        if self.choices:
            if choice_list:
                self.validators.append(StrOrChoiceListValidator(self.choices))
            else:
                self.validators.append(ChoiceValidator(self.choices))
        if self.subchoices:
            self.validators.append(SubchoiceValidator(self.subchoices))
        if self.range:
            self.validators.append(RangeValidator(self.range[0], self.range[1]))

    def __set_name__(self, owner, name):
        self.name = name
        owner._schema[name] = self
        Instruction._registry[name] = self

    def __get__(self, instance, owner):
        if instance is None:
            return self  # Was ist Introspektion?
        return instance.__dict__.get(self.name, self.default)

    def __set__(self, instance, value):
        if value is None and self.default is None:
            instance.__dict__[self.name] = None
            return
        for validator in self.validators:
            if not validator.validate(value):
                self._warn(value)
                instance.__dict__[self.name] = self.default
                return
        if self.normalize is not None:
            instance.__dict__[self.name] = self.normalize(value)
        else:
            instance.__dict__[self.name] = self.type_(value)
    
    def _warn(self, value):
        logger.warning(f"Invalid value '{value}' for '{self.name}' --> Using default: {self.default}")


class Instruction01(Instruction):  # for probabilities range=[0., 1.]
    def __init__(self, default: float):
        super().__init__(float, default, range=[0.0, 1.0])


if __name__ == '__main__':
    pass
