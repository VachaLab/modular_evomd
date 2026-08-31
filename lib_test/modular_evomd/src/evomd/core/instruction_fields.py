# === instruction_fields.py ===
"""
Instruction: a validating descriptor for Instructor configuration options.

Each configuration option on the Instructor class is declared as an Instruction
instance. Instruction is a Python descriptor, so it intercepts attribute reads
and writes on the owning instance and applies validation transparently.

Lifecycle:
    1. Declaration: `optimize = Instruction(str, 'maximize', choices=...)` at
       class body time creates the descriptor and, via __set_name__, registers
       it in the owner's `_schema` and in the global `Instruction._registry`.
    2. Assignment (__set__): the incoming value is run through every validator
       built from the constructor arguments. If any fails, a warning is logged
       and the option falls back to its default. A valid value is stored, after
       optional normalization or type coercion.
    3. Read (__get__): returns the stored value, or the default if unset.

Validators are assembled from the constructor arguments (type_, subtype,
choices, subchoices, range) and live in instruction_validators. `normalize` and
`choice_list` support options that accept either a single value or a list (e.g.
populate_method).
"""

import logging
from typing import Any, Type, Optional, Sequence, Dict, List, Callable
from .instruction_validators import *

logger = logging.getLogger(__name__)


class Instruction:
    """
    A validating descriptor for a single Instructor configuration option.

    Declares the option's expected type and default, plus optional validation
    (element subtype, allowed choices, numeric range). On assignment, the value
    is validated and either stored or replaced by the default with a warning.

    Class attributes:
        _registry: Global mapping of option name -> Instruction, populated as
            descriptors are bound to their owning class.
    """

    _registry: Dict[str, "Instruction"] = {}

    def __init__(
            self, 
            type_: Type,
            default: Any,
            subtype: Optional[Type] = None, 
            choices: Optional[Sequence[Any]] = None,
            subchoices: Optional[Any] = None,
            range: Optional[Sequence[Any]] = None,
            normalize: Optional[Callable[[Any], Any]] = None,
            choice_list: bool = False,
            ):
        """
        Configure the option and build its validator chain.

        Args:
            type_ (Type): Expected type of the value (mandatory).
            default (Any): Value used when the option is absent or invalid (mandatory).
            subtype (Type, optional): If set, the value is a list and every
                element must be of this type.
            choices (Sequence, optional): Allowed values. Combined with
                choice_list, allows a single choice or a list of choices.
            subchoices (optional): Allowed values for each element of a list.
            range (Sequence, optional): Inclusive [min, max] bounds for numeric values.
            normalize (Callable, optional): Transform applied to a valid value
                before storing (used instead of plain type coercion).
            choice_list (bool): If True, use StrOrChoiceListValidator so the
                value may be a single choice or a list of choices.
        """
        # --- mandatory
        self.type_ = type_
        self.default = default
        # --- optional
        self.subtype = subtype
        self.choices = choices
        self.subchoices = subchoices
        self.range = range
        self.normalize = normalize

        # Build the validator chain. Order matters only for which warning fires
        # first; a value must pass all of them to be accepted.
        self.validators: List[Validator] = [TypeValidator(self.type_)]
        if self.subtype:
            self.validators.append(SubtypeValidator(self.subtype))
        if self.choices:
            if choice_list:
                # value may be a single choice or a list of choices
                self.validators.append(StrOrChoiceListValidator(self.choices))
            else:
                self.validators.append(ChoiceValidator(self.choices))
        if self.subchoices:
            self.validators.append(SubchoiceValidator(self.subchoices))
        if self.range:
            self.validators.append(RangeValidator(self.range[0], self.range[1]))

    def __set_name__(self, owner, name):
        """
        Bind the descriptor to its owner: record its name and register it.

        Called automatically when the owning class is defined. Stores the
        option in the owner's `_schema` and in the global `_registry`.
        """
        self.name = name
        owner._schema[name] = self
        Instruction._registry[name] = self

    def __get__(self, instance, owner):
        """Return the stored value, the default if unset, or the descriptor itself for class access."""
        if instance is None:
            return self  # accessed on the class, not an instance: return the descriptor
        return instance.__dict__.get(self.name, self.default)

    def __set__(self, instance, value):
        """
        Validate and store an assigned value, falling back to the default.

        A None value is allowed only when the default is also None (the option
        is genuinely optional). Otherwise the value must pass every validator;
        if not, the default is stored and a warning is logged. A valid value is
        normalized (if a normalizer was given) or coerced to type_.
        """
        # Allow explicit None only when None is the declared default.
        if value is None and self.default is None:
            instance.__dict__[self.name] = None
            return
        # Reject on the first failing validator -> store default.
        for validator in self.validators:
            if not validator.validate(value):
                self._warn(value)
                instance.__dict__[self.name] = self.default
                return
        # Accepted: normalize if possible, else coerce to the declared type.
        if self.normalize is not None:
            instance.__dict__[self.name] = self.normalize(value)
        else:
            instance.__dict__[self.name] = self.type_(value)
    
    def _warn(self, value):
        """Log a warning that an invalid value was replaced by the default."""
        logger.warning(f"Invalid value '{value}' for '{self.name}' --> Using default: {self.default}")


class Instruction01(Instruction):
    """Convenience Instruction for probabilities: a float constrained to [0.0, 1.0]."""

    def __init__(self, default: float):
        super().__init__(float, default, range=[0.0, 1.0])


if __name__ == '__main__':
    pass
