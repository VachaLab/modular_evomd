# === instruction_validators.py ===
"""
Validators for Instruction fields.

Each Validator encapsulates one rule a configuration value must satisfy (type,
membership in a choice set, numeric range, etc.). An Instruction owns a list of
these validators and, on assignment, runs the value through every one of them;
if any returns False, the Instruction discards the value and falls back to its
default.

Design:
    - All validators share the Validator interface: a validate(value) -> bool
      method. (The base signature also lists type_, but concrete validators
      that need a type store it at construction time instead.)
    - "Subtype" validators (SubtypeValidator, SubchoiceValidator) apply a
      per-element rule to every item of a list, via the validate_list decorator.
    - Numeric values are coerced to the expected numeric type before the type
      check, so an int passes where a float is expected (and vice versa).

These validators only report validity; they do not mutate the Instruction or
emit warnings. That responsibility lives in the Instruction descriptor.
"""

from typing import Any, Sequence, Optional
import numbers


def validate_list(func):
    """
    Decorator turning a single-value validator into a list validator.

    Wraps a per-element validate(self, value) method so it requires a list and
    returns True only if every element passes the wrapped check.

    Raises:
        TypeError: If the value passed is not a list.
    """
    def wrapper(self, values):
        if not isinstance(values, list):
            raise TypeError(f"Expected a list, got {type(values).__name__}")
        return all(func(self, value) for value in values)
    return wrapper


class Validator:
    """
    Base class for all validators.

    Subclasses implement validate() to return True when a value satisfies their
    rule. The base also provides numeric coercion shared by type checks.
    """

    def validate(self, type_: type, value: Any) -> bool:
        """Return True if `value` satisfies this validator's rule."""
        raise NotImplementedError

    def adjust_numbers(self, type_, value) -> Any:
        """
        Coerce a numeric value to the target numeric type, leaving others as-is.

        If `value` is a number, return type_(value) (e.g. int->float); otherwise
        return `value` unchanged. This lets a type check accept any number where
        a specific numeric type is expected.
        """
        if isinstance(value, numbers.Number):
            return type_(value)
        else:
            return value


class TypeValidator(Validator):
    """
    Validates that a value is an instance of the expected type.

    Numbers are coerced to the expected type first (see adjust_numbers), so any
    numeric value is accepted where a numeric type is expected.
    """

    def __init__(self, type_: type) -> None:
        self.type_ = type_

    def validate(self, value: Any) -> bool:
        value = self.adjust_numbers(self.type_, value)
        if isinstance(value, self.type_):
            return True
        else:
            return False


class SubtypeValidator(TypeValidator):
    """
    Validates that every element of a list matches the expected element type.

    Reuses TypeValidator's per-element check via the validate_list decorator.
    """

    @validate_list
    def validate(self, value: Any) -> bool:
        return super().validate(value)


class ChoiceValidator(Validator):
    """Validates that a value belongs to an allowed set of choices."""

    def __init__(self, choices: Sequence[Any]) -> None:
        self.choices = choices

    def validate(self, value: Any) -> bool:
        return value in self.choices


class SubchoiceValidator(ChoiceValidator):
    """
    Validates that every element of a list belongs to the allowed choices.

    Reuses ChoiceValidator's membership check via the validate_list decorator.
    """

    @validate_list
    def validate(self, value: Any) -> bool:
        return super().validate(value)


class RangeValidator(Validator):
    """Validates that a value lies within an inclusive [min, max] range."""

    def __init__(self, min_: Any, max_: Any) -> None:
        self.min = min_
        self.max = max_

    def validate(self, value: Any) -> bool:
        return self.min <= value <= self.max


class StrOrChoiceListValidator(ChoiceValidator):
    """
    Validates a value that may be either a single choice or a list of choices.

    Accepts either:
      - a non-empty str that is in choices, or
      - a non-empty list whose every element is in choices.

    Empty list and empty str are rejected, so the caller falls back to the
    default. Used for options like populate_method that accept one name or
    several.
    """

    def validate(self, value: Any) -> bool:
        if isinstance(value, str):
            return value in self.choices
        if isinstance(value, list):
            if not value:                      # empty list -> invalid -> default
                return False
            return all(item in self.choices for item in value)
        return False


if __name__ == '__main__':
    pass
