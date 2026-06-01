# === instructor_validators.py ===
from typing import Any, Sequence, Optional
import numbers


def validate_list(func):
    def wrapper(self, values):
        if not isinstance(values, list):
            raise TypeError(f"Expected a list, got {type(values).__name__}")
        return all(func(self, value) for value in values)
    return wrapper

class Validator:
    def validate(self, type_: type, value: Any) -> bool:
        raise NotImplementedError
    
    def adjust_numbers(self, type_, value) -> Any:
        """
        If value is a number, the method transforms value into the 
        adequate type of number (float or int). otherwise, it returns
        the original value.
        """
        if isinstance(value, numbers.Number):
            return type_(value)
        else:
            return value

class TypeValidator(Validator):
    """
    Validates type.
    If value is a number, it is always transformed into the adequate type.
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
    @validate_list
    def validate(self, value: Any) -> bool:
        return super().validate(value)

class ChoiceValidator(Validator):
    def __init__(self, choices: Sequence[Any]) -> None:
        self.choices = choices

    def validate(self, value: Any) -> bool:
        return value in self.choices

class SubchoiceValidator(ChoiceValidator):
    @validate_list
    def validate(self, value: Any) -> bool:
        return super().validate(value)

class RangeValidator(Validator):
    def __init__(self, min_: Any, max_: Any) -> None:
        self.min = min_
        self.max = max_

    def validate(self, value: Any) -> bool:
        return self.min <= value <= self.max

class StrOrChoiceListValidator(ChoiceValidator):
    """
    Accepts either:
      - a non-empty str that is in choices, or
      - a non-empty list whose every element is in choices.
    Empty list / empty str are rejected (caller falls back to default).
    """
    def validate(self, value: Any) -> bool:
        if isinstance(value, str):
            return value in self.choices
        if isinstance(value, list):
            if not value:                      # empty list -> invalid -> default
                return False
            return all(item in self.choices for item in value)
        return False

