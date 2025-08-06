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
        if isinstance(value, numbers.Number):
            return type_(value)
        else:
            return value

class TypeValidator(Validator):
    def __init__(self, type_: type):
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

class ChoicesValidator(Validator):
    def __init__(self, choices: Sequence[Any]):
        self.choices = choices

    def validate(self, value: Any) -> bool:
        return value in self.choices

class SubchoicesValidator(ChoicesValidator):
    @validate_list
    def validate(self, value: Any) -> bool:
        return super().validate(value)

class RangeValidator(Validator):
    def __init__(self, min_: Any, max_: Any):
        self.min = min_
        self.max = max_

    def validate(self, value: Any) -> bool:
        return self.min <= value <= self.max
