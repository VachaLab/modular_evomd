# fields.py
from typing import Any, Type, Optional, Sequence, Dict

class InstructionField:
    _registry: Dict[str, "InstructionField"] = {}

    def __init__(
            self, type_: Type, default: Any, 
            subtype: Optional[Type] = None, 
            choices: Optional[Sequence[Any]] = None
            ):
        self.type_ = type_
        self.default = default
        self.subtype = subtype
        self.choices = choices

    def __set_name__(self, owner, name):
        self.name = name
        owner._schema[name] = self
        InstructionField._registry[name] = self

    def __get__(self, instance, owner):
        return instance.__dict__.get(self.name, self.default)

    def __set__(self, instance, value):
        # Validación de tipo principal
        if not isinstance(value, self.type_):
            raise TypeError(f"'{self.name}' espera {self.type_.__name__}, no {type(value).__name__}")

        # Validación de subtipo si es lista
        if self.subtype and isinstance(value, list):
            if not all(isinstance(v, self.subtype) for v in value):
                raise TypeError(f"Todos los elementos de '{self.name}' deben ser {self.subtype.__name__}")

        # Validación de opciones si se especifican
        if self.choices is not None and value not in self.choices:
            raise ValueError(f"'{self.name}' debe ser uno de {self.choices}, no '{value}'")

        instance.__dict__[self.name] = value
