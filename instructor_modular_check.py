# instructor/fields.py
from typing import Any, Type, Optional

class InstructionField:
    """Descriptor que registra nombre, tipo, sub-tipo (para listas) y valor por defecto."""
    _registry: dict[str, "InstructionField"] = {}

    def __init__(self, type_: Type, default: Any, subtype: Optional[Type] = None):
        self.type_ = type_
        self.subtype = subtype
        self.default = default

    def __set_name__(self, owner, name):
        self.name = name
        owner._schema[name] = self
        InstructionField._registry[name] = self

    def __get__(self, instance, owner):
        return instance.__dict__.get(self.name, self.default)

    def __set__(self, instance, value):
        # Validación de tipo
        if not isinstance(value, self.type_):
            raise TypeError(f"'{self.name}' espera {self.type_.__name__}, no {type(value).__name__}")
        # Si es lista, validar sub-tipo de elementos
        if self.subtype and isinstance(value, list):
            if not all(isinstance(v, self.subtype) for v in value):
                raise TypeError(f"Todos los elementos de '{self.name}' deben ser {self.subtype.__name__}")
        instance.__dict__[self.name] = value


# instructor/instructor.py
import yaml
import logging
from .fields import InstructionField

class Instructor:
    _schema: dict[str, InstructionField] = {}

    # Declaración de instrucciones: basta con añadir un atributo aquí
    threshold   = InstructionField(float, 0.5)
    max_iter    = InstructionField(int,   100)
    use_gpu     = InstructionField(bool,  False)
    model_name  = InstructionField(str,   "default-model")
    layers      = InstructionField(list,  [], subtype=int)

    def __init__(self, yaml_path: str):
        data = self._load_yaml(yaml_path)
        for name, field in self._schema.items():
            raw = data.get(name, None)
            if raw is None:
                # no viene en el YAML → valor por defecto
                setattr(self, name, field.default)
                continue
            try:
                setattr(self, name, raw)
            except TypeError as e:
                logging.warning("%s. Usando valor por defecto '%s'.", e, field.default)
                setattr(self, name, field.default)

    def _load_yaml(self, path: str) -> dict:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}



#--------------------------------------



