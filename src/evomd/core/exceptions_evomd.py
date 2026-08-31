# === exceptions_evomd.py ===
"""
Exceptions for evoMD implementation.
"""

class MethodFailedError(Exception):
    """
    Exception raised when an exteral method fails.

    Class attributes:
        cose (int): Code to classify exception.
    """
    def __init__(self, message, code=0):
        # pass message to superclass
        super().__init__(message)
        self.code = code
        
class MethodExistError(Exception):
    """
    Exception raised if an external method is not found.
    """
    def __init__(self, message):
        # pass message to superclass
        super().__init__(message)
    
class EmptyPopulationError(Exception):
    """
    Exception raised if Evolver.sequences is empty when sequences are expected.
    """
    def __init__(self, message):
        # pass message to superclass
        super().__init__(message)
    
class SequenceNotFoundError(Exception):
    """
    Exception raised when a sequence cannot be found in a list.
    """
    def __init__(self, message):
        # pass message to superclass
        super().__init__(message)

class EvolverStateError(Exception):
    """
    Exception raised when Evolver cannot continue.
    """
    def __init__(self, message):
        # pass message to superclass
        super().__init__(message)


if __name__ == '__main__':
    pass
