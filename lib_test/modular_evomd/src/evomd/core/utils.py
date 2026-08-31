# === utils.py ===
import logging
logger = logging.getLogger(__name__)
import time
import os
import sys
import pickle
from datetime import datetime
import importlib
import importlib.util
import json
from .residue import Residue


class ResidueError(Exception):
    def __init__(self, *args):
        super().__init__(*args)
        
class TimeCount:
    """
    A simple timer class to track the elapsed time from initialization or from the last reset.
    """
    name = 'timecount'

    def __init__(self) -> None:
        """
        Initializes the timer with the current time as the starting point.
        """
        self.ini_time = time.time()

    def restart(self) -> None:
        """
        Resets the starting point of the timer to the current time.
        """
        self.ini_time = time.time()

    def calc_time(self, format: str = "full") -> str:
        """
        Calculates the elapsed time since the timer started or was last reset.

        Parameters:
            format (str): The format of the output time. Options:
                - "s": seconds
                - "m": minutes
                - "h": hours
                - "full": full breakdown as "0 h 0 m 0 s"

        Returns:
            str: The elapsed time in the specified format.

        Raises:
            ValueError: If an unsupported format string is provided.
        """
        elapsed = time.time() - self.ini_time

        if format == "s":
            return f"{int(elapsed)} s"
        elif format == "m":
            return f"{int(elapsed / 60)} m"
        elif format == "h":
            return f"{int(elapsed / 3600)} h"
        elif format == "full":
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            return f"{hours} h {minutes} m {seconds} s"
        else:
            raise ValueError("Invalid format. Choose 's', 'm', 'h', or 'full'.")

# self-explanatory
def current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def exists(filename: str) -> bool:
    """
    Checks whether a file exists in the current working directory.

    Parameters:
        filename (str): Name of the file to check.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    full_path = os.path.join(os.getcwd(), filename)
    file_exists = os.path.isfile(full_path)
    logger.debug(f'Does "{filename}" exists? --> {file_exists}')
    return file_exists


def save_pkl(object, outfile: str) -> None:
    with open(outfile, 'wb') as fo:
        logger.debug(f'Writing {outfile} . . .')
        pickle.dump(object, fo)
        fo.flush()             # Vacía el búfer del intérprete
        os.fsync(fo.fileno())  # Fuerza escritura física a disco

def read_pkl(infile):
    with open(infile, 'rb') as fi:
        logger.info(f'Reading {infile} . . .')
        return pickle.load(fi)
    
def save_json(object, outfile: str) -> None:
    import numpy as np
    # transform object into a dictionary
    dictionary = object
    if not isinstance(dictionary, dict):
        dictionary = object.__dict__
    # remove np.arrays, tuples and sets; also Residue class
    dictionary = {k: v for k, v in dictionary.items() if not isinstance(dictionary[k], (np.ndarray, tuple, set, Residue))}
    # Check for lists where all elements are instances of Residue and remove them
    for k, v in list(dictionary.items()):
        if isinstance(v, list) and any(isinstance(i, Residue) for i in v):
            del dictionary[k]
    # print(dictionary, type(dictionary), '************')
    # save as json file
    with open(outfile, 'w') as jsonfile:
        logger.info(f'Writing {outfile} . . .')
        json.dump(dictionary, jsonfile, indent=4)

def read_json(infile):
    with open(infile, 'r') as fi:
        logger.info(f'Reading {infile} . . .')
        data = json.load(fi)
        return data

def dynamic_import(module_name: str):
    """
    Dynamically imports a module from:
    1. PYTHONPATH
    2. The main script's directory
    3. The current working directory

    :param module_name: Name of the module (without '.py')
    :return: Imported module object
    :raises ImportError: If the module cannot be located
    """
    if not module_name:
        raise ImportError("No module name was provided.")

    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        # Try other directories
        main_dir = os.path.dirname(os.path.abspath(sys.modules['__main__'].__file__))
        # Include here known directories where module can be found
        candidate_paths = [os.getcwd(), main_dir]  

        for path in candidate_paths:
            module_file = os.path.join(path, module_name + ".py")
            if os.path.exists(module_file):
                if path not in sys.path:
                    sys.path.insert(0, path)
                spec = importlib.util.spec_from_file_location(module_name, module_file)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                return module

        raise ImportError(
            f"Module '{module_name}' not found in any tested directory."
        )

def get_module_function(module_name: str, function_name: str, critical: bool = False):
    """
    Imports a module and returns a specified function or attribute.

    :param module_name: Name of the module (without '.py')
    :param function_name: Name of the function or attribute to retrieve
    :param critical: If True, raises error on failure. Otherwise returns None.
    :return: Callable or object from the module
    :raises ImportError or AttributeError
    """
    try:
        module = dynamic_import(module_name)
    except ImportError as e:
        msg = f"Module '{module_name}' could not be imported: {e}"
        if critical:
            logger.error(msg)
            raise
        else:
            logger.warning(msg)
            return None

    if not hasattr(module, function_name):
        msg = f"Function '{function_name}' not found in module '{module_name}'"
        if critical:
            logger.error(msg)
            raise AttributeError(msg)
        else:
            logger.warning(msg)
            return None

    return getattr(module, function_name)

def change_directory(path: str) -> None:
    """
    Change the current working directory to the specified path.

    Parameters:
        path (str): The target directory path.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    if not os.path.exists(path):
        logger.error(f"Directory '{path}' does not exist.")
        raise FileNotFoundError(f"Directory '{path}' does not exist.")
    
    os.chdir(path)
    logger.info(f"Changed working directory to: {path}")

def int_to_roman(n: int) -> str:
    """Converts a positive integer to its Roman numeral representation."""
    values  = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    symbols = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
    result = ''
    for v, s in zip(values, symbols):
        while n >= v:
            result += s
            n -= v
    return result

if __name__ == '__main__':
    pass
