# === calculations.py ===
import logging
logger = logging.getLogger(__name__)

import os
from contextlib import contextmanager
import shutil
import subprocess

@contextmanager
def change_dir(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)

def calculator_method(sequence) -> None:
    """
    receives a sequence and run calculations.
    """
    logger.info(f'Executing sequence {str(sequence)}')
    # list of membranes
    membranes = ['ecoli', 'human']
    for m in membranes:
        logger.info(f'Membrane: {m}')
        try:
            with change_dir(m):
                # write here the code
                source_path = '/storage/brno14-ceitec/shared/softmatter/alejandro/evo-md/mdp/fake_job'
                destination_dir = os.getcwd()
                destination_path = os.path.join(destination_dir, os.path.basename(source_path))
                # copy
                shutil.copy(source_path, destination_path)
                process = subprocess.Popen(["./fake_job"])
        except Exception as e:
            logger.error(f"Execution failed for membrane '{m}': {e}")

def calculator_check(sequence) -> bool:
    """
    receives a sequence and returns True when the computation is ready
    """
    logger.info(f'Checking sequence {str(sequence)}')
    membranes = ['ecoli', 'human']
    true_values = []
    for m in membranes:
        logger.info(f'Membrane: {m}')
        try:
            with change_dir(m):
                # write here the code
                file_stdout = 'result.txt'
                if os.path.exists(file_stdout):
                    logger.info(f'sequence {str(sequence)} with membrane {m} is done')
                    true_values.append(True)
                else:
                    logger.info(f'sequence {str(sequence)} is still running')
                    return False
        except Exception as e:
            logger.error(f"Check failed for membrane '{m}': {e}")
            return False
    return all(true_values) if true_values else False

if __name__ == '__main__':
    pass
