# === calculations.py ===
import logging
logger = logging.getLogger(__name__)
from typing import List
import os
from contextlib import contextmanager
import subprocess
import re
import shutil
JOB_ID_RE = re.compile(r"Submitted batch job (\d+)")
membranes = ['ecoli', 'popc']
back_file = 'job_id.log'
slurm_scripts = '/users/alejanhe/evo-md/slurm_scripts'

@contextmanager
def change_dir(path, execute_dir=None):
    prev = os.getcwd()
    if execute_dir:
        path = os.path.join(path, execute_dir)
        os.makedirs(path, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)

def submit_job(args: List[str]) -> str:
    """
    Ejecuta `sbatch` con los argumentos dados y devuelve el job_id.
    """
    cmd = ["sbatch"] + args
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()

    m = JOB_ID_RE.search(out)
    if proc.returncode != 0 or not m:
        msg = "Fallo al ejecutar sbatch."
        if out:
            msg += f"\nSTDOUT:\n{out}"
        if err:
            msg += f"\nSTDERR:\n{err}"
        raise RuntimeError(msg)

    return m.group(1)

def read_back_file(back_file_name):
    with open(back_file_name, 'r') as f:
        lines = f.readlines()
    job_ids = [k.split()[0] for k in lines]
    return job_ids

def look_for_tag(infile: str, tag: str):
    try:
        with open(infile, 'r') as f:
            lines = f.readlines()
    except:
        return False
    mask = [tag in k for k in lines]
    return any(mask)

def get_confs():
    with open('configurations.txt', 'r') as f:
        line = f.read()
    return line.split()

def calculator_method(sequence) -> None:
    """
    receives a sequence and run calculations.
    """
    logger.info(f'Executing sequence {str(sequence)}')
    # list of membranes
    for m in membranes:
        logger.info(f'Membrane: {m}')

        try:
            with change_dir(m):
                # copy script
                eq_script = os.path.join(slurm_scripts, 'eq_lumi.sh')
                umb_script = os.path.join(slurm_scripts, 'umb_lumi.sh')
                
                shutil.copy2(eq_script, ".")  # 
                shutil.copy2(umb_script, ".")  # 

                id1 = submit_job(["eq_lumi.sh"])
                id2 = submit_job([f"--dependency=afterok:{id1}", "umb_lumi.sh"])

                with open(back_file, 'w') as f:
                    f.write(f'{id1}  ---  Equilibration\n{id2}  ---  Umbrella\n')
                
        except Exception as e:
            logger.error(f"Execution failed for membrane '{m}': {e}")

def calculator_check(sequence) -> bool:
    """
    receives a sequence and returns True when the computation is ready
    """
    logger.info(f'Checking sequence {str(sequence)}')
    
    true_values = []  # all of these values must be True to consider that the job has finished
    logger.debug('-------')
    for m in membranes:
        logger.info(f'Membrane: {m}')

        try:
            with change_dir(m):
                # read back_file
                job_ids = read_back_file(back_file_name=back_file)
                # check if equilibration has finished
                if look_for_tag(f'slurm-{job_ids[0]}.out', 'equX'):
                    logger.info('Equilibration done')
                    true_values.append(True)
                else:
                    logger.info('Waiting for equilibration')
                    return False
                # check all the windows
                # first: read windows
                windows = get_confs()
                windows = [f'Window done: {k}' for k in windows]
                windows_mask = [look_for_tag(f'slurm-{job_ids[1]}.out', k) for k in windows]
                text = f'Windows executed: {len([1 for k in windows_mask if k])}'
                logger.info(text)
                if all(windows_mask):
                    true_values.append(True)
                else:
                    return False
                # did umbrella execution finished?
                if look_for_tag(f'slurm-{job_ids[1]}.out', 'umbCG'):
                    logger.info('Umbrella sampling done')
                    true_values.append(True)
                else:
                    logger('Waiting for umbrella sampling to finish')
                    return False

        except Exception as e:
            logger.error(f"Check failed for membrane '{m}': {e}")
            return False
    logger.debug(f'------- True values: {true_values}')
    return all(true_values)

if __name__ == '__main__':
    pass
