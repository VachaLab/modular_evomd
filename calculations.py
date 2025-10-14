# === calculations.py ===
import logging
logger = logging.getLogger(__name__)

import os
from contextlib import contextmanager
import shutil
import subprocess

num_windows = 14  # change here the numbre of windows for umbrella sampling

bash_equ = """
#!/bin/bash

module add anaconda3
source activate /storage/brno14-ceitec/shared/softmatter/alejandro/env/aht-env
module add gromacs:2021.4-plumed
export GMXLIB=/storage/brno14-ceitec/shared/softmatter/alejandro/forcefields

# Runtime variables
NTOMP=8			# use 8 OMP threads. Can try 16 with 2 GPUs - but system should be > 100-150k atoms
NTMPI=1			# use 1 MPI rank
NPME=1			# with 1 MPI rank, npme=1
PIN='off'		# use only mpirun option --bind-to
PINSTRIDE=1		# not used as PIN=off
NB='gpu'		# always compute non-bonded interactions on the GPU
BONDED='gpu'	# if 8 cores/GPU: bonded on GPU, if more, bonded might have to be on CPU
UPDATE='cpu'	# if the system allows it, try it
DLB='yes'		# always use dynamic load balancing
BINDTO='none'	# always - there are possibly better settings which depend on the number of threads used

# Environment variables - mainly not very necessary as most jobs will be ran on 1 GPU and 1 MPI rank.
export GMX_ENABLE_DIRECT_GPU_COMM=1 	# force GPU direct communication - from gmx 2020
export GMX_GPU_DD_COMMS=1				# halo exchange - from gmx 2020
export GMX_GPU_PME_PP_COMMS=1 			# PP-PME ranks can communicate direct GPU-GPU communication
export OMP_NUM_THREADS=${NTOMP}

# this should be defined by calculator method
ROOTDIR=CHANGEHERE

MDP=/storage/brno14-ceitec/shared/softmatter/alejandro/evo-md/mdp
SCRP=/storage/brno14-ceitec/shared/softmatter/alejandro/evo-md/scripts

# minimize energy
gmx_mpi grompp -f ${MDP}/minimization.mdp -c system.pdb -r system.pdb -p system.top -n system.ndx -o minimization.tpr &> grompp_minimization.info
gmx_mpi mdrun -ntomp ${NTOMP} -s minimization.tpr -deffnm minimization -cpi minimization.cpt -v &> mdrun_minimization.info
if [ $? -ne 0 ]; then exit 2; fi  # exit if this fails

# equilibration_1
gmx_mpi grompp -f ${MDP}/equilibration_1.mdp -c minimization.gro -r minimization.gro -p system.top -n system.ndx -o equilibration_1.tpr &> grompp_equilibration_1.info
mpirun -np ${NTMPI} --bind-to ${BINDTO} gmx_mpi mdrun -dlb ${DLB} -pin ${PIN} -pinstride ${PINSTRIDE} -ntomp ${NTOMP} -nb ${NB} -tunepme -bonded ${BONDED} -update ${UPDATE} -s equilibration_1.tpr -deffnm equilibration_1 -cpi equilibration_1.cpt -v &> mdrun_equilibration_1.info

# equilibration_2
gmx_mpi grompp -f ${MDP}/equilibration_2.mdp -c equilibration_1.gro -r equilibration_1.gro -p system.top -n system.ndx -o equilibration_2.tpr &> grompp_equilibration_2.info
mpirun -np ${NTMPI} --bind-to ${BINDTO} gmx_mpi mdrun -dlb ${DLB} -pin ${PIN} -pinstride ${PINSTRIDE} -ntomp ${NTOMP} -nb ${NB} -tunepme -bonded ${BONDED} -update ${UPDATE} -s equilibration_2.tpr -deffnm equilibration_2 -cpi equilibration_2.cpt -v &> mdrun_equilibration_2.info
 
# compute restraints
python3 ${SCRP}/restraint_geom.py equilibration_2.gro &> restraints.info

# equilibration_3
gmx_mpi grompp -f ${MDP}/equilibration_3.mdp -c equilibration_2.gro -r restraints.gro -p system.top -n system.ndx -o equilibration_3.tpr &> grompp_equilibration_3.info
mpirun -np ${NTMPI} --bind-to ${BINDTO} gmx_mpi mdrun -dlb ${DLB} -pin ${PIN} -pinstride ${PINSTRIDE} -ntomp ${NTOMP} -nb ${NB} -tunepme -bonded ${BONDED} -update ${UPDATE} -s equilibration_3.tpr -deffnm equilibration_3 -cpi equilibration_3.cpt -v &> mdrun_equilibration_3.info
 
# equilibration_4
gmx_mpi grompp -f ${MDP}/equilibration_4.mdp -c equilibration_3.gro -r restraints.gro -p system.top -n system.ndx -o equilibration_4.tpr &> grompp_equilibration_4.info
mpirun -np ${NTMPI} --bind-to ${BINDTO} gmx_mpi mdrun -dlb ${DLB} -pin ${PIN} -pinstride ${PINSTRIDE} -ntomp ${NTOMP} -nb ${NB} -tunepme -bonded ${BONDED} -update ${UPDATE} -s equilibration_4.tpr -deffnm equilibration_4 -cpi equilibration_4.cpt -v &> mdrun_equilibration_4.info
 
# equilibration_5
gmx_mpi grompp -f ${MDP}/equilibration_5.mdp -c equilibration_4.gro -r restraints.gro -p system.top -n system.ndx -o equilibration_5.tpr &> grompp_equilibration_5.info
mpirun -np ${NTMPI} --bind-to ${BINDTO} gmx_mpi mdrun -dlb ${DLB} -pin ${PIN} -pinstride ${PINSTRIDE} -ntomp ${NTOMP} -nb ${NB} -tunepme -bonded ${BONDED} -update ${UPDATE} -s equilibration_5.tpr -deffnm equilibration_5 -cpi equilibration_5.cpt -v &> mdrun_equilibration_5.info
if [ $? -ne 0 ]; then exit 2; fi  # exit if this fails
 
# pulling
gmx_mpi grompp -f ${MDP}/pull.mdp -c equilibration_5.gro -r restraints.gro -p system.top -n system.ndx -o pull.tpr &> grompp_pull.info
mpirun -np ${NTMPI} --bind-to ${BINDTO} gmx_mpi mdrun -dlb ${DLB} -pin ${PIN} -pinstride ${PINSTRIDE} -ntomp ${NTOMP} -nb ${NB} -tunepme -bonded ${BONDED} -update ${UPDATE} -s pull.tpr -deffnm pull -cpi pull.cpt -v &> mdrun_pull.info
if [ $? -ne 0 ]; then exit 2; fi  # exit if this fails
 
# extract confs and set windows
mkdir -p "${ROOTDIR}/confs"
echo "System" | gmx trjconv -s pull.tpr -f pull.xtc -o ${ROOTDIR}/confs/conf.gro -sep

# select confs
python3 ${SCRP}/select_confs_pullx.py pull_pullx.xvg NUMWINDOWS 5.2 &> configurations.txt
read -r line < configurations.txt

# run windows
base_dir=$(pwd)
for num in $line; do
    mkdir -p ${ROOTDIR}/conf${num}
    cp ${SCRP}/umbCG ${ROOTDIR}/conf${num}
    sed -i "s|RAIZumb|${ROOTDIR}|" umbCG
    cp system.*  ${ROOTDIR}/conf${num}
    cp restraints.gro  ${ROOTDIR}/conf${num}
    cp ${ROOTDIR}/confs/conf${num}.gro ${ROOTDIR}/conf${num}/conf.gro
    cd ${ROOTDIR}/conf${num}
    psubmit gpu umbCG ncpus=8,ngpus=1,mem=10gb,walltime=1d -y
    cd ${base_dir}
done

# print a message if the script goes until the end
echo "equX"
"""

def write_script(new_path, script=None, name='equX'):
    # modify ROOTDIR
    bash_script = script.replace("CHANGEHERE", new_path)
    bash_script = bash_script.replace("NUMWINDOWS", f'{num_windows}')
    # write script
    with open(name, 'w') as fo:
        fo.write(bash_script)
    logger.info(f'Bash script {name} saved in {new_path}')

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

def calculator_method(sequence) -> None:
    """
    receives a sequence and run calculations.
    """
    logger.info(f'Executing sequence {str(sequence)}')
    # list of membranes
    membranes = ['ecoli', 'popc']
    for m in membranes:
        logger.info(f'Membrane: {m}')
        root_dir = os.path.join(os.getcwd(), m)
        bash_name = 'equX'
        try:
            with change_dir(m):
                # write script
                write_script(root_dir, script=bash_equ, name=bash_name)
                command = [
                    'psubmit',
                    'gpu',
                    f'{bash_name}',
                    'ncpus=8,ngpus=1,mem=10gb,walltime=1d',
                    '-y'
                ]
                with open('evo_submit.log', 'w') as logfile:
                    proceso = subprocess.run(command, stdout=logfile, stderr=subprocess.STDOUT, text=True)
        except Exception as e:
            logger.error(f"Execution failed for membrane '{m}': {e}")

def calculator_check(sequence) -> bool:
    """
    receives a sequence and returns True when the computation is ready
    """
    logger.info(f'Checking sequence {str(sequence)}')
    membranes = ['ecoli', 'popc']
    true_values = []  # all of these values must be True to consider that the job has finished
    logger.debug('-------')
    for m in membranes:
        logger.info(f'Membrane: {m}')
        root_dir = os.path.join(os.getcwd(), m)
        try:
            with change_dir(m):
                # This directory should be: iter_?/membrane
                # check if equX finished
                file_stdout = 'equX.stdout'
                if os.path.exists(file_stdout):
                    with open(file_stdout, 'r') as f:
                        content = f.readlines()
                        for line in content:
                            if "equX" in line:
                                logger.info(f'sequence {str(sequence)} with membrane {m}: windows were executed')
                # check if windows finished
                file_confs = 'configurations.txt'
                finished_windows = []
                if os.path.exists(file_confs):
                    logger.debug(f'checker found {file_confs} file *****')
                    nums = []
                    with open(file_confs, 'r') as num_list:
                        nums = num_list.readlines()[0].strip().split()
                    for dir in nums:
                        try:
                            with open(f'conf{dir}/umbCG.stdout', 'r') as outfile:
                                lines = outfile.readlines()
                            line = [k for k in lines if 'umbCG' in k][0]
                            finished_windows.append('umbCG' in line)
                        except:
                            finished_windows.append(False)
                            continue
                    logger.info(f'Number of finished windows: {len([k for k in finished_windows if k])}')
                if len([k for k in finished_windows if k]) == num_windows:
                    logger.info(f'All the windows are finished in membrane {m}')
                    true_values.append(all(finished_windows))
                else:
                    logger.info(f'sequence {str(sequence)} is still running in membrane {m}')
                    true_values.append(False)
        except Exception as e:
            logger.error(f"Check failed for membrane '{m}': {e}")
            return False
    logger.debug(f'------- True values: {true_values}')
    return all(true_values)

if __name__ == '__main__':
    pass
