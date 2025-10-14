# === contructor.py ===
# logging instead of print
import logging
logger = logging.getLogger(__name__)

import argparse
import os
import numpy as np
try:
    from pymol import cmd
except ImportError:
    logger.error("PyMOL not found. Ensure it is installed and usable in this environment.")
    raise
import subprocess
import tempfile
import random
from contextlib import contextmanager


class Model:
    name = 'constructor'

    def __init__(self, pdb, top, setmodel=True) -> None:
        self.name = 'model'
        self.pdb = pdb
        self.top = top
        # only lipids here defined will be taken into account
        # in the future this information will come from force field files
        # W and WF are neutral particles
        self.lipids = {
            'POPE':  0.,
            'POPG': -1.,
            'POPC':  0.,
            'POSM':  0.,
            'CDL2': -2.,
            'POPS': -1.,
        }
        self.aminoacids = {
            'ALA', 'ACE', 'NME', 'ARG', 'ASN', 'ASP',
            'CYS', 'GLN', 'QLN', 'GLU', 'GLY', 'HISD',
            'HISE', 'HISH', 'HSE', 'ILE', 'LYS', 'LEU',
            'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 
            'TYR', 'VAL',
        }
        if setmodel:
            self.atoms, self.molecules = self.set_model()

    def set_model(self):
        atoms = []
        for line in self.pdb:
            if len(line) < 60:
                continue
            if line[:4] == 'ATOM':
                idx = int(line[6:11])
                name = line[11:16]
                vec = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                res = line[17:21]
                atoms.append(Atom(idx, name, vec, res))
        # if [ molecules ] tag is found
        mol_section = find_lines(self.top, 'molecules')
        types = {}
        if len(mol_section) > 0:
            for line in self.top[mol_section[-1]:]:
                cad = line.split()
                try:
                    molname = cad[0]
                    number = int(cad[1])
                    if molname not in types:
                        types[molname] = number
                    else:
                        types[molname] += number
                except:
                    continue
        return atoms, types
    
    def compute_charge(self):
        # charges will be appended into the next list
        charges = []
        # first read charges available in the topology file after [ atoms ] tag
        # stop reading when new tag is found
        atoms_tag = find_lines(self.top, 'atoms')
        if len(atoms_tag) > 0:
            all_tags = find_lines(self.top, '[')
            atoms_tag = atoms_tag[0]
            next_tag = 0
            for i in all_tags:
                if i > atoms_tag:
                    next_tag = i
                    break
            for line in self.top[atoms_tag:next_tag]:
                try:
                    charges.append(float(line.split()[6]))
                except:
                    continue
        # continue reading the [ molecules ] section
        molecules_tag = find_lines(self.top, 'molecules')
        if len(molecules_tag) > 0:
            molecules_tag = molecules_tag[-1]
            for line in self.top[molecules_tag:]:
                try:
                    lipid_name = line.split()[0]
                    lipid_number = int(line.split()[1])
                    if lipid_name not in self.lipids:
                        continue
                    charges.append(self.lipids[lipid_name] * lipid_number)
                except:
                    continue
        return sum(charges)
    
    def add_ions(self, conc=0.15, water_ratio=4):
        # count number of W and WF particles (water_ratio=4 water molecules per W/WF particle)
        # water number density rhoN = 3.343e-02 A-3
        # 1st: select all the water and anti-freeze particles
        waters = []
        for atom in self.atoms:
            if atom.name.strip() not in ['W', 'WF']:
                continue
            waters.append(atom)
        if len(waters) == 0:
            logger.info('No water found in the system')
            return None
        else:
            prev_water_num = len(waters)
            logger.info(f'W and WF particles found: {prev_water_num}')
        # 2nd: compute number of ions
        rhoN_ions = conc * 6.022 * 1E-04  # number of ions per A3
        iw_ratio = rhoN_ions / 3.343e-02  # ions / water ratio
        ion_num = int(len(waters) * iw_ratio)
        charge = int(self.compute_charge())
        if charge == 0:
            cl_num = ion_num
            na_num = ion_num
        elif charge < 0:
            cl_num = ion_num
            na_num = ion_num + abs(charge)
        elif charge > 0:
            cl_num = ion_num + abs(charge)
            na_num = ion_num
        else:
            logger.info('wtf happend?')
            exit(1)
        # 3rd: randomly change names
        # the CL- and NA+ positions are selected randomly from waters list
        to_cl = random.sample(waters, cl_num)
        waters = [k for k in waters if k not in to_cl]
        to_na = random.sample(waters, na_num)
        waters = [k for k in waters if k not in to_na]
        logger.info(f'To CL: {len(to_cl)}; to NA: {len(to_na)}; remaining W/WF: {len(waters)}')
        logger.info(f'Does it make sense?: {len(to_cl) + len(to_na) + len(waters)} = {prev_water_num}?')
        # change the name of the atoms 
        for a in to_cl:
            a.name = 'CL-'
            a.residue = 'ION'
        for a in to_na:
            a.name = 'NA+'
            a.residue = 'ION'
        # 4th: create new files
        # === PDB ===
        new_pdb = []
        for line in self.pdb:
            try:
                idx = int(line[6:11])
                a = self.atom_by_index(idx)
                if a.residue == 'ION':
                    continue
                new_pdb.append(line)
            except:
                if 'END' in line:
                    continue
                new_pdb.append(line)
                continue
        # add ions
        for line in self.pdb:
            try:
                idx = int(line[6:11])
                a = self.atom_by_index(idx)
                if a not in to_na:
                    continue
                newline = line[:11] + f"{a.name:>5}" + f"{a.residue:>5}" + line[21:]
                new_pdb.append(newline)
            except:
                continue
        for line in self.pdb:
            try:
                idx = int(line[6:11])
                a = self.atom_by_index(idx)
                if a not in to_cl:
                    continue
                newline = line[:11] + f"{a.name:>5}" + f"{a.residue:>5}" + line[21:]
                new_pdb.append(newline)
            except:
                continue
        new_pdb.append('END\n')
        self.pdb = new_pdb
        self.reindex_pdb()
        # === TOP ===
        new_top = []
        for line in self.top:
            try:
                mol_name = line.split()[0]
                if mol_name in ['W', 'WF']:
                    continue
                new_top.append(line)
            except:
                new_top.append(line)
        new_top.append(f"W    {len([k for k in self.atoms if k.name.strip() == 'W'])}\n")
        new_top.append(f"WF   {len([k for k in self.atoms if k.name.strip() == 'WF'])}\n")
        new_top.append(f"NA+  {len([k for k in self.atoms if k.name.strip() == 'NA+'])}\n")
        new_top.append(f"CL-  {len([k for k in self.atoms if k.name.strip() == 'CL-'])}\n")
        self.top = new_top
    
    def reindex_pdb(self):
        new_pdb = []
        i = 0
        for line in self.pdb:
            if line[:4] != 'ATOM':
                new_pdb.append(line)
                continue
            newline = line[:6] + f'{i+1:>5}' + line[11:]
            new_pdb.append(newline)
            i += 1
        self.pdb = new_pdb

    def atoms_by_name(self, names=[]):
        atoms = []
        for a in self.atoms:
            if a.name.strip() in names:
                atoms.append(a)
        return atoms
    
    def remove_atom(self, closeto=np.array([0., 0., 0.]), tolerance=0.0):
        # find atoms to be removed
        to_remove = []
        for atom in self.atoms:
            if np.linalg.norm(atom.position - closeto) > tolerance:
                continue
            to_remove.append(atom)
        # remove lines in pdb before removing atom object
        # we need the indexes of the lines in pdb 
        if len(to_remove) > 0:
            indexes = []  # this is to save line index
            atom_indexes = [k.index for k in to_remove]
            for num, line in enumerate(self.pdb):
                if line[:4] != 'ATOM':
                    continue
                if int(line[6:11]) in atom_indexes:
                    indexes.append(num)
            new_pdb = [tx for i, tx in enumerate(self.pdb) if i not in indexes]
            self.pdb = new_pdb  # update pdb
            # update atoms
            new_atoms = [a for a in self.atoms if a not in to_remove]
            self.atoms = new_atoms
            # edit topology if tag [ molecules ] is present
            molecules_top = find_lines(self.top, 'molecules')
            if len(molecules_top) > 0:
                new_topo = self.top[:molecules_top[-1]]
                for line in self.top[molecules_top[-1]:]:
                    try:
                        cad = line.split()
                        if cad[0].strip() not in [k.name.strip() for k in to_remove]:
                            new_topo.append(line)
                            continue
                        new_topo.append(f'{cad[0]}   {len([k for k in self.atoms if k.name.strip() == cad[0].strip()])}\n')
                    except:
                        new_topo.append(line)
                        continue
                self.top = new_topo
    
    def get_center(self, atomnames="all"):
        if atomnames == "all":
            atoms = self.atoms
        else:
            atoms = self.atoms_by_name(names=atomnames)
        center = np.zeros(3)
        for a in atoms:
            center += a.position
        return center/len(atoms)
    
    def translate_to(self, to=np.array([0.,0.,0.])):
        for a in self.atoms:
            a.position += to
    
    def get_inertia(self, atomnames="all"):
        if atomnames == "all":
            atoms = self.atoms
        else:
            atoms = self.atoms_by_name(names=atomnames)
        # define positions 
        positions = np.array([k.position for k in atoms])
        # 1. center of mass
        com = np.mean(positions, axis=0)
        
        # 2. relative positions
        positions_rel = positions - com
        
        # 3. inertia tensor
        x = positions_rel[:, 0]
        y = positions_rel[:, 1]
        z = positions_rel[:, 2]
        
        I_xx = np.sum(y**2 + z**2)  # Sum(y² + z²)
        I_yy = np.sum(x**2 + z**2)  # Sum(x² + z²)
        I_zz = np.sum(x**2 + y**2)  # Sum(x² + y²)
        
        I_xy = -np.sum(x * y)
        I_xz = -np.sum(x * z)
        I_yz = -np.sum(y * z)
        
        inertia_tensor = np.array([
            [I_xx, I_xy, I_xz],
            [I_xy, I_yy, I_yz],
            [I_xz, I_yz, I_zz]
        ])
        
        # 4. diagonalize
        eigenvalues, eigenvectors = np.linalg.eigh(inertia_tensor)
        
        # 5. eigenvalues and eigenvectors
        return eigenvalues, eigenvectors #.T  
    
    def rotate(self, rotmat=np.eye(3), around=np.array([0.,0.,0.])):
        for a in self.atoms:
            # translate to origin
            a.position -= around
            # apply rotation
            a.position = rotmat @ a.position
            # move to original position
            a.position += around

    def atom_by_index(self, index):
        for a in self.atoms:
            if index == a.index:
                return a

    def update_pdb(self):
        for num, line in enumerate(self.pdb):
            if line[:4] != 'ATOM':
                continue
            atom = self.atom_by_index(int(line[6:11]))
            ax, ay, az = atom.position
            newline = line[:30] + f'{ax:>8.3f}{ay:>8.3f}{az:>8.3f}' + line[54:]
            self.pdb[num] = newline
    
    def write_pdb(self):
        with open(f'{self.name}.pdb', 'w') as fo:
            for line in self.pdb:
                fo.write(line)
        logger.info(f'Model {self.name}.pdb is written')

    def write_top(self):
        with open(f'{self.name}.top', 'w') as fo:
            for line in self.top:
                fo.write(line)
        logger.info(f'Model topology {self.name}.top is written')
    
    def write_ndx(self):
        with open(f'{self.name}.ndx', 'w') as fo:
            # === [ System ] ===
            fo.write('[ System ]\n')
            for i in range(0, len(self.atoms), 15):
                line = " ".join(map(str, [k.index for k in self.atoms[i:i+15]]))
                fo.write(line + '\n')
            # === [ Protein ] ===
            protein = [a for a in self.atoms if a.residue.strip() in self.aminoacids]
            fo.write('[ Protein ]\n')
            for i in range(0, len(protein), 15):
                line = " ".join(map(str, [k.index for k in protein[i:i+15]]))
                fo.write(line + '\n')
            # === [ Membrane ] ===
            membrane = [a for a in self.atoms if a.residue.strip() in self.lipids]
            fo.write('[ Membrane ]\n')
            for i in range(0, len(membrane), 15):
                line = " ".join(map(str, [k.index for k in membrane[i:i+15]]))
                fo.write(line + '\n')
            # === [ Water_Ions ] ===
            res_wi = ['W', 'WF', 'ION']
            water_ions = [a for a in self.atoms if a.residue.strip() in res_wi]
            fo.write('[ Water_Ions ]\n')
            for i in range(0, len(water_ions), 15):
                line = " ".join(map(str, [k.index for k in water_ions[i:i+15]]))
                fo.write(line + '\n')
            # === [ Res1 ] ===
            sort_by_index = lambda a: a.index
            protein.sort(key=sort_by_index)
            fo.write('[ Res1 ]\n')
            fo.write(f'{protein[0].index}\n')
            # === [ ALLPO4 ] ===
            names_allpo4 = ['P', 'PO4']
            allpo4 = [a for a in self.atoms if a.name.strip() in names_allpo4]
            fo.write('[ ALLPO4 ]\n')
            for i in range(0, len(allpo4), 15):
                line = " ".join(map(str, [k.index for k in allpo4[i:i+15]]))
                fo.write(line + '\n')
        logger.info(f'Model index {self.name}.ndx is written')

class Atom:
    def __init__(self, index, name, position, residue) -> None:
        self.index = index
        self.name = name
        self.position = position
        self.residue = residue

class Creator:
    def __init__(self, sequence, membrane) -> None:
        self.sequence = sequence
        self.membrane = membrane
        self.allatom = self.get_allatom()
        self.cg_model = self.get_coarse()
        self.mem_model = self.read_membrane()
        self.lipids = self.get_lipids()
    
    def get_allatom(self):
        cmd.fab(self.sequence, ss=1, name='sequence')
        pdbstr = cmd.get_pdbstr()
        return pdbstr
    
    def get_lipids(self):
        # get [ molecules ] index
        lipids = set()
        for atom in self.mem_model.atoms:
            if atom.residue.strip() in ['W', 'WF']:
                continue
            lipids.add(atom.residue.strip())
        return list(lipids)
    
    def get_coarse(self):
        # Crear archivos temporales para la estructura PDB y para recibir la salida de martinize2.
        with tempfile.NamedTemporaryFile(suffix='.pdb', mode='w+', delete=False) as pdb_file:
            pdb_file.write(self.allatom) # This is the all atom structure
            pdb_file.flush()  # Swiftly transfer data from Python's temporary memory to a file

            # temporal names for martinize2 output files
            output_topology = tempfile.NamedTemporaryFile(suffix='.top', delete=False)
            output_structure = tempfile.NamedTemporaryFile(suffix='.pdb', delete=False)

            # martinize2 as subprocess
            try:
                subprocess.run([
                    "martinize2", 
                    "-f", pdb_file.name,             # Input PDB file
                    "-o", output_topology.name,      # Output topology file
                    "-x", output_structure.name,     # Output structure file
                    "-ff", "martini22",              # Specify Martini 2.2 force field
                    "-p", "backbone",                # Add position restraints to backbone
                    "-ss", "H",                      # Set alpha helix secondary structure
                    "-maxwarn", "2",                 # Accept a maximum of 2 warning messages
                ], check=True, capture_output=True)

                # Read molecule_0.itp
                molecule_topology_file = 'molecule_0.itp'
                if os.path.exists(molecule_topology_file):
                    with open(molecule_topology_file, 'r') as itp_file:
                        molecule_topology = itp_file.readlines()

                    # Delete
                    os.remove(molecule_topology_file)
                else:
                    raise FileNotFoundError(f"{molecule_topology_file} was not generated.")

                # Read martinize2 coarse-grain structure
                with open(output_structure.name, 'r') as structure_file:
                    coarse_structure = structure_file.readlines()

            finally:
                # Remove temporal files
                pdb_file.close()
                output_topology.close()
                output_structure.close()
        
        return Model(coarse_structure, molecule_topology)
    
    def read_membrane(self):
        # look for membrane structure files
        directory = os.getenv('COORDIR')
        if os.path.exists(os.path.join(directory, self.membrane + '.pdb')) and os.path.exists(os.path.join(directory, self.membrane + '.top')):
            with open(os.path.join(directory, self.membrane + '.pdb'), 'r') as fi:
                structure = fi.readlines()
            with open(os.path.join(directory, self.membrane + '.top'), 'r') as fi:
                topology = fi.readlines()
        else:
            logger.info(f'Membrane {self.membrane} is not in COORDIR {directory}')
        return Model(structure, topology)

def merge_models(peptide, membrane):
    # ===== PDB =====
    merged_pdb = ['TITLE     Merged System\n', 'REMARK    SIMULATION BOX\n']
    # read CRYST1 tag in membrane and append to merged_pdb
    cryst1 = find_lines(membrane.pdb, 'CRYST1')
    merged_pdb.append(membrane.pdb[cryst1[0]])
    # append peptide ATOM positions
    i = 0
    for line in peptide.pdb:
        if line[:4] != 'ATOM':
                continue
        newline = line[:6] + f'{i+1:>5}' + line[11:]
        merged_pdb.append(newline)
        i += 1
    # append membrane ATOM positions
    for line in membrane.pdb:
        if line[:4] != 'ATOM':
                continue
        newline = line[:6] + f'{i+1:>5}' + line[11:]
        merged_pdb.append(newline)
        i += 1
    # end pdb file
    merged_pdb.append('END\n')
    # ===== TOP =====
    merged_top = ['#include "martini-2.ff/martini_v2.2.itp"\n', 
                  '#include "martini-2.ff/martini_v2.0_lipids_all_201506.itp"\n',
                  '#include "martini-2.ff/martini_v2.0_ions.itp"\n', '\n']
    # clean peptide topology and append to merged_top
    merged_top.extend(peptide.top)
    # include name
    merged_top.extend(['\n', '[ system ]\n', 'Merged System\n', '\n'])
    # include peptide name and number
    merged_top.extend(['[ molecules ]\n', 'molecule_0   1\n'])
    # include membrane information
    membrane_from = find_lines(membrane.top, 'molecules')[0]
    merged_top.extend(membrane.top[membrane_from+1:])
    # ===== MOD =====
    # create a model
    new_model = Model(merged_pdb, merged_top)
    new_model.name = 'merged_model'
    return new_model

def find_lines(lines, tag):
    index = []
    for num, line in enumerate(lines):
        if tag in line:
            index.append(num)
    return index

def clean_comment(lines, commsym=';'):
    new = []
    for line in lines:
        clean = line.split(commsym)[0].strip()
        if len(clean) < 1:
            continue
        new.append(clean + '\n')
    return new

def parser():
    # ArgumentParser object
    parser = argparse.ArgumentParser(description="Create a system for CG fitness function")

    # add arguments
    parser.add_argument('-s', '--sequence', type=str, required=True, help='Sequence')
    parser.add_argument('-m', '--membrane', type=str, required=True, help='Membrane name')
    parser.add_argument('-p', '--prefix',   type=str, required=False, help='prefix name', default=None)

    # Parse arguments
    args = parser.parse_args()

    return args


def construct(sequence, membrane, output_prefix=None):
    # COORDIR environment variable must be defined --> where are membrane structure files?
    # in ale
    # export COORDIR=/home/tanguma_ah/martini_evolution/evoMD/evoMD_setup/structures
    # in tanguma-MUNI
    # export COORDIR=/home/tanguma/Documentos/vacha/github/evoMD_setup/structures
    
    # creator receives sequence and membrane name
    creator = Creator(sequence, membrane)

    # Trasnslate to 7 nm above membrane's center
    vec2 = creator.mem_model.get_center(atomnames=['PO4', 'PO41', 'PO42']) + np.array([0.,0.,60.])
    vec1 = creator.cg_model.get_center(atomnames=['BB'])
    vec3 = vec2 - vec1
    creator.cg_model.translate_to(to=vec3)
    creator.cg_model.update_pdb()
    # creator.cg_model.write_pdb()
    
    # get inertial axis 
    _, evec = creator.cg_model.get_inertia(atomnames=['BB'])

    # evec is a rotation matrix. it shoulf be right-handed
    if np.linalg.det(evec) < 0:
        evec[:, 2] *= -1
    
    # sort evec
    evec = np.array([
        evec[1],
        evec[0],
        evec[2],
    ])
    
    # rotate peptide to align vaa with x axis
    creator.cg_model.rotate(rotmat=evec, around=creator.cg_model.get_center(atomnames='BB'))
    creator.cg_model.update_pdb()
    creator.cg_model.name = 'model_new'
    # creator.cg_model.write_pdb()

    # remove particles in mem_model
    for a in creator.cg_model.atoms:
        creator.mem_model.remove_atom(closeto=a.position, tolerance=10.0)
    
    # merge peptide and membrane
    merged = merge_models(creator.cg_model, creator.mem_model)
    merged.add_ions()
    if output_prefix:
        merged.name = f"{output_prefix}"
    else:
        merged.name = "merged_model"
    merged.write_pdb()
    merged.write_top()
    merged.write_ndx()

@contextmanager
def change_dir(path):
    prev = os.getcwd()
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)

def constructor_method(sequence) -> None:
    # check if COORDIR is defined
    directory = os.getenv('COORDIR')
    if not directory:
        logger.error("COORDIR environment variable not set.")
        raise EnvironmentError("COORDIR is required but not defined.")
    # list of membranes
    membranes = ['ecoli', 'popc']
    for m in membranes:
        try:
            with change_dir(m):
                construct(str(sequence), m, output_prefix='system')
        except Exception as e:
            logger.error(f"Constructor failed for membrane '{m}': {e}")

def penalty_method(sequence) -> float:
    """
    Computes hydrophobic moment penalty based on alpha-helix geometry.
    Returns the penalty as a normalized value [0,1]
    """
    # --- Hydrophobicity scale and threshold ---
    method_name = 'eisenberg'  # Options: 'eisenberg', 'kyte-doolittle', 'wimley-white', 'fauchere-pliska'
    threshold = 0.4

    hydrophobicity_scales = {
        'eisenberg': {
            'A': 0.25, 'R': -1.76, 'N': -0.64, 'D': -0.72, 'C': 0.04, 'Q': -0.69,
            'E': -0.62, 'G': 0.16, 'H': -0.4,  'I': 0.73, 'L': 0.53, 'K': -1.1,
            'M': 0.26, 'F': 0.61, 'P': -0.07, 'S': -0.26, 'T': -0.18, 'W': 0.37,
            'Y': 0.02, 'V': 0.54,
        },
        'kyte-doolittle': {
            'A': 1.8,  'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,  'Q': -3.5,
            'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,  'L': 3.8,  'K': -3.9,
            'M': 1.9,  'F': 2.8,  'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9,
            'Y': -1.3, 'V': 4.2,
        },
        'wimley-white': {
            'A': 0.17, 'R': 0.81, 'N': 0.42, 'D': 1.23, 'C': -0.24, 'Q': 0.58,
            'E': 2.02, 'G': 0.01, 'H': 0.96, 'I': -0.31, 'L': -0.56, 'K': 0.99,
            'M': -0.23, 'F': -1.13, 'P': 0.45, 'S': 0.13, 'T': 0.14, 'W': -1.85,
            'Y': -0.94, 'V': 0.07,
        },
        'fauchere-pliska': {
            'A': 0.31,  'R': -1.01, 'N': -0.60, 'D': -0.77, 'C': 1.54,  'Q': -0.22,
            'E': -0.64, 'G': 0.00,  'H': 0.13,  'I': 1.80,  'L': 1.70,  'K': -0.99,
            'M': 1.23,  'F': 1.79,  'P': 0.72,  'S': -0.04, 'T': 0.26,  'W': 2.25,
            'Y': 0.96,  'V': 1.22,
        },
    }

    if method_name not in hydrophobicity_scales:
        raise ValueError(f"Hydrophobicity method '{method_name}' not implemented.")

    hydrophobicity = hydrophobicity_scales[method_name]

    # --- Preprocess sequence ---
    seq = str(sequence).upper()

    # --- Helical projection parameters ---
    theta = 100  # degrees per residue in alpha-helix
    rad = np.deg2rad(theta)
    x, y = 0.0, 0.0

    # --- Vector sum for hydrophobic moment ---
    for i, aa in enumerate(seq):
        h = hydrophobicity.get(aa, 0.0)
        angle = i * rad
        x += h * np.cos(angle)
        y += h * np.sin(angle)

    muH = np.sqrt(x**2 + y**2)

    # --- Normalization ---
    penalty = min(1, muH / threshold)

    logger.info(f'Penalty method ({method_name}): μH = {muH:.3f}, penalty = {penalty:.3f}')
    return penalty


if __name__ == '__main__':
    # receive arguments
    args = parser()
    construct(args.sequence, args.membrane, output_prefix=args.prefix)

