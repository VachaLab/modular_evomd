import numpy as np
import sys
import re


class Model:
    def __init__(self, atoms: list = None, box: np.ndarray = None) -> None:
        self.atoms = atoms
        self.box = box

    def read_gro(self, gro_file: str) -> None:
        # get lines
        with open(gro_file, 'r') as f:
            lines = f.readlines()
        # get box
        box = np.array(lines[-1].split(), dtype=float)
        # get number of atoms
        num_atoms = int(lines[1].strip())
        # get atoms
        atoms = []
        for line in lines[2:num_atoms+2]:
            a = Atom(
                    index=int(line[15:20].strip()), 
                    name=line[10:15].strip(), 
                    resname=line[5:10].strip(),
                    residx=int(line[:5].strip()),
                    position=np.array(line[20:44].split(), dtype=float),
                    )
            atoms.append(a)
        self.atoms = atoms
        self.box = box

    def write_gro(self, out='model.gro') -> None:
        with open(out, 'w') as f:
            f.write(f'Model restraint\n{len(self.atoms)}\n')
            for a in self.atoms:
                f.write(f'{a}\n')
            f.write(f'{self.box[0]:>10.5f}{self.box[1]:>10.5f}{self.box[2]:>10.5f}\n')

    def rest_mem(self) -> None:
        pattern = re.compile(r"(PC|PG|OSM|PS|PA|PE)")
        mem_atoms = [k for k in self.atoms if pattern.search(k.resname)]
        # get average z position
        z = 0
        for a in mem_atoms:
            z += a.z
        z /= len(mem_atoms)
        for a in self.atoms:
            if not pattern.search(a.resname):
                continue
            a.position = np.array([a.x, a.y, z])

class Atom:
    def __init__(
            self, 
            index: int, 
            name: str, 
            resname: str, 
            residx: int,
            position: np.ndarray
            ) -> None:
        self.index = index
        self.name = name
        self.resname = resname
        self.residx = residx
        self.position = position
    
    def __str__(self) -> str:
        text = f'{self.residx:>5}{self.resname:<5}{self.name:>5}{self.index:>5}{self.x:>8.3f}{self.y:>8.3f}{self.z:>8.3f}'
        return text

    @property
    def x(self):
        return self.position[0]

    @property
    def y(self):
        return self.position[1]

    @property
    def z(self):
        return self.position[2]


def main():
    # Geometry
    geom = sys.argv[1]
    model = Model()
    model.read_gro(geom)
    model.rest_mem()
    model.write_gro(out='restraints.gro')
    # Read geometry

if __name__ == '__main__':
    main()
