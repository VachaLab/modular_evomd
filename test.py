# === test.py ===
from radial_sequence import get_radial
import sys
from instructor import Instructor
from generator import Generator
from evolver import Evolver

def read_csv(csv_file):
    with open(csv_file, 'r') as f:
        lines = f.readlines()
    lines = [[k.split(',')[0], k.split(',')[1]] for k in lines[1:]]
    return lines

def main():
    file_name = sys.argv[1]
    inst = Instructor(file_name)
    print(inst.populate_method)
    inst.configure_generator()
    print(inst.generator)
    evo = Evolver(instructor=inst)
    evo.populate()
    print(evo.sequences)
    evo.started = True
    evo.parent_sequences = [k for k in evo.sequences]
    evo.sequences = []
    evo.populate()

if __name__ == '__main__':
    main()

    
