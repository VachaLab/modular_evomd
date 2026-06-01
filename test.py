# === test.py ===
from radial_sequence import get_radial
import sys
from instructor import Instructor
from generator import Generator

def read_csv(csv_file):
    with open(csv_file, 'r') as f:
        lines = f.readlines()
    lines = [[k.split(',')[0], k.split(',')[1]] for k in lines[1:]]
    return lines

def main():
    inst = Instructor("inputfile.yaml")
    print(inst.populate_method)
    inst.configure_generator()
    print(inst.generator)

if __name__ == '__main__':
    main()

    
