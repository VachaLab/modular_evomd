# === test.py ===
from instructor import Instructor
from evolver import Evolver
import utils


def main():
    evo_pre = 'evolver.pkl'
    evo = utils.read_pkl(evo_pre)
    print(evo.instructor)
    new_inst = Instructor('inputfile.yaml')
    new_evo = Evolver(instructor=new_inst)
    evo.instructor = new_inst
    print(evo.instructor)
    evo.save_pkl()

if __name__ == '__main__':
    main()

    