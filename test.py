# === test.py ===

from instructor_modular import Instructor
from instruction_validators import *
from insturction_fields import Instruction

if __name__ == '__main__':
    file_name = 'inputfile.yaml'
    instructor = Instructor(filename=file_name)
    print('------------')
    print(instructor)
    print('------------')
    print([k for k in instructor])

