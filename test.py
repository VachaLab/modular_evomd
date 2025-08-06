# === test.py ===

# from instructor_modular import Instructor

# file_name = 'inputfile.yaml'
# instructor = Instructor(filename=file_name)
# print(instructor.yaml_data)


from instruction_validators import *


if __name__ == '__main__':
    # val = SubtypeValidator(str)
    # val = TypeValidator(list)
    val = SubchoicesValidator(['hola', 'amigo'])
    print(val)
    print(val.validate(['hola', 'amigo']))
