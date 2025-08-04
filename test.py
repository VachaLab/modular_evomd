# === test.py ===

from instructor_modular import Instructor

file_name = 'inputfile.yaml'
instructor = Instructor(filename=file_name)
print(instructor.yaml_data)