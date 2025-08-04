import yaml
file_name = 'inputfile.yaml'
with open(file_name, 'r') as f:
    yaml_data = yaml.load(f, Loader=yaml.FullLoader)


print([type(k) for k in yaml_data['mixture_weights']])