import yaml
from os.path import dirname, join

class DotDict(dict):
    def __getattr__(self, item):
        if item in self.keys():
            return self[item]
        return None
    
    def __setattr__(self, key, value):
        self[key] = value

def set_param(dic, key, value):
    for k in dic.keys():
        if type(dic[k]) == dict:
            if set_param(dic[k], key, value):
                return True
        if k == key:
            dic[k] = value
            return True

def to_dot_dict(dic):
    for k in dic.keys():
        if type(dic[k]) == dict:
            dic[k] = to_dot_dict(dic[k])
    return DotDict(dic)

def load_config(yaml_file, args, specific_key=None):
    with open(yaml_file, 'r') as f:
        if specific_key is not None:
            dic = yaml.load(f, Loader=yaml.FullLoader)[specific_key]
        else:
            dic = yaml.load(f, Loader=yaml.FullLoader)
    for arg_name, arg_value in (args if type(args) == DotDict else vars(args)).items():
        if arg_value is not None:
            found = set_param(dic, arg_name, arg_value)
            if not found:
                dic[arg_name] = arg_value
    return to_dot_dict(dic)


def ckpt_to_config(ckpt_path):
    config_path = join(dirname(dirname(ckpt_path)), 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return to_dot_dict(config)