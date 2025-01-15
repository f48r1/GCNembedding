import os
import re

# Root of this project
DIR_ROOT = './'

DIR_DATA = os.path.join(DIR_ROOT,'data')

DIR_RAWDATASET = os.path.join(DIR_DATA,'dataset')
DIR_MATRIX = os.path.join(DIR_DATA, 'matrix')
DIR_ADJACENCY = os.path.join(DIR_DATA, 'adj')
DIR_COMMUNITIES = os.path.join(DIR_DATA, 'communities')
DIR_INTRAIDXS = os.path.join(DIR_DATA, 'intraDomainIdxs')


DIR_EXPERIMENTS = os.path.join(DIR_ROOT, 'experiments')

def path_pred_setup(fp_type:str, cls_name:str, cls_configs:dict):
    if not cls_configs:
        path_configs = 'configs=none'
    else:
        path_configs = '-'.join([f'{k}={v}' for k,v in cls_configs.items()])

    return os.path.join(DIR_EXPERIMENTS,f'fp={fp_type}', f'cls={cls_name}', path_configs)

def name_file_cv(n_split:int, seed:int):
    return f'kfold={n_split}-seed={seed}'

def dict_args_from(path_file:str):
    path, ext = os.path.splitext(path_file)
    splitted_path = os.path.normpath(path).split(os.sep)
    relevant_path = os.path.join(*splitted_path[-4:])

    compiler_kwargs = re.compile(r'(?P<key>[^-/]+)=(?P<value>[^-/]+)')
    kwargs = compiler_kwargs.findall(relevant_path)
    return dict(kwargs)