from .common import configs as common_configs
from .nn import configs as nn_configs
from .cv import configs as cv_configs
import os

configs = common_configs | nn_configs | cv_configs

def add_arguments_to(parser, arguments):
    sub_configs = dict((k,configs[k]) for k in arguments if k in common_configs)
    for argument, arg_config in sub_configs.items():
        parser.add_argument(f'--{argument}',**arg_config)

def add_nn_params_to(parser):
    for argument, arg_config in nn_configs.items():
        parser.add_argument(f'--{argument}',**arg_config)

def add_cv_params_to(parser):
    for argument, arg_config in cv_configs.items():
        parser.add_argument(f'--{argument}',**arg_config)

def nn_params_path(args):
    args_dict = vars(args)

    nn_dict={'model':'GCN'}
    nn_dict|={k:args_dict[k] for k in nn_configs.keys() if k!='epochs'}

    return '-'.join([f'{k}={v}' for k,v in nn_dict.items()])

def cv_params_path(args):
    args_dict = vars(args)

    cv_dict={k:args_dict[k] for k in cv_configs.keys()}

    return '-'.join([f'{k}={v}' for k,v in cv_dict.items()])