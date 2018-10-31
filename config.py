import os

from few_shot.utils import mkdir


PATH = os.path.dirname(os.path.realpath(__file__))

DATA_PATH = None

if DATA_PATH is None:
    raise Exception('Configure your data folder location in config.py before continuing!')


# Quick hack to create folders
mkdir(PATH + '/logs/')
mkdir(PATH + '/logs/proto_nets')
mkdir(PATH + '/logs/matching_nets')
mkdir(PATH + '/models/')
mkdir(PATH + '/models/proto_nets')
mkdir(PATH + '/models/matching_nets')
