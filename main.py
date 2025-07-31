import json
import os
from train import train
from test import do_test
from options.train_options import TrainOptions
from options.test_options import TestOptions
from util import util
import sys

def print_options(opt):
    message = '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        message += f'{k:>25}: {str(v):<30}\n'
    message += '----------------- End -------------------'
    print(message)

    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    util.mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, f'{opt.phase}_opt.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message + '\n')

def apply_custom_config(opt, config_dict):
    for k, v in config_dict.items():
        if hasattr(opt, k):
            setattr(opt, k, v)
    return opt

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def main(mode: str, config_path: str):
    # Prevent argparse from reading CLI args passed to run.py
    sys.argv = [sys.argv[0]]

    config = load_config(config_path)

    if mode == "train":
        opt = TrainOptions().parse()
        opt = apply_custom_config(opt, config)
        print_options(opt)
        train(opt)

    elif mode == "test":
        opt = TestOptions().parse()
        opt = apply_custom_config(opt, config)
        print_options(opt)
        do_test(opt)
