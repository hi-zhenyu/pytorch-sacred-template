import os
import random
import argparse
import logging
import numpy as np

# sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds


from network import LeNet
from main import train, test

# parse_args
parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument('--data_dir', type=str, default='./data',
                    help='input size')
parser.add_argument('--output_root', type=str, default='./output',
                    help='input size')
parser.add_argument('--val_split', type=float, default=0.01,
                    help='validation split from training data')

parser.add_argument('--num_classes', type=int, default=10,
                    help='num_classes')
parser.add_argument('--num_epochs', type=int, default=1,
                    help='num_epochs')
parser.add_argument('--log_step', type=int, default=100,
                    help='loging step')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--learning_rate', type=float, default=1e-3,
                    help='learning rate')
args = parser.parse_args()

# sacred experiment
ex = Experiment('pytorch')
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(FileStorageObserver.create(args.output_root))

ex.add_config(args.__dict__)

@ex.automain
def run(_run):
    # Experiment
    logging.info('\n *--------------- Experiment Config ---------------*')
    output_dir = os.path.join(args.output_root, _run._id)
    args.output_dir = output_dir
    logging.info(args)

    # network
    net = LeNet(args.num_classes)
    net.cuda()

    # Train
    print('\n*--------------- Training ---------------*')
    train_result = train(_run, args, net)

    # Test
    print('\n*--------------- Testing ---------------*')
    test_result = test(_run, args, net)
    print('ACC', test_result)

    return test_result

