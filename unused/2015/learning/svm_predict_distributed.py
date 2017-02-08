#! /usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description="Generate visualizations of score maps.")
parser.add_argument("stack", help="stack name, e.g. MD593")

args = parser.parse_args()

import os
import sys
import time

stack = args.stack

##################################################

t = time.time()

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from preprocess_utility import *
from data_manager import *
from metadata import *

exclude_nodes = [33]
first_sec, last_sec = DataManager.load_cropbox(stack)[4:]

run_distributed4(command='%(script_path)s %(stack)s %%(first_sec)d %%(last_sec)d' % \
                {'script_path': os.path.join(os.environ['REPO_DIR'], 'learning') + '/svm_predict.py',
                'stack': stack},
                kwargs_list=dict(sections=range(first_sec, last_sec+1)),
                exclude_nodes=exclude_nodes,
                argument_type='partition')

sys.stderr.write('done in %f seconds\n' % (time.time() - t))
