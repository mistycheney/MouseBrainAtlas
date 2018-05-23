#! /usr/bin/env python

import sys
import os
import time

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg') # https://stackoverflow.com/a/3054314
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from registration_utilities import *
from annotation_utilities import *
from metadata import *
from data_manager import *
from learning_utilities import *

import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='')

parser.add_argument("brain_name", type=str, help="Brain name")
parser.add_argument("--section", type=int, help="Section number. If specified, do detection on this one section; otherwise, use all valid sections.")
parser.add_argument("--win_id", type=int, help="Window id (Default: %(default)s).", default=7)
args = parser.parse_args()

stack = args.brain_name
if hasattr(args, 'section') and args.section is not None:
    sections = [args.section]
else:
    sections = metadata_cache['valid_sections'][stack]
win_id = args.win_id
    
batch_size = 256
model_dir_name = 'inception-bn-blue'
model_name = 'inception-bn-blue'
model, mean_img = load_mxnet_model(model_dir_name=model_dir_name, model_name=model_name, 
                                   num_gpus=1, batch_size=batch_size)

for sec in sections:

    compute_and_save_features_one_section(
#                                 scheme='normalize_mu_region_sigma_wholeImage_(-1,5)', 
                                scheme='none', 
#                             bbox=(11217, 16886, 13859, 18404),
#                                 method='glcm',
                            method='cnn',
                            win_id=win_id, prep_id=2,
                            stack=stack, sec=sec,
                            model=model, model_name=model_name,
                             mean_img=mean_img, 
                             batch_size=batch_size, 
        attach_timestamp=False, 
        recompute=True)