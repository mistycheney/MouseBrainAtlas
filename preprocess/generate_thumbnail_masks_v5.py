#! /usr/bin/env python

import os
import sys
import time

import numpy as np
from multiprocess import Pool
import morphsnakes

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from data_manager import *
from preprocess_utilities import *

#######################################################################################

import argparse
import json

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Generate masks for aligned thumbnail images')
parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("filenames", type=str, help="image filenames, json encoded, no extensions")
parser.add_argument("init_snake_contours_fp", type=str, help="initial snake contour file path")
parser.add_argument("--min_size", type=int, help="minimum submask size", default=MIN_SUBMASK_SIZE)
parser.add_argument("--default_channel", type=int, help="default RGB channel to do snake on", default=0)
args = parser.parse_args()

stack = args.stack_name
filenames = json.loads(args.filenames)
min_size = args.min_size
default_channel = args.default_channel

init_snake_contours_fp = args.init_snake_contours_fp
download_from_s3_to_ec2(init_snake_contours_fp)
init_snake_contour_vertices = load_pickle(init_snake_contours_fp) # {fn: vertices}

##############################################

def generate_contours(fn, init_cnt):
    
    img = imread(DataManager.get_image_filepath(stack=stack, fn=fn, resol='thumbnail', version='aligned'))[..., default_channel]
    img = brightfieldize_image(img)
    img = contrast_stretch_image(img)
    submasks = snake(img, init_contours=[init_cnt], lambda1=1., min_size=min_size)
    submasks = dict(enumerate(submasks))

    assert len(submasks) > 0, "No submask is found."
    
    # Create output dir.
    create_if_not_exists(DataManager.get_auto_submask_dir_filepath(stack=stack, fn=fn))
    
    # Save submasks
    for submask_ind, m in submasks.iteritems():
        submask_fp = DataManager.get_auto_submask_filepath(stack=stack, fn=fn, what='submask', submask_ind=submask_ind)
        imsave(submask_fp, np.uint8(m)*255)
        upload_from_ec2_to_s3(submask_fp)

    submask_decisions = {sm_i: True for sm_i in submasks.iterkeys()}
        
    # Save submask decisions
    decisions_fp = DataManager.get_auto_submask_filepath(stack=stack, fn=fn, what='decisions')
    from pandas import Series
    Series(submask_decisions).to_csv(decisions_fp)
    upload_from_ec2_to_s3(decisions_fp)

    
t = time.time()
pool = Pool(NUM_CORES/2)
pool.map(lambda fn: generate_contours(fn, init_snake_contour_vertices[fn]), filenames)
pool.close()
pool.join()

# for fn in filenames:
#     generate_contours(fn, init_snake_contour_vertices[fn])

sys.stderr.write('Generate contours: %.2f\n' % (time.time()-t))
