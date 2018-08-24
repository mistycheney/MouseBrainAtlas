#! /usr/bin/env python

import sys
import os

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from data_manager import *
from metadata import *
from distributed_utilities import *
from multiprocess import Pool

import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Rescale image')

parser.add_argument("input_fp", type=str, help="Input image name list, json file")
parser.add_argument("output_fp", type=str, help="Output image names list, json file")
parser.add_argument("-f", "--rescale_factor", type=float, help="Rescale factor")
parser.add_argument("-W", "--width", type=int, help="Width")
parser.add_argument("-H", "--height", type=int, help="Height")
parser.add_argument("-j", "--jobs", type=int, help="Number of parallel jobs", default=1)
args = parser.parse_args()

if input_fp.endswith('json'):
    input_fp_map = load_json(args.input_fp)
    in_image_names = input_fp_map.keys()
    parallel = True

if output_fp.endswith('json'):
    output_fp_map = load_json(args.output_fp)
    
if hasattr(args, "rescale_factor"):
    rescale_factor = args.rescale_factor
else:
    w = args.width
    h = args.height

n_jobs = args.jobs

def worker(img_name):

    input_fp = input_fp_map[img_name]
    output_fp = output_fp_map[img_name]
    create_parent_dir_if_not_exists(output_fp)

    img = imread(input_fp)
    save_data(img[::1/rescale_factor, ::1/rescale_factor], output_fp)


pool = Pool(n_jobs)
_ = pool.map(worker, in_image_names)
pool.close()
pool.join()

# run_distributed('convert \"%%(input_fp)s\" -crop %(w)dx%(h)d+%(x)d+%(y)d  \"%%(output_fp)s\"' % \
#                 {'w':w_raw, 'h':h_raw, 'x':x_raw, 'y':y_raw},
#                 kwargs_list=[{'input_fp': ,
#                               'output_fp': output_fp_map[img_name]}
#                              for img_name in metadata_cache['valid_filenames'][stack]],
#                 argument_type='single',
#                jobs_per_node=1,
#                local_only=True)