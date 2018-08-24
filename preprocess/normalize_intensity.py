#! /usr/bin/env python

import os
import argparse
import sys
import time

import numpy as np
from multiprocess import Pool

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from metadata import *
from data_manager import *
from learning_utilities import *

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='')

parser.add_argument("stack", type=str, help="Brain name")
parser.add_argument("input_version", type=str, help="Input image version", default='Ntb')
parser.add_argument("output_version", type=str, help="Output image version", default='NtbNormalized')
args = parser.parse_args()


def worker(img_name):

    t = time.time()
    
    in_fp = input_fp_map[img_name]
    out_fp = output_fp_map[img_name]
    create_parent_dir_if_not_exists(out_fp)

    cmd = """convert "%(in_fp)s" -normalize -depth 8 "%(out_fp)s" """ % {'in_fp': in_fp, 'out_fp': out_fp}
    execute_command(cmd)
    
    sys.stderr.write("Intensity normalize: %.2f seconds." % (time.time() - t))
    
pool = Pool(n_jobs)
_ = pool.map(worker, in_image_names)
pool.close()
pool.join()

