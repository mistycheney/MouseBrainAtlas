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
parser.add_argument("versions", type=str, help="json encoded str list")
parser.add_argument("resolutions", type=str, help="json encoded str list")
parser.add_argument("prep_in", type=str, help="")
parser.add_argument("prep_out", type=str, help="")
parser.add_argument("input_crop_json", type=str, help="")
parser.add_argument("output_crop_json", type=str, help="")
parser.add_argument("n_jobs", type=int, help="", default=1)
args = parser.parse_args()

versions = json.loads(args.versions)
if isinstance(versions, str):
    versions = [versions]
else:
    assert isinstance(versions, list), "Argument versions must be str or str list."

resolutions = json.loads(args.resolutions)
if isinstance(resolutions, str):
    resolutions = [resolutions]
else:
    assert isinstance(resolutions, list), "Argument resolutions must be str or str list."
    
n_jobs = args.n_jobs


def crop(stack, img_name, version, resol, x,y,w,h):

    input_fp = DataManager.get_image_filepath_v2(stack=stack, prep_id=5, resol=resol, version=version, fn=img_name)
    output_fp = DataManager.get_image_filepath_v2(stack=stack, fn=img_name, prep_id=2, version=version, resol=resol)

    img = imread(input_fp)
    save_data(img[y:y+h, x:x+w], output_fp)

for version in versions:
    for resol in resolutions:

        if resol == 'raw':
            x = x_tb * 32
            y = y_tb * 32
            w = w_tb * 32
            h = h_tb * 32
        elif resol == 'thumbnail':
            x = x_tb
            y = y_tb
            w = w_tb
            h = h_tb
        else:
            raise

#             input_dir = DataManager.get_image_dir_v2(stack=stack, prep_id=5, version=version, resol='raw')
        out_dir = DataManager.get_image_dir_v2(stack=stack, prep_id=2, resol=resol, version=version)
        print 'out_dir:', out_dir
#             script = os.path.join(REPO_DIR, 'preprocess', 'warp_crop_IM_v3.py')

#         ! rm -rf {out_dir}
        create_if_not_exists(out_dir)

        t = time.time()

        pool = Pool(8)
        _ = pool.map(lambda img_name: crop(stack=stack, img_name=img_name, version=version, resol=resol, 
                                        x=x, y=y, w=w, h=h), 
                     metadata_cache['valid_filenames'][stack])
        pool.close()
        pool.join()

#             for img_name in metadata_cache['valid_filenames'][stack]:
#                 f(stack=stack, img_name=img_name, version=version, resol=resol, 
#                                             x=x, y=y, w=w, h=h)

    #     run_distributed('convert \"%%(input_fp)s\" -crop %(w)dx%(h)d+%(x)d+%(y)d  \"%%(output_fp)s\"' % \
    #                     {'w':w_raw, 'h':h_raw, 'x':x_raw, 'y':y_raw},
    #                     kwargs_list=[{'input_fp': DataManager.get_image_filepath_v2(stack=stack, prep_id=5, resol='raw', version=version, fn=img_name),
    #                                   'output_fp': DataManager.get_image_filepath_v2(stack=stack, fn=img_name, prep_id=2, version=version, resol='raw')}
    #                                  for img_name in metadata_cache['valid_filenames'][stack]],
    # #                                  for img_name in ['CHATM3_slide35_2018_02_17-S1']],
    #                     argument_type='single',
    #                    jobs_per_node=1,
    #                    local_only=True)

        # wait_qsub_complete()

        print 'done in', time.time() - t, 'seconds' # 1500s