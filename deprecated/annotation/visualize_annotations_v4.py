#! /usr/bin/env python

import argparse
import json

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='visualize annotations, version 4')
parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("filenames", type=str, help="image filenames")
parser.add_argument("-d", "--downsample", type=int, help="downsample factor", default=8)
args = parser.parse_args()

######################################

import os
import sys

from multiprocess import Pool

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from metadata import *
from utilities2015 import *
from data_manager import *
from learning_utilities import *

#####################################

stack = args.stack_name
filenames = json.loads(args.filenames)
downsample_factor = args.downsample

contours_df, _ = DataManager.load_annotation_v3(stack=stack)
contours = contours_df[(contours_df['orientation'] == 'sagittal') & (contours_df['downsample'] == 1)]
contours = contours.drop_duplicates(subset=['section', 'name', 'side', 'filename', 'downsample', 'creator'])
contours = convert_annotation_v3_original_to_aligned_cropped(contours, stack=stack)

structure_colors = {n: np.random.randint(0, 255, (3,)) for n in all_known_structures}

def generate_annotation_viz_one_section(stack, fn, structure_colors=structure_colors, downsample_factor=downsample_factor):
    global contours
    
    if is_invalid(fn):
        return
    
    img_fp = DataManager.get_image_filepath(stack=stack, fn=fn, resol='lossless', version='compressed')
    download_from_s3(img_fp)
    img = imread(img_fp)
    viz = img[::downsample_factor, ::downsample_factor].copy()
    
    for name_u, color in structure_colors.iteritems():
        matched_contours = contours[(contours['name'] == name_u) & (contours['filename'] == fn)]
        for cnt_id, cnt_props in matched_contours.iterrows():
            cv2.polylines(viz, [(cnt_props['vertices']/downsample_factor).astype(np.int)], True, color, 2)
    
    viz_fp = DataManager.get_annotation_viz_filepath(stack=stack, fn=fn)
    create_parent_dir_if_not_exists(viz_fp)
    imsave(viz_fp, viz)
    upload_to_s3(viz_fp)

# for fn in filenames:
#     generate_annotation_viz_one_section(fn=fn)

pool = Pool(NUM_CORES/2)
pool.map(lambda fn: generate_annotation_viz_one_section(stack=stack, fn=fn, structure_colors=structure_colors, downsample_factor=downsample_factor), filenames)
pool.close()
pool.join()
