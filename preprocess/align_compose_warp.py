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

parser.add_argument("--stack", type=str, help="Brain name")
parser.add_argument("--resol", type=str, help="Resolution", default='thumbnail')
parser.add_argument("--version", type=str, help="")
parser.add_argument("--image_names", type=str, help="Image name list")
parser.add_argument("--filelist", type=str, help="csv file. Each row is imageName,filePath .")
parser.add_argument("--anchor", type=str, help="Anchor image name")
parser.add_argument("--elastix_output_dir", type=str, help="Folder for pairwise transforms computed by Elastix")
parser.add_argument("--custom_output_dir", type=str, help="Folder for pairwise transforms provided by human")


args = parser.parse_args()

if hasattr(args, 'stack') and hasattr(args, 'image_names') and hasattr(args, 'version') and hasattr(args, 'resol'):
    image_names = load_txt(args.image_names)
    filelist = [(imgName, DataManager.get_image_filepath_v2(stack=args.stack, fn=imgName, resol=args.resol, version=args.version)) 
                 for imgName in image_names]
elif hasattr(args, 'filelist'):
    filelist = load_csv(args.filelist)
else:
    raise Exception('Must provide filelist or (resol, version, image_list).')

    
# Step 1: Compute pairwise transforms.
    
t = time.time()
print 'Align...'

run_distributed("%(script)s \"%(input_dir)s\" \"%(output_dir)s\" \'%%(kwargs_str)s\' %(fmt)s -p %(param_fp)s -r" % \
                {'script': os.path.join(REPO_DIR, 'preprocess', 'align_consecutive_v3.py'),
                'output_dir': args.elastix_output_dir,
                 'param_fp': '/home/yuncong/Brain/preprocess/parameters/Parameters_Rigid_MutualInfo_noNumberOfSpatialSamples_4000Iters.txt'
                },
                kwargs_list=[{'prev_img_name': filelist[i-1][0],
                              'curr_img_name': filelist[i][0],
                              'prev_fp': filelist[i-1][1],
                              'curr_fp': filelist[i][1],
                             } 
                            for i in range(1, len(filelist))],
                argument_type='list',
                jobs_per_node=8,
               local_only=True)

# wait_qsub_complete()

print 'done in', time.time() - t, 'seconds' # 2252 seconds full stack

# Step 2: Compose pairwise transforms to get each image's transform to anchor.

if hasattr(args, 'anchor'):
    anchor_img_name = args.anchor
else:
    assert hasattr(args, 'stack')
    anchor_img_name = DataManager.load_anchor_filename(stack=args.stack)

img_name_list = [img_name for img_name, _ in filelist]

toanchor_transforms_fp = os.path.join(DATA_DIR, stack, '%(stack)s_transformsTo_%(anchor_img_name)s.pkl' % \
                         dict(stack=stack, anchor_img_name=anchor_img_name))

t = time.time()
print 'Composing transform...'

cmd = "%(script)s --elastix_output_dir \"%(elastix_output_dir)s\" --custom_output_dir \"%(custom_output_dir)s\" --image_name_list \"%(image_name_list)s\" --anchor_img_name \"%(anchor_img_name)s\" --toanchor_transforms_fp \"%(toanchor_transforms_fp)s\"" % \
            {
                # 'stack': stack,
            'script': os.path.join(REPO_DIR, 'preprocess', 'compose_transform_thumbnail_v3.py'),
            'elastix_output_dir': args.elastix_output_dir,
            'custom_output_dir': args.custom_output_dir,
             'image_name_list': image_name_list,
             'anchor_img_name': anchor_img_name,
            'toanchor_transforms_fp': toanchor_transforms_fp}

execute_command(cmd)
        
print 'done in', time.time() - t, 'seconds' # 20 seconds

