# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from utilities import *
from joblib import Parallel, delayed

if 'SSH_CONNECTION' in os.environ:
    DATA_DIR = '/home/yuncong/project/DavidData2014tif/'
    REPO_DIR = '/home/yuncong/Brain'
else:
    DATA_DIR = '/home/yuncong/BrainLocal/DavidData_v4'
    REPO_DIR = '/home/yuncong/Brain'

dm = DataManager(DATA_DIR, REPO_DIR)

import argparse
import sys

parser = argparse.ArgumentParser(
formatter_class=argparse.RawDescriptionHelpFormatter,
description='Execute feature extraction pipeline',
epilog="""
The following command processes image RS141_x5_0001.tif using the specified parameters.
python %s RS141 1 -g blueNisslWide -s blueNisslRegular -v blueNissl
"""%(os.path.basename(sys.argv[0]), ))

parser.add_argument("stack_name", type=str, help="stack name")
# parser.add_argument("resolution", type=str, help="resolution string")
parser.add_argument("slice_ind", type=int, help="slice index")
parser.add_argument("-g", "--gabor_params_id", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNisslWide')
parser.add_argument("-s", "--segm_params_id", type=str, help="segmentation parameters id (default: %(default)s)", default='blueNisslRegular')
parser.add_argument("-v", "--vq_params_id", type=str, help="vector quantization parameters id (default: %(default)s)", default='blueNissl')
args = parser.parse_args()

# class args:
#     stack_name = 'RS140'
#     resolution = 'x5'
#     slice_ind = 4
#     gabor_params_id = 'blueNisslWide'
# #     gabor_params_id = 'blueNissl'
#     segm_params_id = 'blueNisslRegular'
#     vq_params_id = 'blueNissl'
    
dm.set_image(args.stack_name, args.resolution, args.slice_ind)
dm.set_gabor_params(gabor_params_id=args.gabor_params_id)
dm.set_segmentation_params(segm_params_id=args.segm_params_id)
dm.set_vq_params(vq_params_id=args.vq_params_id)

# <codecell>

def execute_command(cmd):
	try:
	    retcode = call(cmd, shell=True)
	    if retcode < 0:
	        print >>sys.stderr, "Child was terminated by signal", -retcode
	    else:
	        print >>sys.stderr, "Child returned", retcode
	except OSError as e:
	    print >>sys.stderr, "Execution failed:", e

# <codecell>

%run gabor_filter_nocrop_noimport.ipynb
%run segmentation_nocrop_noimport.ipynb
%run rotate_features_noimport.ipynb
%run assign_textons_max_anchored_noimport.ipynb
%run compute_texton_histograms_noimport.ipynb
%run grow_regions_clean_noimport.ipynb

# <codecell>


