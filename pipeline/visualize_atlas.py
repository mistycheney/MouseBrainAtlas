#! /usr/bin/env python

import os
import argparse
import sys
import time

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Find growed cluster consensus',
    epilog="")

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("slice_ind", type=int, help="slice index")
parser.add_argument("-g", "--gabor_params_id", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNisslWide')
parser.add_argument("-s", "--segm_params_id", type=str, help="segmentation parameters id (default: %(default)s)", default='blueNisslRegular')
parser.add_argument("-v", "--vq_params_id", type=str, help="vector quantization parameters id (default: %(default)s)", default='blueNissl')
args = parser.parse_args()


from joblib import Parallel, delayed

sys.path.append(os.path.join(os.environ['GORDON_REPO_DIR'], 'notebooks'))
from utilities2015 import *

# dm = DataManager(data_dir=os.environ['GORDON_DATA_DIR'], 
#                  repo_dir=os.environ['GORDON_REPO_DIR'], 
#                  result_dir=os.environ['GORDON_RESULT_DIR'], 
#                  labeling_dir=os.environ['GORDON_LABELING_DIR'],
#                  gabor_params_id=args.gabor_params_id, 
#                  segm_params_id=args.segm_params_id, 
#                  vq_params_id=args.vq_params_id,
#                  stack=args.stack_name, 
#                  section=args.slice_ind)

dm = DataManager(segm_params_id='gridsize200', 
                 stack='MD593', 
                 section=98)

#======================================================

b_models = dm.load_pipeline_result('boundaryModels')
bnds = dm.load_pipeline_result('atlasLandmarkIndices')
viz = dm.visualize_edge_sets([b_models[lm_ind][0] for lm_ind, group_ind in bnds], 
                                 show_set_index=True,
                                 labels=[str(group_ind) for lm_ind, group_ind in bnds],
                                bg='originalImage')
dm.save_pipeline_result(viz, 'atlasViz')