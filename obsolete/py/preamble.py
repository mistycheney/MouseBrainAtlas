# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from utilities import *
from joblib import Parallel, delayed
import os

os.environ['GORDON_DATA_DIR'] = '/home/yuncong/project/DavidData2014tif/'
os.environ['GORDON_REPO_DIR'] = '/home/yuncong/Brain'
os.environ['GORDON_RESULT_DIR'] = '/home/yuncong/project/DavidData2014results/'
os.environ['GORDON_LABELING_DIR'] = '/home/yuncong/project/DavidData2014labelings/'

dm = DataManager(data_dir=os.environ['GORDON_DATA_DIR'], 
  repo_dir=os.environ['GORDON_REPO_DIR'], 
  result_dir=os.environ['GORDON_RESULT_DIR'], 
  labeling_dir=os.environ['GORDON_LABELING_DIR'])

class args:
    stack_name = 'RS140'
    resolution = 'x5'
    slice_ind = 0
    gabor_params_id = 'blueNisslWide'
    segm_params_id = 'blueNisslRegular'
    vq_params_id = 'blueNissl'
    
dm.set_image(args.stack_name, args.resolution, args.slice_ind)
dm.set_gabor_params(gabor_params_id=args.gabor_params_id)
dm.set_segmentation_params(segm_params_id=args.segm_params_id)
dm.set_vq_params(vq_params_id=args.vq_params_id)

