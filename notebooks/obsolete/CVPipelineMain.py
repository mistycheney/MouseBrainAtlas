# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# %load_ext autoreload
# %autoreload 2

from IPython.display import Image, FileLink

from CVPipelineModule import CrossValidationPipeline
# from kmeans_module import SaliencyDetector
from utilities import *

import cv2
import matplotlib.pyplot as plt

import random

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# <codecell>

output_dir = '../output'
data_dir = '../data'
param_dump_dir = '../params'

dataset_name = 'PMD1305_reduce2_region0'
img_idx = 244
param_id = 2

import os
param_file = os.path.join(param_dump_dir, 'param%d.json'%param_id)
pipeline = CrossValidationPipeline(param_file)

img_path = os.path.join(data_dir, dataset_name, dataset_name + '_%04d.tif'%img_idx)

# <codecell>

%%time
pipeline.filter_image(img_path)

