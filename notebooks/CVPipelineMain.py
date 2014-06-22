# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# %load_ext autoreload
# %autoreload 2

from IPython.display import Image, FileLink

from CVPipeline_module import CrossValidationPipeline
from kmeans_module import SaliencyDetector
from utilities import *

import cv2
import matplotlib.pyplot as plt

import random

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# <codecell>

CACHE_DIR = '/home/yuncong/scratch/'
IMG_DIR = '/home/yuncong/ParthaData/PMD1305_reduce0_region0/'
img_name_fmt = 'PMD1305_%d_reduce0_region0'
img_id = 244
img_name = img_name_fmt%img_id

params = {
# 'param_id': random.randint(0,9999),
'param_id': 1635,
'theta_interval': 10,
'n_freq': 4,
'max_freq': 0.2,
'n_texton': 20,
'cache_dir': CACHE_DIR,
'img_dir': IMG_DIR
}
pipeline = CrossValidationPipeline(params=params)

# <codecell>

pipeline.filter_image(img_name, output_feature=False)

# <codecell>

pipeline.generate_textonmap()

