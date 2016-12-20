#! /usr/bin/env python

# This must be before any other matplotlib imports
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

import sys
import os

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from registration_utilities import *
from metadata import *
from data_manager import *

from joblib import Parallel, delayed
import time

stack_fixed = sys.argv[1]
train_sample_scheme = int(sys.argv[2])
global_transform_scheme = int(sys.argv[3])
trial_idx = int(sys.argv[4])
atlas_name = sys.argv[5]

# stack_moving = 'atlas_on_MD589'
stack_moving = atlas_name

paired_structures = ['5N', '6N', '7N', '7n', 'Amb', 'LC', 'LRt', 'Pn', 'Tz', 'VLL', 'RMC', 'SNC', 'SNR', '3N', '4N',
                    'Sp5I', 'Sp5O', 'Sp5C', 'PBG', '10N', 'VCA', 'VCP', 'DC']
singular_structures = ['AP', '12N', 'RtTg', 'SC', 'IC']
structures = paired_structures + singular_structures

structures_sided = sum([[n] if n in singular_structures else [convert_to_left_name(n), convert_to_right_name(n)]
                        for n in structures], [])

structures_sided_with_surround = sum([[n, n+'_surround'] for n in structures_sided], [])

###################################################################################

# Load transform parameters
global_params, centroid_m, centroid_f, xdim_m, ydim_m, zdim_m, xdim_f, ydim_f, zdim_f = \
DataManager.load_global_alignment_parameters(stack_moving=stack_moving,
                                                    stack_fixed=stack_fixed,
                                                    train_sample_scheme=train_sample_scheme,
                                                    global_transform_scheme=global_transform_scheme,
                                                    trial_idx=trial_idx)

# Transform moving volume, sided, with surround

for name_s in structures_sided_with_surround:
    print name_s

    vol_m = DataManager.load_score_volume(stack=stack_moving, label=name_s, downscale=32)

    volume_m_alignedTo_f = \
    transform_volume(vol=vol_m, global_params=global_params, centroid_m=centroid_m, centroid_f=centroid_f,
                      xdim_f=xdim_f, ydim_f=ydim_f, zdim_f=zdim_f)

    DataManager.save_transformed_volume(volume_m_alignedTo_f,
                                        stack_m=stack_moving, type_m='score',
                                            stack_f=stack_fixed, type_f='score',
                                            label=name_s,
                                            downscale=32,
                                            train_sample_scheme_f=train_sample_scheme,
                                            global_transform_scheme=global_transform_scheme)
