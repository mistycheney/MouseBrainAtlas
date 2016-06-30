#! /usr/bin/env python

import numpy as np

import sys
import os

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from registration_utilities import Aligner2, parallel_where

import matplotlib.pyplot as plt

from joblib import Parallel, delayed
import time

available_label_indices = sorted(labels_sided_indices.values())
labelIndexMap_sidedToUnsided = {i: labels_unsided_indices[labelMap_sidedToUnsided[labels_sided[i-1]]]
                                for i in available_label_indices}
stack = 'MD589'
subject_volume_allLabels = {}

for ind_f in set(labelIndexMap_sidedToUnsided.values()):
    print ind_f

    subject_volume_roi = bp.unpack_ndarray_file(os.path.join(volume_dir, '%(stack)s/%(stack)s_scoreVolume_%(label)s.bp' % \
                                                      {'stack': stack, 'label': labels_unsided[ind_f-1]})).astype(np.float16)
    subject_volume_allLabels[ind_f] = subject_volume_roi
    del subject_volume_roi

gradient_filepath_map_f = {ind_f: volume_dir + '/%(stack)s/%(stack)s_scoreVolume_%(label)s_%%(suffix)s.bp' % \
                           {'stack': stack, 'label': labels_unsided[ind_f-1]}
                           for ind_f in set(labelIndexMap_sidedToUnsided.values())}

labelIndexMap_m2f = {labels_sided_indices[name_s]: labels_unsided_indices[name_u]
                        for name_s, name_u in labelMap_sidedToUnsided.iteritems() \
                        if name_s in [labels_sided[i-1] for i in available_label_indices]}
#                                      if name_u in ['7N', '5N', '7n', '12N', 'Pn']}
#                                      if name_s in ['Pn_L']}
print labelIndexMap_m2f

# atlas_volume = bp.unpack_ndarray_file(os.path.join(volume_dir, 'MD589/volume_MD589_annotation_withOuterContour.bp'))
atlas_volume = bp.unpack_ndarray_file(volume_dir + '/atlasVolume_icp.bp')

all_indices_m = list(set(labelIndexMap_m2f.keys()) & set(np.unique(atlas_volume)))

# nzvoxels_m = Parallel(n_jobs=16)(delayed(parallel_where)(atlas_volume, i, num_samples=int(1e5))
#                                  for i in all_indices_m)
# nzvoxels_m = dict(zip(all_indices_m, nzvoxels_m))

aligner = Aligner2(volume_f=subject_volume_allLabels, volume_m=atlas_volume,
                #   nzvoxels_m=nzvoxels_m,
                  labelIndexMap_m2f=labelIndexMap_m2f)

print 'search'
aligner.grid_search()
