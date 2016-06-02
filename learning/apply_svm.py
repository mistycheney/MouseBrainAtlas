#! /usr/bin/env python

import numpy as np

import sys, os
from sklearn.svm import SVC
from sklearn.externals import joblib
import time

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *

from joblib import Parallel, delayed

stack = sys.argv[1]
first_sec = int(sys.argv[2])
last_sec = int(sys.argv[3])

####################################################

# input
features_dir = '/home/yuncong/csd395/CSHL_patch_features_Sat16ClassFinetuned/'

# output

# predictions_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_patch_predictions_svm_Sat16ClassFinetuned'
# predictions_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_patch_predictions_svm_Sat16ClassFinetuned_v2'
predictions_rootdir = create_if_not_exists('/oasis/projects/nsf/csd395/yuncong/CSHL_patch_predictions_svm_Sat16ClassFinetuned_v3')

# svm_dir = 'svm_Sat16ClassFinetuned'
# svm_dir = 'svm_Sat16ClassFinetuned_v2'
svm_dir = os.environ['REPO_DIR'] + '/learning/svm_Sat16ClassFinetuned_v3'
create_if_not_exists(svm_dir)
    
labels = ['BackG', '5N', '7n', '7N', '12N', 'Pn', 'VLL', 
          '6N', 'Amb', 'R', 'Tz', 'RtTg', 'LRt', 'LC', 'AP', 'sp5']

# Load pre-computed svm classifiers
svc_allClasses = {}
for label_ind, label in enumerate(labels[1:]):
    svc_allClasses[label] = joblib.load(svm_dir + '/%(label)s_svm.pkl' % {'label': label})
    
def svm_predict(stack, sec):
    
    test_features_dir = features_dir + '/%(stack)s/%(sec)04d' % {'stack': stack, 'sec': sec}
#     features_roi = bp.unpack_ndarray_file(test_features_dir + '/%(stack)s_%(sec)04d_features.bp' % {'stack': stack, 'sec': sec})
    features_roi = load_hdf(test_features_dir + '/%(stack)s_%(sec)04d_roi1_features.hdf' % {'stack': stack, 'sec': sec})
    n = features_roi.shape[0]

    predictions_dir = predictions_rootdir + '/%(stack)s/%(sec)04d' % {'stack': stack, 'sec': sec}
    create_if_not_exists(predictions_dir)

###### all labelmaps in one file #########

#     probs = np.zeros((n, len(labels)-1))
#     for label_ind, label in enumerate(labels[1:]):
#         svc = svc_allClasses[label]
#         probs[:, label_ind] = svc.predict_proba(features_roi)[:, svc.classes_.tolist().index(1.)]
#     np.save(predictions_dir + '/%(stack)s_%(sec)04d_roi1_scores.npy' % {'stack': stack, 'sec': sec}, probs)
    
###### separate labelmap in different files #######

    for label_ind, label in enumerate(labels[1:]):
        svc = svc_allClasses[label]
        probs = svc.predict_proba(features_roi)[:, svc.classes_.tolist().index(1.)]
        np.save(predictions_dir + '/%(stack)s_%(sec)04d_roi1_%(label)s_sparseScores.npy' % {'stack': stack, 'sec': sec, 'label': label}, 
                probs)
        
Parallel(n_jobs=8)(delayed(svm_predict)(stack=stack, sec=sec) for sec in range(first_sec, last_sec+1))
