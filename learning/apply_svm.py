#! /usr/bin/env python

import sys, os
from sklearn.svm import SVC
from sklearn.externals import joblib
import time

from joblib import Parallel, delayed

stack = sys.argv[1]
first_sec = int(sys.argv[2])
last_sec = int(sys.argv[3])

# input
features_dir = '/home/yuncong/csd395/CSHL_patch_features'

# output
predictions_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_patch_predictions_svm'
if not os.path.exists(predictions_rootdir):
    os.makedirs(predictions_rootdir)
    
labels = ['BackG', '5N', '7n', '7N', '12N', 'Pn', 'VLL', 
          '6N', 'Amb', 'R', 'Tz', 'RtTg', 'LRt', 'LC', 'AP', 'sp5']

svc_allClasses = {}
for label_ind, label in enumerate(labels[1:]):
    svc_allClasses[label] = joblib.load('svm/%(label)s_svm.pkl' % {'label': label})
    
def f(stack, sec):
    
    test_features_dir = features_dir + '/%(stack)s' % {'stack': stack, 'sec': sec}
    features_roi = load_hdf(test_features_dir + '/%(stack)s_%(sec)04d_features.hdf' % {'stack': stack, 'sec': sec})

    predictions_dir = predictions_rootdir + '/%(stack)s/%(sec)04d' % {'stack': stack, 'sec': sec}
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)
    
    for label_ind, label in enumerate(labels[1:]):
        svc = svc_allClasses[label]
        probs = svc.predict_proba(features_roi)[:, svc.classes_.tolist().index(1.)]
    
        np.save(predictions_dir + '/%(stack)s_%(sec)04d_roi1_%(label)s_scores.npy' % \
                {'stack': stack, 'sec': sec, 'label': label}, probs)

Parallel(n_jobs=16)(delayed(f)(stack=stack, sec=sec) for sec in range(first_sec, last_sec+1))