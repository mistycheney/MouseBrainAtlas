#! /usr/bin/env python

import os
import argparse
import sys
import time

from joblib import Parallel, delayed

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from metadata import *
from data_manager import *

import numpy as np

from learning_utilities import *

from sklearn.svm import SVC
from sklearn.externals import joblib

stack = sys.argv[1]
first_sec = int(sys.argv[2])
last_sec = int(sys.argv[3])
train_sample_scheme = int(sys.argv[4])

####################################################

paired_structures = ['5N', '6N', '7N', '7n', 'Amb', 'LC', 'LRt', 'Pn', 'Tz', 'VLL', 'RMC', 'SNC', 'SNR', '3N', '4N',
                    'Sp5I', 'Sp5O', 'Sp5C', 'PBG', '10N', 'VCA', 'VCP', 'DC']
singular_structures = ['AP', '12N', 'RtTg', 'SC', 'IC']
structures = paired_structures + singular_structures

# Load pre-computed svm classifiers
# train_sample_scheme = 1
# svm_suffix = 'trainSampleScheme_%d'%train_sample_scheme

# svc_allClasses = {}
# for label in structures:
#     svc_allClasses[label] = joblib.load(DataManager.get_svm_filepath(label=label, train_sample_scheme=train_sample_scheme))

svc_allClasses = {}
for label in structures:
    try:
        if stack in ['MD635']:
            # Neurotrace blue
            svc_allClasses[label] = joblib.load(DataManager.get_svm_neurotraceBlue_filepath(label=label, train_sample_scheme=train_sample_scheme))
        else:
            # regular nissl
            svc_allClasses[label] = joblib.load(DataManager.get_svm_filepath(label=label, train_sample_scheme=train_sample_scheme))
    except Exception as e:
        sys.stderr.write('%s\n' % e)
        sys.stderr.write('Detector for %s is not trained.\n' % label)

structures = svc_allClasses.keys()

filenames_to_sections, sections_to_filenames = DataManager.load_sorted_filenames(stack)
# first_sec, last_sec = DataManager.load_cropbox(stack)[4:]
anchor_fn = DataManager.load_anchor_filename(stack)

def svm_predict(stack, sec):
    fn = sections_to_filenames[sec]
    if fn in ['Nonexisting', 'Rescan', 'Placeholder']:
        return

    feature_fn = PATCH_FEATURES_ROOTDIR + '/%(stack)s/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_features.hdf' % dict(stack=stack, fn=fn, anchor_fn=anchor_fn)

    try:
        features = load_hdf(feature_fn)
    except Exception as e:
        sys.stderr.write(e.message + '\n')
        return

    # output_dir = create_if_not_exists(os.path.join(SPARSE_SCORES_ROOTDIR, stack, '%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped' % \
    #                                   {'fn': fn, 'anchor_fn': anchor_fn}))

    for label in structures:
        svc = svc_allClasses[label]
        probs = svc.predict_proba(features)[:, svc.classes_.tolist().index(1.)]
        output_fn = DataManager.get_sparse_scores_filepath(stack=stack, fn=fn, anchor_fn=anchor_fn, label=label, train_sample_scheme=train_sample_scheme)
        create_if_not_exists(os.path.dirname(output_fn))
        # output_fn = output_dir + '/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_%(label)s_sparseScores_trainSampleScheme_%(scheme)d.hdf' % \
        #             {'fn': fn, 'anchor_fn': anchor_fn, 'label':label, 'scheme': train_sample_scheme}
        bp.pack_ndarray_file(probs, output_fn)


t = time.time()
Parallel(n_jobs=8)(delayed(svm_predict)(stack=stack, sec=sec) for sec in range(first_sec, last_sec+1))
sys.stderr.write('svm predict: %.2f seconds\n' % (time.time() - t)) # 35 s / 10 section; 863 seconds /stack
