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
from sklearn.externals import joblib
from multiprocess import Pool

stack = sys.argv[1]
first_sec = int(sys.argv[2])
last_sec = int(sys.argv[3])
train_sample_scheme = int(sys.argv[4])

####################################################

paired_structures = ['5N', '6N', '7N', '7n', 'Amb', 'LC', 'LRt', 'Pn', 'Tz', 'VLL', 'RMC', 'SNC', 'SNR', '3N', '4N',
                    'Sp5I', 'Sp5O', 'Sp5C', 'PBG', '10N', 'VCA', 'VCP', 'DC']
singular_structures = ['AP', '12N', 'RtTg', 'SC', 'IC']
structures = paired_structures + singular_structures

############################

clf_ntb_allClasses = {}
for label in structures:
    try:
        clf_ntb_allClasses[label] = joblib.load(DataManager.get_classifier_neurotraceBlue_filepath(label=label, train_sample_scheme=train_sample_scheme))
    except Exception as e:
        sys.stderr.write('%s\n' % e)
        sys.stderr.write('NTB detector for %s is not trained.\n' % label)

clf_nissl_allClasses = {}
for label in structures:
    try:
        clf_nissl_allClasses[label] = joblib.load(DataManager.get_classifier_filepath(label=label, train_sample_scheme=train_sample_scheme))
    except Exception as e:
        sys.stderr.write('%s\n' % e)
        sys.stderr.write('Nissl detector for %s is not trained.\n' % label)

structures = set(clf_ntb_allClasses.keys()) | set(clf_nissl_allClasses.keys())

def clf_predict(stack, sec):

    if is_invalid(metadata_cache['sections_to_filenames'][stack][sec]):
        return

    try:
        features = DataManager.load_dnn_features(stack=stack, section=sec)
    except Exception as e:
        sys.stderr.write(e.message + '\n')
        return

    for label in structures:

        # if stack in all_ntb_stacks:
        #     clf = clf_ntb_allClasses[label]
        # elif stack in all_nissl_stacks:
        #     clf = clf_nissl_allClasses[label]
        # else:
        #     raise Exception('Not implemented.')

        clf = clf_nissl_allClasses[label]

        probs = clf.predict_proba(features)[:, clf.classes_.tolist().index(1.)]

        output_fn = DataManager.get_sparse_scores_filepath(stack=stack, sec=sec, label=label, train_sample_scheme=train_sample_scheme)
        create_if_not_exists(os.path.dirname(output_fn))

        bp.pack_ndarray_file(probs, output_fn)


t = time.time()

pool = Pool(8)
pool.map(lambda sec: clf_predict(stack=stack, sec=sec), range(first_sec, last_sec+1))
pool.close()
pool.join()

sys.stderr.write('Classifier predict: %.2f\n' % (time.time()-t)) # 35 s / 10 section; 863 seconds /stack
