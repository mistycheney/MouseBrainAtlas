#! /usr/bin/env python

import os
import argparse
import sys
import time

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
setting = int(sys.argv[4])
# setting_nissl = int(sys.argv[4])
# setting_ntb = int(sys.argv[5])

############################

if setting == 12:

    setting_nissl = 2
    setting_ntb = 10

    clf_nissl_allClasses = DataManager.load_classifiers(setting=setting_nissl)
    clf_ntb_allClasses = DataManager.load_classifiers(setting=setting_ntb)

else:
    clf_allClasses = DataManager.load_classifiers(setting=setting)

def clf_predict(stack, sec):

    try:
        features = DataManager.load_dnn_features(stack=stack, section=sec)
    except Exception as e:
        sys.stderr.write('%s\n' % e.message)
        return

    if setting == 12:
        stain = 'nissl' # use some heuristic to decide the stain of current section

        if stain == 'nissl':
            setting_ = setting_nissl
            clf_allClasses_ = clf_nissl_allClasses
        else:
            setting_ = setting_ntb
            clf_allClasses_ = clf_ntb_allClasses
    else:
        setting_ = setting
        clf_allClasses_ = clf_allClasses

    for structure, clf in clf_allClasses_.iteritems():

        probs = clf.predict_proba(features)[:, clf.classes_.tolist().index(1.)]

        output_fn = DataManager.get_sparse_scores_filepath(stack=stack, structure=structure, setting=setting_, sec=sec)
        create_if_not_exists(os.path.dirname(output_fn))

        bp.pack_ndarray_file(probs, output_fn)


t = time.time()

pool = Pool(8)
pool.map(lambda sec: clf_predict(stack=stack, sec=sec), range(first_sec, last_sec+1))
pool.close()
pool.join()

sys.stderr.write('Classifier predict: %.2f\n' % (time.time()-t))
