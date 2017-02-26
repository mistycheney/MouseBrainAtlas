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
    available_classifiers = {2: DataManager.load_classifiers(setting=2),
                             10: DataManager.load_classifiers(setting=10)}
else:
    available_classifiers = {setting: DataManager.load_classifiers(setting=setting)}

def clf_predict(stack, sec):

    if is_invalid(metadata_cache['sections_to_filenames'][stack][sec]):
        return

    try:
        features = DataManager.load_dnn_features(stack=stack, model_name='Sat16ClassFinetuned', section=sec)
    except Exception as e:
        sys.stderr.write('%s\n' % e.message)
        return

    actual_setting = resolve_actual_setting(setting=setting, stack=stack, sec=sec)
    clf_allClasses_ = available_classifiers[actual_setting]

    for structure, clf in clf_allClasses_.iteritems():

        probs = clf.predict_proba(features)[:, clf.classes_.tolist().index(1.)]

        output_fn = DataManager.get_sparse_scores_filepath(stack=stack, structure=structure, setting=actual_setting, sec=sec)
        create_if_not_exists(os.path.dirname(output_fn))

        bp.pack_ndarray_file(probs, output_fn)


t = time.time()

pool = Pool(8)
pool.map(lambda sec: clf_predict(stack=stack, sec=sec), range(first_sec, last_sec+1))
pool.close()
pool.join()

sys.stderr.write('Classifier predict: %.2f\n' % (time.time()-t))
