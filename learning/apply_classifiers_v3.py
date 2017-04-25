#! /usr/bin/env python

import os
import argparse
import sys
import time

import numpy as np
from multiprocess import Pool

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from metadata import *
from data_manager import *
from learning_utilities import *

stack = sys.argv[1]
first_sec = int(sys.argv[2])
last_sec = int(sys.argv[3])
classifier_id = int(sys.argv[4])

############################

if classifier_id == 12:
    available_classifiers = {2: DataManager.load_classifiers(setting=2),
                             10: DataManager.load_classifiers(setting=10)}
else:
    available_classifiers = {classifier_id: DataManager.load_classifiers(setting=classifier_id)}

def clf_predict(stack, sec, model_name='Inception-BN'):

    if is_invalid(stack=stack, sec=sec):
        return

    try:
        features = DataManager.load_dnn_features(stack=stack, model_name=model_name, section=sec)
    except Exception as e:
        sys.stderr.write('%s\n' % e.message)
        return

    actual_setting = resolve_actual_setting(setting=classifier_id, stack=stack, sec=sec)
    clf_allClasses_ = available_classifiers[actual_setting]

    for structure, clf in clf_allClasses_.iteritems():

        probs = clf.predict_proba(features)[:, clf.classes_.tolist().index(1.)]
        output_fn = DataManager.get_sparse_scores_filepath(stack=stack, structure=structure, 
                                                           setting=actual_setting, sec=sec)
        create_parent_dir_if_not_exists(output_fn)
        bp.pack_ndarray_file(probs, output_fn)
        
        upload_from_ec2_to_s3(output_fn)


t = time.time()

# for sec in range(first_sec, last_sec+1):
#     clf_predict(stack=stack, sec=sec)

pool = Pool(NUM_CORES/2)
pool.map(lambda sec: clf_predict(stack=stack, sec=sec), range(first_sec, last_sec+1))
pool.close()
pool.join()

sys.stderr.write('Classifier predict: %.2f\n' % (time.time()-t))
