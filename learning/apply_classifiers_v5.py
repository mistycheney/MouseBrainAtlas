#! /usr/bin/env python

import os
import sys
import time

import numpy as np
from multiprocess import Pool

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from metadata import *
from data_manager import *
from learning_utilities import *

###################################

import json
import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Apply classifiers')

parser.add_argument("stack", type=str, help="Stack")
parser.add_argument("filenames", type=str, help="Filenames")
parser.add_argument("detector_id", type=int, help="Detector id")
#parser.add_argument("-v", "--input_img_version", type=str, help="Input image version", default='gray')
args = parser.parse_args()

stack = args.stack
filenames = json.loads(args.filenames)
detector_id = args.detector_id
#input_img_version = args.input_img_version

detector_properties = detector_settings.loc[detector_id]
input_img_version = detector_properties['input_version']
windowing_id = int(detector_properties['windowing_id'])
model_name = detector_properties['feature_network']
classifier_id = int(detector_properties['feature_classifier_id'])

classifier_properties = classifier_settings.loc[classifier_id]
#input_img_version = classifier_properties['input_img_version']
#model_name = dataset_settings.loc[int(classifier_properties['train_set_id'].split('/')[0])]['network_model']
assert dataset_settings.loc[int(classifier_properties['train_set_id'].split('/')[0])]['network_model'] == model_name
svm_id = int(classifier_properties['svm_id'])

############################

# if classifier_id == 12:
#     available_classifiers = {2: DataManager.load_classifiers(classifier_id=2),
#                              10: DataManager.load_classifiers(classifier_id=10)}
# else:

available_classifiers = {svm_id: DataManager.load_classifiers(classifier_id=svm_id)}

def clf_predict(stack, fn):

    if is_invalid(stack=stack, fn=fn):
        return

    try:
        features = DataManager.load_dnn_features(stack=stack, model_name=model_name, fn=fn, 
                                                 win=windowing_id, input_img_version=input_img_version)
    except Exception as e:
        sys.stderr.write('%s\n' % e.message)
        return

    # actual_setting = resolve_actual_setting(setting=classifier_id, stack=stack, fn=fn)
    # clf_allClasses_ = available_classifiers[actual_setting]
    
    clf_allClasses_ = available_classifiers[svm_id]

    for structure, clf in clf_allClasses_.iteritems():

        probs = clf.predict_proba(features)[:, clf.classes_.tolist().index(1.)]
        # output_fn = DataManager.get_sparse_scores_filepath(stack=stack, structure=structure, 
        #                                                    classifier_id=actual_setting, fn=fn)
        output_fn = DataManager.get_sparse_scores_filepath(stack=stack, structure=structure, 
                                                           detector_id=detector_id, fn=fn)
        create_parent_dir_if_not_exists(output_fn)
        bp.pack_ndarray_file(probs, output_fn)
        upload_to_s3(output_fn)


t = time.time()

pool = Pool(NUM_CORES/2)
pool.map(lambda fn: clf_predict(stack=stack, fn=fn), filenames)
pool.close()
pool.join()

sys.stderr.write('Classifier predict: %.2f\n' % (time.time()-t))
