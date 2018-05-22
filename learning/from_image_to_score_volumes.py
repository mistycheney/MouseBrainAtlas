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

atlas_spec = dict(name='atlasV6',
                   vol_type='score'    ,               
                    resolution='10.0um'
                   )

atlas_structures_wrt_canonicalAtlasSpace_atlasResol = \
DataManager.load_original_volume_all_known_structures_v3(atlas_spec, in_bbox_wrt='canonicalAtlasSpace',
                                                        out_bbox_wrt='canonicalAtlasSpace')

# stack = 'CHATM3'
stack = 'MD589'

T_atlas_wrt_canonicalAtlasSpace_subject_wrt_wholebrain_atlasResol = bp.unpack_ndarray_file('/home/yuncong/' + stack + '_T_atlas_wrt_canonicalAtlasSpace_subject_wrt_wholebrain_atlasResol.bp')

registered_atlas_structures_wrt_wholebrainWithMargin_atlasResol = \
{name_s: transform_volume_v4(volume=vo, transform=T_atlas_wrt_canonicalAtlasSpace_subject_wrt_wholebrain_atlasResol, return_origin_instead_of_bbox=True)
for name_s, vo in atlas_structures_wrt_canonicalAtlasSpace_atlasResol.iteritems()}

registered_atlas_structures_bbox_wrt_wholebrainWithMargin_atlasResol = \
{name_s: (o[0], o[0] + v.shape[1] - 1, o[1], o[1] + v.shape[0] - 1, o[2], o[2] + v.shape[2] - 1)
 for name_s, (v, o) in registered_atlas_structures_wrt_wholebrainWithMargin_atlasResol.iteritems()}

registered_atlas_structures_xyzTwoCorners_wrt_wholebrainWithMargin_atlasResol = \
{name_s: ((o[0], o[2], o[4]), (o[1], o[3], o[5]))
for name_s, o in registered_atlas_structures_bbox_wrt_wholebrainWithMargin_atlasResol.iteritems()}

registered_atlas_structures_xyzCorners_wrt_wholebrainWithMargin_atlasResol = \
{name_s: ((o[0], o[2], o[4]), (o[0], o[2], o[5]), (o[0], o[3], o[4]), (o[0], o[3], o[5]), \
         (o[1], o[2], o[4]), (o[1], o[2], o[5]), (o[1], o[3], o[4]), (o[1], o[3], o[5]))
for name_s, o in registered_atlas_structures_bbox_wrt_wholebrainWithMargin_atlasResol.iteritems()}

from data_manager import CoordinatesConverter
converter = CoordinatesConverter(stack=stack)
registered_atlas_structures_wrt_wholebrainXYcropped_xysecTwoCorners = {}

for name_s, corners_xyz in registered_atlas_structures_xyzTwoCorners_wrt_wholebrainWithMargin_atlasResol.iteritems():
#     print name_s
    registered_atlas_structures_wrt_wholebrainXYcropped_xysecTwoCorners[name_s] = \
    converter.convert_frame_and_resolution(p=corners_xyz, 
                                       in_wrt=('wholebrainWithMargin', 'sagittal'),
                                      in_resolution='10.0um',
                                      out_wrt=('wholebrainXYcropped', 'sagittal'),
                                      out_resolution='image_image_section').astype(np.int)

save_json(registered_atlas_structures_wrt_wholebrainXYcropped_xysecTwoCorners, 
          '/home/yuncong/' + stack + '_registered_atlas_structures_wrt_wholebrainXYcropped_xysecTwoCorners.json')

batch_size = 256
model_dir_name = 'inception-bn-blue'
model_name = 'inception-bn-blue'
model, mean_img = load_mxnet_model(model_dir_name=model_dir_name, model_name=model_name, 
                                   num_gpus=1, batch_size=batch_size)


detector_id = 19 # For CSHL nissl data. e.g. MD589, denser window

detector_setting = detector_settings.loc[detector_id]

clfs = DataManager.load_classifiers(classifier_id=detector_setting['feature_classifier_id'])

win_id = detector_setting['windowing_id']

# motor_nuclei = ['Amb', '3N', '4N', '5N', '6N', '7N', '10N', '12N']
motor_nuclei = ['Amb', 'SNR', '7N', '5N', '7n', 'LRt', 'Sp5C', 'SNC', 'VLL', 'SC', 'IC']
# motor_nuclei = ['SC']

