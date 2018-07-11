#! /usr/bin/env python

import sys
import os
import time

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg') # https://stackoverflow.com/a/3054314
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from registration_utilities import *
from annotation_utilities import *
from metadata import *
from data_manager import *

from aligner_v3 import Aligner

import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='This script computes the transform matrices and generate the transformed moving brain volumes.')

parser.add_argument("fixed_brain_spec", type=str, help="Fixed brain specification, json")
parser.add_argument("moving_brain_spec", type=str, help="Moving brain specification, json")
parser.add_argument("-r", "--registration_setting", type=int, help="Registration setting, int, defined in registration_settings.csv", default=7)
parser.add_argument("-g", "--use_simple_global", action='store_false', help="Set this flag to NOT initialize with simple global registration")
# parser.add_argument("--out_dir", type=str, help="Output directory")
args = parser.parse_args()

brain_f_spec = load_json(args.fixed_brain_spec)
brain_m_spec = load_json(args.moving_brain_spec)
registration_setting = args.registration_setting
use_simple_global = args.use_simple_global

# if hasattr(args, "out_dir"):
#     out_dir = args.out_dir
# else:
#     out_dir = None

structures_f = brain_f_spec['structure']
if isinstance(structures_f, str):
    structures_f = [structures_f]
    
structures_m = brain_m_spec['structure']
if isinstance(structures_m, str):
    structures_m = [structures_m]

if brain_f_spec['vol_type'] == 'annotationAsScore': # If Neurolucida annotation
    fixed_surroundings_have_positive_value = True
    fixed_use_surround = True
elif brain_f_spec['vol_type'] == 'score': # If detection score map
    fixed_surroundings_have_positive_value = False
    fixed_use_surround = False
else:
    raise

alignment_spec = dict(stack_m=brain_m_spec, stack_f=brain_f_spec, warp_setting=registration_setting)

brain_m_spec0 = brain_m_spec.copy()
brain_m_spec0.pop("structure")
brain_f_spec0 = brain_f_spec.copy()
brain_f_spec0.pop("structure")
simpleGlobal_alignment_spec = dict(stack_m=brain_m_spec0, stack_f=brain_f_spec0, warp_setting=0)

aligner_parameters = generate_aligner_parameters_v2(alignment_spec=alignment_spec, 
                                                    structures_m=structures_m,
                                                   fixed_structures_are_sided=True,
 fixed_surroundings_have_positive_value=fixed_surroundings_have_positive_value,
                                                   fixed_use_surround=fixed_use_surround)

aligner = Aligner(aligner_parameters['volume_fixed'], 
                  aligner_parameters['volume_moving'], 
                  labelIndexMap_m2f=aligner_parameters['label_mapping_m2f'])

aligner.compute_gradient(smooth_first=True)
aligner.set_label_weights(label_weights=aligner_parameters['label_weights_m'])

################################

if use_simple_global:
    T_atlas_wrt_canonicalAtlasSpace_subject_wrt_wholebrain_atlasResol = bp.unpack_ndarray_file('/home/yuncong/CSHL_simple_global_registration/' + brain_f_spec['name'] + '_T_atlas_wrt_canonicalAtlasSpace_subject_wrt_wholebrain_atlasResol.bp')
    aligner.set_initial_transform(T_atlas_wrt_canonicalAtlasSpace_subject_wrt_wholebrain_atlasResol)
    aligner.set_centroid(centroid_m='structure_centroid', centroid_f='centroid_m')
else:
    T_atlas_wrt_canonicalAtlasSpace_subject_wrt_wholebrain_atlasResol = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
    aligner.set_initial_transform(T_atlas_wrt_canonicalAtlasSpace_subject_wrt_wholebrain_atlasResol)
    aligner.set_centroid(centroid_m='structure_centroid', centroid_f='structure_centroid')

################################

grid_search_T, _ = aligner.do_grid_search(grid_search_iteration_number=0, grid_search_sample_number=10, 
                       std_tx=100, std_ty=100, std_tz=30, 
                       grid_search_eta=3.0, 
                       stop_radius_voxel=10, indices_m=None, parallel=True, 
                       init_T=None)

# grid_search_T = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])

# init_T = compose_alignment_parameters([T_atlas_wrt_canonicalAtlasSpace_subject_wrt_wholebrain_atlasResol, grid_search_T])

init_T = grid_search_T
aligner.set_initial_transform(init_T)
aligner.set_centroid(centroid_m='structure_centroid', centroid_f='centroid_m')

########################################

t = time.time()
_, _ = aligner.optimize(tf_type=aligner_parameters['transform_type'], 
                             max_iter_num=1000,
                             history_len=100, 
                             terminate_thresh_trans=.01,
                            terminate_thresh_rot=.01,
                             full_lr=np.array([1,1,1,.01,.01,.01]),
                            )
sys.stderr.write("Optimize: %.2f seconds.\n" % (time.time() - t))

# plot_alignment_results(traj=aligner.Ts, scores=aligner.scores, select_best='max_value')

# DataManager.save_alignment_results_v3(aligner=aligner, 
#                                       select_best='max_value',
#                                       alignment_spec=alignment_spec,
#                                       )

tf_atlas_to_subj = compose_alignment_parameters([init_T, convert_transform_forms(aligner=aligner, out_form=(3,4))])

print tf_atlas_to_subj

DataManager.save_alignment_results_v3(transform_parameters=convert_transform_forms(transform=tf_atlas_to_subj, out_form='dict'),
                   score_traj=aligner.scores,
                   parameter_traj=aligner.Ts,
                  alignment_spec=alignment_spec,
                                     upload_s3=False)

# Transform moving structures. Save transformed version.

for structure_m in structures_m:

    for s in [structure_m, convert_to_surround_name(name=structure_m, margin='200um')]:
    
        stack_m_spec = dict(name='atlasV7',
                   vol_type='score',
                   structure=s,
                    resolution='10.0um'
                   )

    #     stack_f_spec = dict(name=stack,
    #                        vol_type='score',
    #                        detector_id=detector_id,
    #                        structure=convert_to_original_name(structure_m),
    #                         resolution='10.0um'
    #                        )

        # local_alignment_spec = dict(stack_m=stack_m_spec, 
        #                       stack_f=stack_f_spec,
        #                       warp_setting=registration_setting)

        # DataManager.save_alignment_results_v3(transform_parameters=convert_transform_forms(transform=tf_atlas_to_subj, out_form='dict'),
        #                score_traj=aligner.scores,
        #                parameter_traj=aligner.Ts,
        #               alignment_spec=local_alignment_spec)

        # tf_atlas_to_subj = DataManager.load_alignment_results_v3(local_alignment_spec, what='parameters', out_form=(4,4))

        atlas_structure_wrt_canonicalAtlasSpace_atlasResol = \
        DataManager.load_original_volume_v2(stack_spec=stack_m_spec, bbox_wrt='canonicalAtlasSpace', structure=s)

        aligned_structure_wrt_wholebrain_inputResol = \
        transform_volume_v4(volume=atlas_structure_wrt_canonicalAtlasSpace_atlasResol,
                            transform=tf_atlas_to_subj,
                            return_origin_instead_of_bbox=True)

        DataManager.save_transformed_volume_v2(volume=aligned_structure_wrt_wholebrain_inputResol, 
                                               alignment_spec=alignment_spec,
                                              structure=s,
                                               upload_s3=False)

        ###############################


        aligned_structure_wrt_wholebrain_inputResol = \
        transform_volume_v4(volume=atlas_structure_wrt_canonicalAtlasSpace_atlasResol,
                            transform=init_T,
                            return_origin_instead_of_bbox=True)

        DataManager.save_transformed_volume_v2(volume=aligned_structure_wrt_wholebrain_inputResol, 
                                               alignment_spec=simpleGlobal_alignment_spec,
                                              structure=s,
                                               upload_s3=False)
