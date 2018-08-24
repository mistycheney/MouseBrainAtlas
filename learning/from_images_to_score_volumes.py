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

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='')

parser.add_argument("brain_name", type=str, help="Brain name")
parser.add_argument("detector_id", type=int, help="Detector id")
parser.add_argument("--structure_list", type=str, help="Json-encoded list of structures (unsided) (Default: all known structures)")
args = parser.parse_args()

stack = args.brain_name
detector_id = args.detector_id

import json
if hasattr(args, 'structure_list'): 
    structure_list = json.loads(args.structure_list)
else:
    # structure_list = all_known_structures
    structure_list = ['Amb', 'SNR', '7N', '5N', '7n', 'LRt', 'Sp5C', 'SNC', 'VLL', 'SC', 'IC']
    
atlas_spec = dict(name='atlasV6',
                   vol_type='score'    ,               
                    resolution='10.0um'
                   )

atlas_structures_wrt_canonicalAtlasSpace_atlasResol = \
DataManager.load_original_volume_all_known_structures_v3(atlas_spec, in_bbox_wrt='canonicalAtlasSpace',
                                                        out_bbox_wrt='canonicalAtlasSpace')


# # Define 3-D ROI for which to compute scores based on simpleGlobal registered atlas.

# T_atlas_wrt_canonicalAtlasSpace_subject_wrt_wholebrain_atlasResol = bp.unpack_ndarray_file('/home/yuncong/' + stack + '_T_atlas_wrt_canonicalAtlasSpace_subject_wrt_wholebrain_atlasResol.bp')

# registered_atlas_structures_wrt_wholebrainWithMargin_atlasResol = \
# {name_s: transform_volume_v4(volume=vo, transform=T_atlas_wrt_canonicalAtlasSpace_subject_wrt_wholebrain_atlasResol, return_origin_instead_of_bbox=True)
# for name_s, vo in atlas_structures_wrt_canonicalAtlasSpace_atlasResol.iteritems()}

# registered_atlas_structures_bbox_wrt_wholebrainWithMargin_atlasResol = \
# {name_s: (o[0], o[0] + v.shape[1] - 1, o[1], o[1] + v.shape[0] - 1, o[2], o[2] + v.shape[2] - 1)
#  for name_s, (v, o) in registered_atlas_structures_wrt_wholebrainWithMargin_atlasResol.iteritems()}

# registered_atlas_structures_xyzTwoCorners_wrt_wholebrainWithMargin_atlasResol = \
# {name_s: ((o[0], o[2], o[4]), (o[1], o[3], o[5]))
# for name_s, o in registered_atlas_structures_bbox_wrt_wholebrainWithMargin_atlasResol.iteritems()}

# registered_atlas_structures_xyzCorners_wrt_wholebrainWithMargin_atlasResol = \
# {name_s: ((o[0], o[2], o[4]), (o[0], o[2], o[5]), (o[0], o[3], o[4]), (o[0], o[3], o[5]), \
#          (o[1], o[2], o[4]), (o[1], o[2], o[5]), (o[1], o[3], o[4]), (o[1], o[3], o[5]))
# for name_s, o in registered_atlas_structures_bbox_wrt_wholebrainWithMargin_atlasResol.iteritems()}

# from data_manager import CoordinatesConverter
# converter = CoordinatesConverter(stack=stack)
# registered_atlas_structures_wrt_wholebrainXYcropped_xysecTwoCorners = {}

# for name_s, corners_xyz in registered_atlas_structures_xyzTwoCorners_wrt_wholebrainWithMargin_atlasResol.iteritems():
# #     print name_s
#     registered_atlas_structures_wrt_wholebrainXYcropped_xysecTwoCorners[name_s] = \
#     converter.convert_frame_and_resolution(p=corners_xyz, 
#                                        in_wrt=('wholebrainWithMargin', 'sagittal'),
#                                       in_resolution='10.0um',
#                                       out_wrt=('wholebrainXYcropped', 'sagittal'),
#                                       out_resolution='image_image_section').astype(np.int)

# save_json(registered_atlas_structures_wrt_wholebrainXYcropped_xysecTwoCorners, 
#           '/home/yuncong/' + stack + '_registered_atlas_structures_wrt_wholebrainXYcropped_xysecTwoCorners.json')


# Compute score maps.

batch_size = 256
model_dir_name = 'inception-bn-blue'
model_name = 'inception-bn-blue'
model, mean_img = load_mxnet_model(model_dir_name=model_dir_name, model_name=model_name, 
                                   num_gpus=1, batch_size=batch_size)

# detector_id = 19 # For CSHL nissl data. e.g. MD589, denser window
detector_setting = detector_settings.loc[detector_id]
clfs = DataManager.load_classifiers(classifier_id=detector_setting['feature_classifier_id'])
win_id = detector_setting['windowing_id']

output_resolution = '10.0um'
out_resolution_um = convert_resolution_string_to_um(resolution=output_resolution, stack=stack)

valid_secmin = np.min(metadata_cache['valid_sections'][stack])
valid_secmax = np.max(metadata_cache['valid_sections'][stack])

registered_atlas_structures_wrt_wholebrainXYcropped_xysecTwoCorners = \
load_json('/home/yuncong/' + stack + '_registered_atlas_structures_wrt_wholebrainXYcropped_xysecTwoCorners.json')

######## Identify ROI based on simple global alignment ########

registered_atlas_structures_wrt_wholebrainXYcropped_bboxes_perSection = defaultdict(dict)

section_margin_um = 400.
section_margin = int(section_margin_um / SECTION_THICKNESS)

image_margin_um = 2000.
image_margin = int(np.round(image_margin_um / convert_resolution_string_to_um('raw', stack)))


for name_u in structure_list:

    if name_u in singular_structures:

        (xmin, ymin, secmin), (xmax, ymax, secmax) = registered_atlas_structures_wrt_wholebrainXYcropped_xysecTwoCorners[name_u]

        for sec in range(max(secmin - section_margin, valid_secmin), min(secmax + 1 + section_margin, valid_secmax)):

            if is_invalid(sec=sec, stack=stack):
                continue

            registered_atlas_structures_wrt_wholebrainXYcropped_bboxes_perSection[name_u][sec] = \
            (max(xmin - image_margin, 0), 
             xmax + image_margin, 
             max(ymin - image_margin, 0), 
             ymax + image_margin)
    else:

        a = defaultdict(list)

        lname = convert_to_left_name(name_u)        
        (xmin, ymin, secmin), (xmax, ymax, secmax) = registered_atlas_structures_wrt_wholebrainXYcropped_xysecTwoCorners[lname]

        for sec in range(max(secmin - section_margin, valid_secmin), min(secmax + 1 + section_margin, valid_secmax)):

            if is_invalid(sec=sec, stack=stack):
                continue

            a[sec].append((max(xmin - image_margin, 0), 
             xmax + image_margin, 
             max(ymin - image_margin, 0), 
             ymax + image_margin))

        rname = convert_to_right_name(name_u)
        (xmin, ymin, secmin), (xmax, ymax, secmax) = registered_atlas_structures_wrt_wholebrainXYcropped_xysecTwoCorners[rname]

        for sec in range(max(secmin - section_margin, valid_secmin), min(secmax + 1 + section_margin, valid_secmax)):

            if is_invalid(sec=sec, stack=stack):
                continue

            a[sec].append((max(xmin - image_margin, 0), 
             xmax + image_margin, 
             max(ymin - image_margin, 0), 
             ymax + image_margin))

        for sec, bboxes in a.iteritems():
            if len(bboxes) == 1:
                registered_atlas_structures_wrt_wholebrainXYcropped_bboxes_perSection[name_u][sec] = bboxes[0]
            else:
                xmin, ymin = np.min(bboxes, axis=0)[[0,2]]
                xmax, ymax = np.max(bboxes, axis=0)[[1,3]]
                registered_atlas_structures_wrt_wholebrainXYcropped_bboxes_perSection[name_u][sec] = (xmin, xmax, ymin, ymax)


######### Generate score maps ###########

for name_u in structure_list:

    for sec, bbox in sorted(registered_atlas_structures_wrt_wholebrainXYcropped_bboxes_perSection[name_u].items()):

#         if is_invalid(sec=sec, stack=stack):
#             continue

        print name_u, sec

        try:

            ############# Generate both scoremap and viz #################

            viz_all_landmarks, scoremap_all_landmarks = \
            draw_scoremap(clfs={name_u: clfs[name_u]}, 
                                  bbox=bbox,
                            scheme='none', 
                        win_id=win_id, prep_id=2,
                        stack=stack, 
                          return_what='both',
                          sec=sec,
                        model=model, model_name=model_name,
                         mean_img=mean_img, 
                         batch_size=batch_size,
                          output_patch_size=224,
                          is_nissl=False,
                       out_resolution_um=out_resolution_um,
                    image_shape=metadata_cache['image_shape'][stack],
                                  return_wholeimage=True)

            sm = scoremap_all_landmarks[name_u]
            viz = viz_all_landmarks[name_u]

            t = time.time()
            scoremap_bp_filepath = \
            DataManager.get_downscaled_scoremap_filepath(stack=stack, section=sec, 
                                                         structure=name_u,
                                                         detector_id=detector_id,
                                                         out_resolution_um=out_resolution_um)
            save_data(sm.astype(np.float16), scoremap_bp_filepath, upload_s3=False)
            sys.stderr.write('Save scoremap: %.2f seconds\n' % (time.time() - t))
            
            t = time.time()
            viz_filepath = \
            DataManager.get_scoremap_viz_filepath_v2(stack=stack, section=sec, 
                                                         structure=name_u,
                                                         detector_id=detector_id,
                                                         out_resolution=output_resolution)
            save_data(viz, viz_filepath, upload_s3=False)
            sys.stderr.write('Save scoremap viz: %.2f seconds\n' % (time.time() - t))

            del viz_all_landmarks, scoremap_all_landmarks

            ################ Generate scoremap only ################

#                 scoremap_all_landmarks = \
#                 draw_scoremap(clfs={name_u: clfs[name_u]}, 
#                                       bbox=bbox,
#                                 scheme='none', 
#                             win_id=win_id, prep_id=2,
#                             stack=stack, 
#                               return_what='scoremap',
#                               sec=sec,
#                             model=model, model_name=model_name,
#                              mean_img=mean_img, 
#                              batch_size=batch_size,
#                               output_patch_size=224,
#                               is_nissl=False,
#                            out_resolution_um=out_resolution_um,
#                         image_shape=metadata_cache['image_shape'][stack],
#                                       return_wholeimage=True)

#                 sm = scoremap_all_landmarks[name_u]

#                 scoremap_bp_filepath = \
#                 DataManager.get_downscaled_scoremap_filepath(stack=stack, section=sec, 
#                                                              structure=name_u,
#                                                              detector_id=detector_id,
#                                                              out_resolution_um=out_resolution_um)
#                 save_data(sm.astype(np.float16), scoremap_bp_filepath, upload_s3=False)

#                 del scoremap_all_landmarks

        except Exception as e:
            sys.stderr.write('%s\n' % e)
            continue


######### Generate score volumes ##########

#     for name_u in all_known_structures:

    for name_s in [convert_to_left_name(name_u), convert_to_right_name(name_u)]:

        scoremaps = {}

#         for sec in metadata_cache['valid_sections'][stack]:

        (xmin, ymin, s1), (xmax, ymax, s2) = registered_atlas_structures_wrt_wholebrainXYcropped_xysecTwoCorners[name_s]

        for sec in range(max(s1 - section_margin, metadata_cache['section_limits'][stack][0]), 
                         min(s2 + 1 + section_margin, metadata_cache['section_limits'][stack][1])):

            if is_invalid(sec=sec, stack=stack):
                continue

            try:
                scoremap = DataManager.load_downscaled_scoremap(stack=stack, section=sec, structure=name_u, 
                                                                prep_id='alignedBrainstemCrop',
                                                              out_resolution_um=out_resolution_um, 
                                                                detector_id=detector_id).astype(np.float32)
            except Exception as e:
                sys.stderr.write('%s\n' % e)
                continue

            mask = DataManager.load_image_v2(stack=stack, section=sec, 
                                 prep_id='alignedBrainstemCrop', 
                                 resol='thumbnail', version='mask')

            mask_outResol = rescale_by_resampling(mask, new_shape=(scoremap.shape[1], scoremap.shape[0]))

            scoremap[~mask_outResol] = 0
            scoremaps[sec] = scoremap

        t = time.time()
        volume_outVolResol, volume_origin_wrt_wholebrainXYcropped_outVolResol = \
        images_to_volume_v2(images=scoremaps, spacing_um=20.,
                                in_resol_um=out_resolution_um,
                                out_resol_um=out_resolution_um)
        sys.stderr.write('Images to volume: %.2f seconds\n' % (time.time() - t))
        
        brain_spec = dict(name=stack,
                       vol_type='score',
                        detector_id=detector_id,
                       resolution=output_resolution)

        # Save volume and origin.

        t = time.time()
        save_data(volume_outVolResol.astype(np.float16), \
                  DataManager.get_original_volume_filepath_v2(stack_spec=brain_spec, structure=name_s))

        wholebrainXYcropped_origin_wrt_wholebrain_outVolResol = \
        DataManager.get_domain_origin(stack=stack, domain='wholebrainXYcropped', 
                                      resolution=output_resolution)
        volume_origin_wrt_wholebrain_outVolResol =\
        volume_origin_wrt_wholebrainXYcropped_outVolResol + wholebrainXYcropped_origin_wrt_wholebrain_outVolResol

        save_data(volume_origin_wrt_wholebrain_outVolResol,
                  DataManager.get_original_volume_origin_filepath_v3(stack_spec=brain_spec, structure=name_s, wrt='wholebrain'))
        sys.stderr.write('Save score volume: %.2f seconds\n' % (time.time() - t))

        # Compute gradients.

        t = time.time()
        gradients = compute_gradient_v2((volume_outVolResol, volume_origin_wrt_wholebrain_outVolResol), 
                                        smooth_first=True)
        sys.stderr.write('Compute gradient: %.2f seconds\n' % (time.time() - t))

        t = time.time()
        DataManager.save_volume_gradients(gradients, stack_spec=brain_spec, structure=name_s)
        sys.stderr.write('Save gradient: %.2f seconds\n' % (time.time() - t))
