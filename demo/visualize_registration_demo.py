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


metadata_cache1 = generate_metadata_cache()
print metadata_cache1['sections_to_filenames'].keys()
print DataManager.load_anchor_filename('DEMO999')

def get_structure_contours_from_structure_volumes_v3(volumes, stack, sections, 
                                                     resolution, level, sample_every=1,
                                                    use_unsided_name_as_key=False):
    """
    Re-section atlas volumes and obtain structure contours on each section.
    Resolution of output contours are in volume resolution.
    v3 supports multiple levels.

    Args:
        volumes (dict of (3D array, 3-tuple)): {structure: (volume, origin_wrt_wholebrain)}. volume is a 3d array of probability values.
        sections (list of int):
        resolution (int): resolution of input volumes.
        level (float or dict or dict of list): the cut-off probability at which surfaces are generated from probabilistic volumes. Default is 0.5.
        sample_every (int): how sparse to sample contour vertices.

    Returns:
        Dict {section: {name_s: contour vertices}}.
    """

    from collections import defaultdict
    
    structure_contours_wrt_alignedBrainstemCrop_rawResol = defaultdict(lambda: defaultdict(dict))

    converter = CoordinatesConverter(stack=stack, section_list=metadata_cache['sections_to_filenames'][stack].keys())

    converter.register_new_resolution('structure_volume', resol_um=convert_resolution_string_to_um(resolution=resolution, stack=stack))
    converter.register_new_resolution('image', resol_um=convert_resolution_string_to_um(resolution='raw', stack=stack))
    
    for name_s, (structure_volume_volResol, origin_wrt_wholebrain_volResol) in volumes.iteritems():

        converter.derive_three_view_frames(name_s, 
        origin_wrt_wholebrain_um=convert_resolution_string_to_um(resolution=resolution, stack=stack) * origin_wrt_wholebrain_volResol,
        zdim_um=convert_resolution_string_to_um(resolution=resolution, stack=stack) * structure_volume_volResol.shape[2])

        positions_of_all_sections_wrt_structureVolume = converter.convert_frame_and_resolution(
        p=np.array(sections)[:,None],
        in_wrt=('wholebrain', 'sagittal'), in_resolution='section',
        out_wrt=(name_s, 'sagittal'), out_resolution='structure_volume')[..., 2].flatten()
            
        structure_ddim = structure_volume_volResol.shape[2]
        
        valid_mask = (positions_of_all_sections_wrt_structureVolume >= 0) & (positions_of_all_sections_wrt_structureVolume < structure_ddim)
        if np.count_nonzero(valid_mask) == 0:
#             sys.stderr.write("%s, valid_mask is empty.\n" % name_s)
            continue

        positions_of_all_sections_wrt_structureVolume = positions_of_all_sections_wrt_structureVolume[valid_mask]
        positions_of_all_sections_wrt_structureVolume = np.round(positions_of_all_sections_wrt_structureVolume).astype(np.int)
        
        if isinstance(level, dict):
            level_this_structure = level[name_s]
        else:
            level_this_structure = level

        if isinstance(level_this_structure, float):
            level_this_structure = [level_this_structure]
                        
        for one_level in level_this_structure:

            contour_2d_wrt_structureVolume_sectionPositions_volResol = \
            find_contour_points_3d(structure_volume_volResol >= one_level,
                                    along_direction='sagittal',
                                    sample_every=sample_every,
                                    positions=positions_of_all_sections_wrt_structureVolume)

            for d_wrt_structureVolume, cnt_uv_wrt_structureVolume in contour_2d_wrt_structureVolume_sectionPositions_volResol.iteritems():

                contour_3d_wrt_structureVolume_volResol = np.column_stack([cnt_uv_wrt_structureVolume, np.ones((len(cnt_uv_wrt_structureVolume),)) * d_wrt_structureVolume])

    #             contour_3d_wrt_wholebrain_uv_rawResol_section = converter.convert_frame_and_resolution(
    #                 p=contour_3d_wrt_structureVolume_volResol,
    #                 in_wrt=(name_s, 'sagittal'), in_resolution='structure_volume',
    #                 out_wrt=('wholebrain', 'sagittal'), out_resolution='image_image_section')

                contour_3d_wrt_alignedBrainstemCrop_uv_rawResol_section = converter.convert_frame_and_resolution(
                    p=contour_3d_wrt_structureVolume_volResol,
                    in_wrt=(name_s, 'sagittal'), in_resolution='structure_volume',
                    out_wrt=('wholebrainXYcropped', 'sagittal'), out_resolution='image_image_section')

                assert len(np.unique(contour_3d_wrt_alignedBrainstemCrop_uv_rawResol_section[:,2])) == 1
                sec = int(contour_3d_wrt_alignedBrainstemCrop_uv_rawResol_section[0,2])

                if use_unsided_name_as_key:
                    name = convert_to_unsided_label(name_s)
                else:
                    name = name_s

                structure_contours_wrt_alignedBrainstemCrop_rawResol[sec][name][one_level] = contour_3d_wrt_alignedBrainstemCrop_uv_rawResol_section[..., :2]
        
    return structure_contours_wrt_alignedBrainstemCrop_rawResol


import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Generate images with aligned atlas structures overlaid.')

# parser.add_argument("fixed_brain_spec", type=str, help="Fixed brain name")
# parser.add_argument("moving_brain_spec", type=str, help="Moving brain name")
# parser.add_argument("registration_setting", type=int, help="Registration setting")
parser.add_argument("per_structure_alignment_spec", type=str, help="per_structure_alignment_spec, json")
parser.add_argument("-g", "--global_alignment_spec", type=str, help="global_alignment_spec, json")
# parser.add_argument("--structure_list", type=str, help="Json-encoded list of structures (unsided) (Default: all known structures)")
args = parser.parse_args()

# brain_f_spec = load_json(args.fixed_brain_spec)
# brain_m_spec = load_json(args.moving_brain_spec)
# registration_setting = args.registration_setting
per_structure_alignment_spec = load_json(args.per_structure_alignment_spec)
simpleGlobal_alignment_spec = load_json(args.global_alignment_spec)

structure_list = per_structure_alignment_spec.keys()

# import json
# if hasattr(args, 'structure_list'): 
#     structure_list = json.loads(args.structure_list)
# else:
#     # structure_list = all_known_structures
#     structure_list = ['Amb', 'SNR', '7N', '5N', '7n', 'LRt', 'Sp5C', 'SNC', 'VLL', 'SC', 'IC']

section_margin_um = 1000.
section_margin = int(section_margin_um / SECTION_THICKNESS)

stack = 'DEMO999'
# stack = brain_f_spec['name']
# valid_secmin = np.min(metadata_cache['valid_sections'][stack])
# valid_secmax = np.max(metadata_cache['valid_sections'][stack])
valid_secmin = 1
valid_secmax = 999

auto_contours_all_sec_all_structures_all_levels = defaultdict(lambda: defaultdict(dict))
simple_global_contours_all_sec_all_structures_all_levels = defaultdict(lambda: defaultdict(dict))

#######################

#     chat_structures_df = DataManager.load_annotation_v4(stack=stack, by_human=True, 
#                                                    suffix='structuresHanddrawn', 
#                                                    timestamp='latest')

#     chat_structures_df = chat_structures_df[~chat_structures_df['volume'].isnull()]

#     chat_structures, chat_structure_resolution = \
#     convert_structure_annotation_to_volume_origin_dict_v2(structures_df=chat_structures_df, 
#                                                           out_resolution='10.0um', stack=stack)

########################

for structure_m in structure_list:

    ####################################################
    
    local_alignment_spec = per_structure_alignment_spec[structure_m]
    
    vo = DataManager.load_transformed_volume_v2(alignment_spec=local_alignment_spec, 
                                                return_origin_instead_of_bbox=True,
                                               structure=structure_m)

    # prep2 because at end of get_structure_contours_from_structure_volumes_v2 we used wholebrainXYcropped
    registered_atlas_structures_wrt_wholebrainXYcropped_xysecTwoCorners = \
    load_json(os.path.join(ROOT_DIR, 'CSHL_simple_global_registration', stack + '_registered_atlas_structures_wrt_wholebrainXYcropped_xysecTwoCorners.json'))

    (_, _, secmin), (_, _, secmax) = registered_atlas_structures_wrt_wholebrainXYcropped_xysecTwoCorners[structure_m]

    atlas_structures_wrt_wholebrainWithMargin_sections = \
    range(max(secmin - section_margin, valid_secmin), min(secmax + 1 + section_margin, valid_secmax))

    levels = [0.1, 0.25, 0.5, 0.75, 0.99]

    auto_contours_all_sections_one_structure_all_levels = \
    get_structure_contours_from_structure_volumes_v3(volumes={structure_m: vo}, stack=stack, 
                                                     sections=atlas_structures_wrt_wholebrainWithMargin_sections,
                                                    resolution='10.0um', level=levels, sample_every=5)

    for sec, contours_one_structure_all_levels in sorted(auto_contours_all_sections_one_structure_all_levels.items()):
        if not is_invalid(sec=sec, stack=stack):
            for name_s, cnt_all_levels in contours_one_structure_all_levels.items():
                for level, cnt in cnt_all_levels.iteritems():
                    auto_contours_all_sec_all_structures_all_levels[sec][name_s][level] = cnt.astype(np.int)

    ####################################################


    simpleGlobal_vo = DataManager.load_transformed_volume_v2(alignment_spec=simpleGlobal_alignment_spec, 
                                                             return_origin_instead_of_bbox=True,
                                                            structure=structure_m)

    simpleGlobal_contours_all_sections_one_structure_all_levels = \
    get_structure_contours_from_structure_volumes_v3(volumes={structure_m: simpleGlobal_vo}, stack=stack, 
                                                     sections=atlas_structures_wrt_wholebrainWithMargin_sections,
                                                    resolution='10.0um', level=levels, sample_every=5)
    
    for sec, contours_one_structure_all_levels in sorted(simpleGlobal_contours_all_sections_one_structure_all_levels.items()):
        if not is_invalid(sec=sec, stack=stack):
            for name_s, cnt_all_levels in contours_one_structure_all_levels.items():
                for level, cnt in cnt_all_levels.iteritems():
                    simple_global_contours_all_sec_all_structures_all_levels[sec][name_s][level] = cnt.astype(np.int)

    ####################################

#         chat_vo = chat_structures[structure_m]

#         chat_contours_all_sections_all_structures_all_levels = \
#         get_structure_contours_from_structure_volumes_v3(volumes={structure_m: chat_vo}, stack=stack, 
#                                                          sections=atlas_structures_wrt_wholebrainWithMargin_sections,
#                                                         resolution='10.0um', level=[.5], sample_every=1)

#######################################

for sec in sorted(auto_contours_all_sec_all_structures_all_levels.keys()):

    if is_invalid(sec=sec, stack=stack):
        continue

    for version in ['NtbNormalizedAdaptiveInvertedGammaJpeg']:
#         for version in ['grayJpeg']:
#         for version in ['NtbNormalizedAdaptiveInvertedGammaJpeg', 'CHATJpeg']:
        
        try:
            img = DataManager.load_image_v2(stack=stack, prep_id=2, resol='raw', version=version, section=sec)

            viz = gray2rgb(img)

            # Draw the locally aligned structure contours in COLOR
            for name_s, cnt_all_levels in auto_contours_all_sec_all_structures_all_levels[sec].iteritems():

                for level, cnt in cnt_all_levels.iteritems():
                    cv2.polylines(viz, [cnt.astype(np.int)], isClosed=True, 
                                  color=LEVEL_TO_COLOR_LINE[level], thickness=10)
            
            # Draw the simple globally aligned structure contours in WHITE
            for name_s, cnt_all_levels in simple_global_contours_all_sec_all_structures_all_levels[sec].iteritems():

                for level, cnt in cnt_all_levels.iteritems():
                    cv2.polylines(viz, [cnt.astype(np.int)], isClosed=True, 
                                  color=(255,255,255), 
                                  thickness=10)

    # #             # Add CHAT contour
    #             if sec in chat_contours_all_sections_all_structures_all_levels:
    #                 chat_cnt = chat_contours_all_sections_all_structures_all_levels[sec][name_s][.5]
    #                 cv2.polylines(viz, [chat_cnt.astype(np.int)], isClosed=True, color=(255,255,255), thickness=20)

    #             fp = os.path.join('/home/yuncong/' + stack + '_atlas_aligned_multilevel_all_structures', version, stack + '_' + version + '_' + ('%03d' % sec) + '.jpg')
    #             print fp
    #             create_parent_dir_if_not_exists(fp)
    #             imsave(fp, viz)

            fp = os.path.join(ROOT_DIR, 'CSHL_registration_visualization', 
                              stack + '_atlas_aligned_multilevel_down16_all_structures', 
                              version, stack + '_' + version + '_' + ('%03d' % sec) + '.jpg')    
            create_parent_dir_if_not_exists(fp)
            imsave(fp, viz[::16, ::16])
 #           upload_to_s3(fp)
            
        except:
            pass
