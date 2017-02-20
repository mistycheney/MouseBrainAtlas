"""
This module stores static meta information.
"""

from utilities2015 import *

########### Data Directories #############

import subprocess
hostname = subprocess.check_output("hostname", shell=True).strip()

if hostname.endswith('sdsc.edu'):
    print 'Setting environment for Gordon'
    RAW_DATA_DIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_data'
    data_dir = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_processed'
    DATA_DIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_processed'
    thumbnail_data_dir = data_dir
    # atlasAlignParams_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_atlasAlignParams_atlas'
    # atlasAlignParams_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_atlasAlignParams_atlas_v2'
    REGISTRTION_PARAMETERS_ROOTDIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_registration_parameters'
    REGISTRATION_VIZ_ROOTDIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_registration_visualization'
    # atlasAlignOptLogs_dir = '/oasis/projects/nsf/csd395/yuncong/CSHL_atlasAlignOptLogs_atlas'
    # atlasAlignOptLogs_dir = '/oasis/projects/nsf/csd395/yuncong/CSHL_atlasAlignOptLogs_atlas_v2'
    # volume_dir = '/oasis/projects/nsf/csd395/yuncong/CSHL_volumes2/'
    # VOLUME_ROOTDIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_volumes2/'
    VOLUME_ROOTDIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_volumes'
    labelingViz_root = '/oasis/projects/nsf/csd395/yuncong/CSHL_annotationsViz'
    # scoremaps_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_scoremaps_lossless_svm_Sat16ClassFinetuned_v3/'
    # scoremaps_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_lossless_scoremaps_Sat16ClassFinetuned_v2/'
    # scoremapViz_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_scoremapViz_svm_Sat16ClassFinetuned_v3'
    # scoremapViz_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_scoremap_viz_Sat16ClassFinetuned_v2'
    # SCOREMAP_VIZ_ROOTDIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_scoremap_viz_Sat16ClassFinetuned_v2'
    annotationViz_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_annotationsViz'
    annotation_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_labelings_losslessAlignCropped/'
    # annotation_midbrainIncluded_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_labelings_losslessAlignCropped_midbrainIncluded/'
    annotation_midbrainIncluded_v2_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_labelings_v3/'
    ANNOTATION_ROOTDIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_labelings_v3/'
    # patch_features_rootdir = '/home/yuncong/csd395/CSHL_patch_features_Sat16ClassFinetuned_v2'
    patch_training_features_rootdir = '/home/yuncong/csd395/CSHL_patch_features_Sat16ClassFinetuned_v2_train'
    patch_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_patches/'
    # SVM_ROOTDIR = '/home/yuncong/csd395/CSHL_patch_features_Sat16ClassFinetuned_v2_classifiers/'
    # SVM_NISSL_ROOTDIR = '/home/yuncong/csd395/CSHL_patch_features_Sat16ClassFinetuned_v2_classifiers/'
    # SVM_NTBLUE_ROOTDIR = '/home/yuncong/csd395/CSHL_patch_features_Sat16ClassFinetuned_v2_classifiers_neurotraceBlue/'
    # CLF_ROOTDIR = '/home/yuncong/csd395/CSHL_patch_features_Sat16ClassFinetuned_v2_classifiers/'
    CLF_ROOTDIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_classifiers'
    CLF_NISSL_ROOTDIR = '/home/yuncong/csd395/CSHL_patch_features_Sat16ClassFinetuned_v2_classifiers'
    CLF_NTBLUE_ROOTDIR = '/home/yuncong/csd395/CSHL_patch_features_Sat16ClassFinetuned_v2_classifiers_neurotraceBlue'
    CELL_FEATURES_CLF_ROOTDIR = '/home/yuncong/csd395/CSHL_cells_v2/classifiers/'
    PATCH_FEATURES_ROOTDIR = '/home/yuncong/csd395/CSHL_patch_features_Sat16ClassFinetuned_v2'
    SPARSE_SCORES_ROOTDIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_sparse_scoremaps'
    # SCOREMAPS_ROOTDIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_lossless_scoremaps_Sat16ClassFinetuned_v2/'
    SCOREMAPS_ROOTDIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_dense_scoremaps'
    SCOREMAP_VIZ_ROOTDIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_scoremap_viz'
    HESSIAN_ROOTDIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_hessians'
    WORKSTATION_ROOTDIR = '/media/yuncong/BstemAtlasData/CSHL_data_processed'

elif hostname == 'yuncong-MacbookPro':
    print 'Setting environment for Local Macbook Pro'

    # REPO_DIR = '/home/yuncong/Brain' # use os.environ['REPO_DIR'] instead

    RAW_DATA_DIR = '/home/yuncong/CSHL_data'
    GORDON_RAW_DATA_DIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_data'
    data_dir = '/media/yuncong/YuncongPublic/CSHL_data_processed'
    thumbnail_data_dir = '/home/yuncong/CSHL_data_processed'
    THUMBNAIL_DATA_DIR = '/home/yuncong/CSHL_data_processed'
    gordon_thumbnail_data_dir = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_processed'
    GORDON_THUMBNAIL_DATA_DIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_processed'

    volume_dir = '/home/yuncong/CSHL_volumes2/'
    VOLUME_ROOTDIR = '/home/yuncong/CSHL_volumes2/'
    # mesh_rootdir = '/home/yuncong/CSHL_meshes'
    MESH_ROOTDIR =  '/home/yuncong/CSHL_meshes_v2'
    # atlasAlignParams_rootdir = '/home/yuncong/CSHL_atlasAlignParams/'
    atlasAlignParams_rootdir = '/home/yuncong/CSHL_atlasAlignParams_atlas_v2/'
    annotation_rootdir = '/home/yuncong/CSHL_data_labelings_losslessAlignCropped/'
    ANNOTATION_ROOTDIR = '/home/yuncong/CSHL_data_labelings_losslessAlignCropped/'
    # annotation_midbrainIncluded_rootdir = '/home/yuncong/CSHL_data_labelings_losslessAlignCropped_midbrainIncluded/'
    annotation_midbrainIncluded_v2_rootdir = '/home/yuncong/CSHL_labelings_v3/'
    cerebellum_masks_rootdir = '/home/yuncong/CSHL_cerebellum_mask_labeligns/'
elif hostname == 'yuncong-Precision-WorkStation-T7500':
    print 'Setting environment for Precision WorkStation'
    data_dir = '/media/yuncong/BstemAtlasData/CSHL_data_processed/'
    thumbnail_data_dir = data_dir
    volume_dir = '/home/yuncong/CSHL_volumes2/'
    annotation_rootdir = '/home/yuncong/CSHL_data_labelings_losslessAlignCropped/'
    # annotation_midbrainIncluded_rootdir = '/home/yuncong/CSHL_data_labelings_losslessAlignCropped_midbrainIncluded/'
    annotation_midbrainIncluded_v2_rootdir = '/home/yuncong/CSHL_labelings_v3/'
    # patch_features_rootdir = '/home/yuncong/CSHL_patch_features_Sat16ClassFinetuned_v2'
    PATCH_FEATURES_ROOTDIR = '/media/yuncong/BstemAtlasData/CSHL_patch_features'
    ANNOTATION_ROOTDIR = None
else:
    print 'Setting environment for Brainstem workstation'

#################### Name conversions ##################

def convert_to_unsided_name(name):
    return convert_name_to_unsided(name)

def convert_name_to_unsided(name):
    if '_' not in name:
        return name
    else:
        return convert_to_original_name(name)

def extract_side_from_name(name):
    if '_' in name:
        return name[-1]
    else:
        return None

def convert_to_left_name(name):
    return convert_name_to_unsided(name) + '_L'

def convert_to_right_name(name):
    return convert_name_to_unsided(name) + '_R'

def convert_to_original_name(name):
    return name.split('_')[0]

def convert_to_nonsurround_name(name):
    if 'surround' in name:
        return name[:-9]
    else:
        return name

def convert_to_surround_name(name, margin=None, suffix=None):

    elements = name.split('_')
    if margin is None:
        if len(elements) > 1 and elements[1] == 'surround':
            if suffix is not None:
                return elements[0] + '_surround_' + suffix
            else:
                return elements[0] + '_surround'
        else:
            if suffix is not None:
                return name + '_surround_' + suffix
            else:
                return name + '_surround'
    else:
        if len(elements) > 1 and elements[1] == 'surround':
            if suffix is not None:
                return elements[0] + '_surround_' + str(margin) + '_' + suffix
            else:
                return elements[0] + '_surround_' + str(margin)
        else:
            if suffix is not None:
                return name + '_surround_' + str(margin) + '_' + suffix
            else:
                return name + '_surround_' + str(margin)


############ Class Labels #############

paired_structures = ['5N', '6N', '7N', '7n', 'Amb', 'LC', 'LRt', 'Pn', 'Tz', 'VLL', 'RMC', 'SNC', 'SNR', '3N', '4N',
                    'Sp5I', 'Sp5O', 'Sp5C', 'PBG', '10N', 'VCA', 'VCP', 'DC']
singular_structures = ['AP', '12N', 'RtTg', 'sp5', 'outerContour', 'SC', 'IC']
all_known_structures = paired_structures + singular_structures
all_known_structures_sided = sum([[n] if n in singular_structures
                        else [convert_to_left_name(n), convert_to_right_name(n)]
                        for n in all_known_structures], [])

linear_landmark_names_unsided = ['outerContour']
volumetric_landmark_names_unsided = list(set(paired_structures + singular_structures) - set(linear_landmark_names_unsided))
all_landmark_names_unsided = volumetric_landmark_names_unsided + linear_landmark_names_unsided

labels_unsided = volumetric_landmark_names_unsided + linear_landmark_names_unsided
labels_unsided_indices = dict((j, i+1) for i, j in enumerate(labels_unsided))  # BackG always 0

labelMap_unsidedToSided = dict([(name, [name+'_L', name+'_R']) for name in paired_structures] + \
                            [(name, [name]) for name in singular_structures])

labelMap_sidedToUnsided = {n: nu for nu, ns in labelMap_unsidedToSided.iteritems() for n in ns}

from itertools import chain
labels_sided = list(chain(*(labelMap_unsidedToSided[name_u] for name_u in labels_unsided)))
labels_sided_indices = dict((j, i+1) for i, j in enumerate(labels_sided)) # BackG always 0

############ Physical Dimension #############

# section_thickness = 20 # in um
SECTION_THICKNESS = 20 # in um
# xy_pixel_distance_lossless = 0.46
XY_PIXEL_DISTANCE_LOSSLESS = 0.46
XY_PIXEL_DISTANCE_TB = XY_PIXEL_DISTANCE_LOSSLESS * 32 # in um, thumbnail

#######################################

all_nissl_stacks = ['MD585', 'MD589', 'MD590', 'MD591', 'MD592', 'MD593', 'MD594', 'MD595', 'MD598', 'MD599', 'MD602', 'MD603']
all_ntb_stacks = ['MD635']
all_alt_nissl_ntb_stacks = ['MD653', 'MD652', 'MD642']
# all_stacks = all_nissl_stacks + all_ntb_stacks
all_stacks = all_nissl_stacks + all_ntb_stacks + all_alt_nissl_ntb_stacks
all_annotated_nissl_stacks = ['MD585', 'MD589', 'MD594']
all_annotated_ntb_stacks = ['MD635']
all_annotated_stacks = all_annotated_nissl_stacks + all_annotated_ntb_stacks
