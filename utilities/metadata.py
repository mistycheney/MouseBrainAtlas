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
    atlasAlignParams_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_atlasAlignParams_atlas_v2'
    # atlasAlignOptLogs_dir = '/oasis/projects/nsf/csd395/yuncong/CSHL_atlasAlignOptLogs_atlas'
    atlasAlignOptLogs_dir = '/oasis/projects/nsf/csd395/yuncong/CSHL_atlasAlignOptLogs_atlas_v2'
    # volume_dir = '/oasis/projects/nsf/csd395/yuncong/CSHL_volumes2/'
    VOLUME_ROOTDIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_volumes2/'
    labelingViz_root = '/oasis/projects/nsf/csd395/yuncong/CSHL_annotationsViz'
    # scoremaps_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_scoremaps_lossless_svm_Sat16ClassFinetuned_v3/'
    scoremaps_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_lossless_scoremaps_Sat16ClassFinetuned_v2/'
    # scoremapViz_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_scoremapViz_svm_Sat16ClassFinetuned_v3'
    # scoremapViz_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_scoremap_viz_Sat16ClassFinetuned_v2'
    SCOREMAP_VIZ_ROOTDIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_scoremap_viz_Sat16ClassFinetuned_v2'
    annotationViz_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_annotationsViz'
    annotation_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_labelings_losslessAlignCropped/'
    # annotation_midbrainIncluded_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_labelings_losslessAlignCropped_midbrainIncluded/'
    annotation_midbrainIncluded_v2_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_labelings_v3/'
    # patch_features_rootdir = '/home/yuncong/csd395/CSHL_patch_features_Sat16ClassFinetuned_v2'
    patch_training_features_rootdir = '/home/yuncong/csd395/CSHL_patch_features_Sat16ClassFinetuned_v2_train'
    patch_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_patches/'
    SVM_ROOTDIR = '/home/yuncong/csd395/CSHL_patch_features_Sat16ClassFinetuned_v2_classifiers/'
    PATCH_FEATURES_ROOTDIR = '/home/yuncong/csd395/CSHL_patch_features_Sat16ClassFinetuned_v2'
    SPARSE_SCORES_ROOTDIR = '/home/yuncong/csd395/CSHL_patch_Sat16ClassFinetuned_v2_predictions'
    SCOREMAPS_ROOTDIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_lossless_scoremaps_Sat16ClassFinetuned_v2/'
    HESSIAN_ROOTDIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_hessians/'
elif hostname == 'yuncong-MacbookPro':
    print 'Setting environment for Local Macbook Pro'
    REPO_DIR = '/home/yuncong/Brain'
    data_dir = '/media/yuncong/YuncongPublic/CSHL_data_processed'
    thumbnail_data_dir = '/home/yuncong/CSHL_data_processed'
    volume_dir = '/home/yuncong/CSHL_volumes2/'
    VOLUME_ROOTDIR = '/home/yuncong/CSHL_volumes2/'
    # mesh_rootdir = '/home/yuncong/CSHL_meshes'
    MESH_ROOTDIR =  '/home/yuncong/CSHL_meshes_v2'
    # atlasAlignParams_rootdir = '/home/yuncong/CSHL_atlasAlignParams/'
    atlasAlignParams_rootdir = '/home/yuncong/CSHL_atlasAlignParams_atlas_v2/'
    annotation_rootdir = '/home/yuncong/CSHL_data_labelings_losslessAlignCropped/'
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
    patch_features_rootdir = '/home/yuncong/CSHL_patch_features_Sat16ClassFinetuned_v2'
else:
    print 'Setting environment for Brainstem workstation'

############ Class Labels #############

paired_structures = ['5N', '6N', '7N', '7n', 'Amb', 'LC', 'LRt', 'Pn', 'Tz', 'VLL', 'R', 'SNC', 'SNR', '3N', '4N']
singular_structures = ['AP', '12N', 'RtTg', 'sp5', 'outerContour', 'SC', 'IC']

# volume_landmark_names_unsided = ['12N', '5N', '6N', '7N', '7n', 'AP', 'Amb', 'LC',
#                                  'LRt', 'Pn', 'R', 'RtTg', 'Tz', 'VLL', 'sp5']
linear_landmark_names_unsided = ['outerContour']
volumetric_landmark_names_unsided = list(set(paired_structures + singular_structures) - set(linear_landmark_names_unsided))
all_landmark_names_unsided = volumetric_landmark_names_unsided + linear_landmark_names_unsided

labels_unsided = volumetric_landmark_names_unsided + linear_landmark_names_unsided
labels_unsided_indices = dict((j, i+1) for i, j in enumerate(labels_unsided))  # BackG always 0

def convert_name_to_unsided(name):
    if '_' not in name:
        return name
    else:
        return name[:-2]

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

def convert_to_surround_name(name):
    if 'surround' in name:
        return name
    else:
        return name + '_surround'

labelMap_unsidedToSided = dict([(name, [name+'_L', name+'_R']) for name in paired_structures] + \
                            [(name, [name]) for name in singular_structures])

# labelMap_unsidedToSided = {'12N': ['12N'],
#                             '5N': ['5N_L', '5N_R'],
#                             '6N': ['6N_L', '6N_R'],
#                             '7N': ['7N_L', '7N_R'],
#                             '7n': ['7n_L', '7n_R'],
#                             'AP': ['AP'],
#                             'Amb': ['Amb_L', 'Amb_R'],
#                             'LC': ['LC_L', 'LC_R'],
#                             'LRt': ['LRt_L', 'LRt_R'],
#                             'Pn': ['Pn_L', 'Pn_R'],
#                             'R': ['R_L', 'R_R'],
#                             'RtTg': ['RtTg'],
#                             'Tz': ['Tz_L', 'Tz_R'],
#                             'VLL': ['VLL_L', 'VLL_R'],
#                             'sp5': ['sp5'],
#
#                            'outerContour': ['outerContour']}

labelMap_sidedToUnsided = {n: nu for nu, ns in labelMap_unsidedToSided.iteritems() for n in ns}

from itertools import chain
labels_sided = list(chain(*(labelMap_unsidedToSided[name_u] for name_u in labels_unsided)))
labels_sided_indices = dict((j, i+1) for i, j in enumerate(labels_sided)) # BackG always 0

############ Physical Dimension #############

section_thickness = 20 # in um
xy_pixel_distance_lossless = 0.46
xy_pixel_distance_tb = xy_pixel_distance_lossless * 32 # in um, thumbnail

#######################################

all_stacks = ['MD585', 'MD589', 'MD590', 'MD591', 'MD592', 'MD593', 'MD594', 'MD595', 'MD598', 'MD599', 'MD602', 'MD603',
'MD635']

# section_number_lookup = { 'MD585': 440, 'MD589': 445,
#                         'MD590': 419, 'MD591': 452, 'MD592': 454,
#                         'MD593': 448, 'MD594': 432, 'MD595': 441,
#                         'MD598': 430, 'MD602': 420, 'MD603': 432,
#                         'MD635': 444, 'MD634': 729}

# section_range_lookup = {'MD585': (78, 347), 'MD589': (93, 368),
#                         'MD590':(80,336), 'MD591': (98,387), 'MD592':(91,371),
#                         'MD593': (69,350), 'MD594': (93, 364), 'MD595': (67,330),
#                         'MD598': (95,354), 'MD602':(96,352), 'MD603':(60,347),
#                         'MD635': (75, 365), 'MD634': (312, 729)}

# section_range_lookup = {'MD585': (83, 352), 'MD589': (92, 370), 'MD590': (82, 343), 'MD594': (92, 367), 'MD602':(111, 375), 'MD603':(61, 353)}
#
# stack_orientation = {'MD585': 'sagittal', 'MD589': 'sagittal',
#                         'MD590':'sagittal', 'MD591':'sagittal', 'MD592':'sagittal',
#                         'MD593':'sagittal', 'MD594': 'sagittal', 'MD595': 'sagittal',
#                         'MD598': 'sagittal', 'MD602':'sagittal', 'MD603':'sagittal',
#                         'MD635': 'horizontal', 'MD634': 'coronal'}

# anchor_filename = {'MD585':'MD585-N47-2015.07.16-22.50.52_MD585_3_0141',
# 'MD589': 'MD589-IHC31-2015.07.30-23.26.22_MD589_1_0091',
# 'MD594': 'MD594-N58-2015.08.27-00.19.01_MD594_1_0172',
# 'MD602': 'MD602-N49-2015.12.01-18.41.46_MD602_2_0146',
# 'MD603': 'MD603-N60-2015.12.03-23.57.05_MD603_2_0179',
# 'MD590': 'MD590-N47-2015.09.12-05.32.06_MD590_2_0140'}

# xmin, ymin, w, h (on original uncropped sections)
# brainstem_bbox_lookup = {'MD585': (610,113,445,408), 'MD589':(643,145,419,367),
#                         'MD590': (652,156,601,536), 'MD591': (697,194,550,665), 'MD592': (807,308,626,407),
#                         'MD593': (645,128,571,500), 'MD594': (616,144,451,362), 'MD595': (645,170,735,519),
#                         'MD598': (680,107,695,459), 'MD602': (641,76,761,474),  'MD603':(621,189,528,401),
#                         'MD635': (773, 341, 565, 427)}

# brainstem_bbox_lookup_midbrain = {'MD585': (528,113,527,408), 'MD589':(563,145,499,367),
#                                 'MD590':(585,142,668,550), 'MD591':(566,182,681,677), 'MD592':(702,299,731,416),
#                                 'MD593':(525,122,691,506), 'MD594': (553,144,514,362), 'MD595':(560,156,820,519),
#                                 'MD598':(582,88,793,478), 'MD602':(555,65,847,485), 'MD603':(514,189,635,401)}


# brainstem_bbox_lookup_midbrain = {'MD585': (551,116,512,375), 'MD589':(569,140,485,373), 'MD590': (620, 125, 548, 408),
#                                 'MD594':(583,120,544,368),
#                                 'MD602': (603,36,698,384), 'MD603': (588,179,654,421)}

# xmin, ymin, w, h
# detect_bbox_lookup = {'MD585': (16,144,411,225), 'MD593': (31,120,368,240), 'MD592': (43,129,419,241), 'MD590': (45,124,411,236), 'MD591': (38,117,410,272), \
#                         'MD594': (29,120,422,242), 'MD595': (60,143,437,236), 'MD598': (48,118,450,231), 'MD602': (56,117,468,219), 'MD589': (0,137,419,230), 'MD603': (0,165,528,236)}
#
# detect_bbox_range_lookup = {'MD585': (132,292), 'MD593': (127,294), 'MD592': (147,319), 'MD590': (135,280), 'MD591': (150,315), \
#                         'MD594': (143,305), 'MD595': (115,279), 'MD598': (150,300), 'MD602': (147,302), 'MD589': (150,316), 'MD603': (130,290)}

# midbrain range
# 'MD589': (115, 325),
# MD585': (110, 325),
# MD594': (123, 322),


# midline_section_lookup = {'MD589': 114, 'MD594': 119}
