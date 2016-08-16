from utilities2015 import *

########### Data Directories #############

import subprocess
hostname = subprocess.check_output("hostname", shell=True).strip()

if hostname.endswith('sdsc.edu'):
    print 'Setting environment for Gordon'
    data_dir = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_processed'
    atlasAlignParams_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_atlasAlignParams_atlas'
    atlasAlignOptLogs_dir = '/oasis/projects/nsf/csd395/yuncong/CSHL_atlasAlignOptLogs_atlas'
    volume_dir = '/oasis/projects/nsf/csd395/yuncong/CSHL_volumes/'
    labelingViz_root = '/oasis/projects/nsf/csd395/yuncong/CSHL_annotationsViz'
    scoremaps_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_scoremaps_lossless_svm_Sat16ClassFinetuned_v3/'
    scoremapViz_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_scoremapViz_svm_Sat16ClassFinetuned_v3'
    annotationViz_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_annotationsViz'
    annotation_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_labelings_losslessAlignCropped/'
    annotation_midbrainIncluded_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_labelings_losslessAlignCropped_midbrainIncluded/'
    patch_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_patches/'
elif hostname == 'yuncong-MacbookPro':
    print 'Setting environment for Local Macbook Pro'
    data_dir = '/media/yuncong/11846a25-2cc1-361b-a6e8-e5773e7689a8/CSHL_data_processed'
    volume_dir = '/home/yuncong/CSHL_volumes/'
    mesh_rootdir = '/home/yuncong/CSHL_meshes'
    atlasAlignParams_rootdir = '/home/yuncong/CSHL_atlasAlignParams/'
    annotation_rootdir = '/home/yuncong/CSHL_data_labelings_losslessAlignCropped/'
    annotation_midbrainIncluded_rootdir = '/home/yuncong/CSHL_data_labelings_losslessAlignCropped_midbrainIncluded/'
elif hostname == 'yuncong-Precision-WorkStation-T7500':
    print 'Setting environment for Precision WorkStation'
    data_dir = '/media/yuncong/BstemAtlasData/CSHL_data_processed/'
    volume_dir = '/home/yuncong/CSHL_volumes/'
    annotation_rootdir = '/home/yuncong/CSHL_data_labelings_losslessAlignCropped/'
    annotation_midbrainIncluded_rootdir = '/home/yuncong/CSHL_data_labelings_losslessAlignCropped_midbrainIncluded/'
else:
    print 'Setting environment for Brainstem workstation'

############ Class Labels #############

volume_landmark_names_unsided = ['12N', '5N', '6N', '7N', '7n', 'AP', 'Amb', 'LC',
                                 'LRt', 'Pn', 'R', 'RtTg', 'Tz', 'VLL', 'sp5']
linear_landmark_names_unsided = ['outerContour']

labels_unsided = volume_landmark_names_unsided + linear_landmark_names_unsided
labels_unsided_indices = dict((j, i+1) for i, j in enumerate(labels_unsided))  # BackG always 0

labelMap_unsidedToSided = {'12N': ['12N'],
                            '5N': ['5N_L', '5N_R'],
                            '6N': ['6N_L', '6N_R'],
                            '7N': ['7N_L', '7N_R'],
                            '7n': ['7n_L', '7n_R'],
                            'AP': ['AP'],
                            'Amb': ['Amb_L', 'Amb_R'],
                            'LC': ['LC_L', 'LC_R'],
                            'LRt': ['LRt_L', 'LRt_R'],
                            'Pn': ['Pn_L', 'Pn_R'],
                            'R': ['R_L', 'R_R'],
                            'RtTg': ['RtTg'],
                            'Tz': ['Tz_L', 'Tz_R'],
                            'VLL': ['VLL_L', 'VLL_R'],
                            'sp5': ['sp5'],
                           'outerContour': ['outerContour']}

labelMap_sidedToUnsided = {n: nu for nu, ns in labelMap_unsidedToSided.iteritems() for n in ns}

from itertools import chain
labels_sided = list(chain(*(labelMap_unsidedToSided[name_u] for name_u in labels_unsided)))
labels_sided_indices = dict((j, i+1) for i, j in enumerate(labels_sided)) # BackG always 0

############ Physical Dimension #############

section_thickness = 20 # in um
xy_pixel_distance_lossless = 0.46
xy_pixel_distance_tb = xy_pixel_distance_lossless * 32 # in um, thumbnail

#######################################

all_stacks = ['MD589', 'MD594', 'MD593', 'MD585', 'MD592', 'MD590', 'MD591', 'MD589', 'MD595', 'MD598', 'MD602', 'MD603']

section_number_lookup = { 'MD585': 440, 'MD589': 445,
                        'MD590': 419, 'MD591': 452, 'MD592': 454,
                        'MD593': 448, 'MD594': 432, 'MD595': 441,
                        'MD598': 430, 'MD602': 420, 'MD603': 432,
                        'MD635': 444}

section_range_lookup = {'MD585': (78, 347), 'MD589': (93, 368),
                        'MD590':(80,336), 'MD591': (98,387), 'MD592':(91,371),
                        'MD593': (69,350), 'MD594': (93, 364), 'MD595': (67,330),
                        'MD598': (95,354), 'MD602':(96,352), 'MD603':(60,347),
                        'MD635': (75, 365)}

# xmin, ymin, w, h
brainstem_bbox_lookup = {'MD585': (610,113,445,408), 'MD589':(643,145,419,367),
                        'MD590': (652,156,601,536), 'MD591': (697,194,550,665), 'MD592': (807,308,626,407),
                        'MD593': (645,128,571,500), 'MD594': (616,144,451,362), 'MD595': (645,170,735,519),
                        'MD598': (680,107,695,459), 'MD602': (641,76,761,474),  'MD603':(621,189,528,401),
                        'MD635': (773, 341, 565, 427)}

brainstem_bbox_lookup_midbrain = {'MD585': (528,113,527,408), 'MD589':(563,145,499,367),
                                'MD590':(585,142,668,550), 'MD591':(566,182,681,677), 'MD592':(702,299,731,416),
                                'MD593':(525,122,691,506), 'MD594': (553,144,514,362), 'MD595':(560,156,820,519),
                                'MD598':(582,88,793,478), 'MD602':(555,65,847,485), 'MD603':(514,189,635,401)}


# xmin, ymin, w, h
detect_bbox_lookup = {'MD585': (16,144,411,225), 'MD593': (31,120,368,240), 'MD592': (43,129,419,241), 'MD590': (45,124,411,236), 'MD591': (38,117,410,272), \
                        'MD594': (29,120,422,242), 'MD595': (60,143,437,236), 'MD598': (48,118,450,231), 'MD602': (56,117,468,219), 'MD589': (0,137,419,230), 'MD603': (0,165,528,236)}

detect_bbox_range_lookup = {'MD585': (132,292), 'MD593': (127,294), 'MD592': (147,319), 'MD590': (135,280), 'MD591': (150,315), \
                        'MD594': (143,305), 'MD595': (115,279), 'MD598': (150,300), 'MD602': (147,302), 'MD589': (150,316), 'MD603': (130,290)}

# midbrain range
# 'MD589': (115, 325),
# MD585': (110, 325),
# MD594': (123, 322),


# midline_section_lookup = {'MD589': 114, 'MD594': 119}
