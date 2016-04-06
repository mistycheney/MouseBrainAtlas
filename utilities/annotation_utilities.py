import numpy as np

import sys
import os

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *

import pandas as pd

from collections import defaultdict


def get_annotation_on_sections(stack=None, username=None, label_polygons=None, filtered_labels=None):
    
    assert stack is not None or label_polygons is not None
    
    if label_polygons is None:
        label_polygons = load_label_polygons_if_exists(stack, username)
    
    annotation_on_sections = {}
    
    if filtered_labels is None:
        labels = set(label_polygons.columns)
    else:
        labels = set(label_polygons.columns) & filtered_labels
    
    for l in labels:
        annotation_on_sections[l] = list(label_polygons[l].dropna().keys())
    
    return annotation_on_sections


def get_landmark_range_limits(stack=None, username=None, label_polygons=None, filtered_labels=None):
    
    assert stack is not None or label_polygons is not None
    
    if label_polygons is None:
        label_polygons = load_label_polygons_if_exists(stack, username)
        
    section_bs_begin, section_bs_end = section_range_lookup[stack]
    mid_sec = (section_bs_begin + section_bs_end + 1)/2

    landmark_limits = {}

    if filtered_labels is None:
        d = set(label_polygons.keys())
    else:
        d = set(label_polygons.keys()) & set(filtered_labels)
    
    for l in d:
        secs = label_polygons[l].dropna().keys()
        if len(secs) < 2:
            continue

#         print l
        diffs = np.diff(secs)
        peak = np.argmax(diffs)
        if diffs[peak] > 5:
            befores = secs[np.where(secs - mid_sec < 0)[0]]
            afters = secs[np.where(secs - mid_sec > 0)[0]]
#             print secs.min(), befores.max()
#             print afters.min(), secs.max()
            landmark_limits[l] = [(secs.min(), befores.max()), (afters.min(), secs.max())]
        else:
#             print secs.min(), secs.max()
            landmark_limits[l] = [(secs.min(), secs.max())]
            
    return landmark_limits
    
    
#     section_bs_begin, section_bs_end = section_range_lookup[stack]
#     mid_sec = (section_bs_begin + section_bs_end + 1)/2

#     interpolated_contours = [{} for _ in range(int(z_xy_ratio_downsampled*section_number_lookup[stack]))]
#     interpolation_limits = {}

#     for l in set(label_polygons.keys()) & set(labels):
#         secs = label_polygons[l].dropna().keys()
#         if len(secs) < 2: continue

#         print l
#     #         print secs.min(), secs.max()
#         diffs = np.diff(secs)
#         peak = np.argmax(diffs)
#         if diffs[peak] > 5:
#             befores = secs[np.where(secs - mid_sec < 0)[0]]
#             afters = secs[np.where(secs - mid_sec > 0)[0]]
#             print secs.min(), befores.max()
#             print afters.min(), secs.max()
#     #             plt.plot(np.diff(secs))
#     #             plt.show();
#             interpolation_limits[l] = [(secs.min(), befores.max()), (afters.min(), secs.max())]
#         else:
#             print secs.min(), secs.max()
#             interpolation_limits[l] = [(secs.min(), secs.max())]

#         for lims in interpolation_limits[l]:
#             considered_secs = sorted(set(range(lims[0], lims[1]+1)) & set(secs))
#             n = len(considered_secs)
#             for i in range(n):
#                 sec = considered_secs[i]
#                 z0 = int(z_xy_ratio_downsampled*sec)
#                 interpolated_contours[z0][l] = label_polygons.loc[sec][l]            
#                 if i + 1 < n:
#                     next_sec = considered_secs[i+1]
#                     z1 = int(z_xy_ratio_downsampled*next_sec)
#                     interp_cnts = interpolate_contours(label_polygons.loc[sec][l], 
#                                                        label_polygons.loc[next_sec][l],
#                                                        z1-z0+1)
#                     for zi, z in enumerate(range(z0+1, z1)):
#                         interpolated_contours[z][l] = interp_cnts[zi+1]


def generate_annotaion_list(stack, username, filepath=None):

    if filepath is None:
        filepath = os.path.join('/oasis/projects/nsf/csd395/yuncong/CSHL_data_labelings_losslessAlignCropped/',
                    '%(stack)s_%(username)s_latestAnnotationFilenames.txt' % {'stack': stack, 'username': username})
        
    f = open(filepath, 'w')
    
    fn_list = []
    
    for sec in range(first_bs_sec, last_bs_sec + 1):

        dm.set_slice(sec)

        ret = dm.load_review_result_path(username, 'latest', suffix='consolidated')
        if ret is not None:
            fn = ret[0]
            # print fn
            fn_list.append(fn)
            f.write(fn + '\n')

    f.close()
    return fn_list

def get_section_contains_labels(label_polygons, filtered_labels=None):

    section_contains_labels = defaultdict(set)
    
    if filtered_labels is None:
        labels = label_polygons.keys()
    else:
        labels = label_polygons.keys() & set(filtered_labels)
    
    for l in labels:
        for s in label_polygons[l].dropna().index:
            section_contains_labels[s].add(l)
    section_contains_labels.default_factory = None
    
    return section_contains_labels


def load_label_polygons_if_exists(stack, username, output_path=None, force=False):
    
    label_polygons_path = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_labelings_losslessAlignCropped/%(stack)s_%(username)s_annotation_polygons.h5' % {'stack': stack, 'username': username}
    
    if os.path.exists(label_polygons_path) and not force:
        label_polygons = pd.read_hdf(label_polygons_path, 'label_polygons')
    else:
        label_polygons = pd.DataFrame(generate_label_polygons(stack, username))
        
        if output_path is None:
            label_polygons.to_hdf(label_polygons_path, 'label_polygons')
        else:
            label_polygons.to_hdf(output_path, 'label_polygons')
        
    return label_polygons


def generate_label_polygons(stack, username, output_path = None, labels_merge_map = {'SolM': 'Sol'}):

    dm = DataManager(stack=stack)
    label_polygons = defaultdict(lambda: {})

    section_bs_begin, section_bs_end = section_range_lookup[stack]
    
    for sec in range(section_bs_begin, section_bs_end+1):

        dm.set_slice(sec)

        ret = dm.load_proposal_review_result(username, 'latest', 'consolidated')

        if ret is None:
            continue

#         print sec

        usr, ts, suffix, annotations = ret

        for ann in annotations:
            label = ann['label']
            if label in labels_merge_map:
                label = labels_merge_map[label]
            label_polygons[label][sec] = np.array(ann['vertices']).astype(np.int)

    label_polygons.default_factory = None
    
    return label_polygons