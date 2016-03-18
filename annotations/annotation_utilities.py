import numpy as np

import sys
import os

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *

import pandas as pd

from collections import defaultdict

def load_label_polygons_if_exists(stack, username):
    label_polygons_path = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_labelings_losslessAlignCropped/%(stack)s_%(username)s_annotation_polygons.h5' % {'stack': stack, 'username': username}
    if os.path.exists(label_polygons_path):
        label_polygons = pd.read_hdf(label_polygons_path, 'label_polygons')
    else:
        label_polygons = generate_label_polygons(stack, username)
        
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

        print sec

        usr, ts, suffix, annotations = ret

        for ann in annotations:
            label = ann['label']
            if label in labels_merge_map:
                label = labels_merge_map[label]
            label_polygons[label][sec] = np.array(ann['vertices']).astype(np.int)

    label_polygons.default_factory = None

    label_polygons = pd.DataFrame(label_polygons)
    
    if output_path is None:
        output_path = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_labelings_losslessAlignCropped/%(stack)s_label_polygons.h5' % {'stack': stack}
    
    label_polygons.to_hdf(output_path, 'label_polygons')
    
    return label_polygons