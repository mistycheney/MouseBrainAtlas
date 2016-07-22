#! /usr/bin/env python

import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Generate downscaled and grayscale versions of images')

parser.add_argument("stack", type=str, help="stack name")
parser.add_argument("sec", type=int, help="section")
parser.add_argument("--train", dest='train', help="train", action='store_true')
parser.add_argument("--test", dest='test', help="ROI", action='store_true')
args = parser.parse_args()

stack = args.stack
sec = args.sec

#########################################

import sys
import os
import numpy as np

sys.path.append(os.environ['REPO_DIR'] + '/utilities/')
from sift_spm_utilities import *
from learning_utilities import *
from annotation_utilities import *

output_dir = '/oasis/projects/nsf/csd395/yuncong/CSHL_SIFT_SPM_features/'
M = 200
patch_size = 224
stride = 56

image_width, image_height = DataManager.get_image_dimension(stack)
grid_spec = (patch_size, stride, image_width, image_height)

sample_locations = grid_parameters_to_sample_locations(grid_spec=grid_spec)

labelmap = compute_labelmap(stack, sec)
mask_tb = DataManager.load_thumbnail_mask(stack, sec)

indices_roi = np.array([], np.int)
if args.test:
    indices_roi = locate_patches(grid_spec=grid_spec, mask_tb=mask_tb, bbox=detect_bbox_lookup[stack])

indices_allLandmarks = {}
if args.train:

    username = 'yuncong'
    label_polygons = load_label_polygons_if_exists(stack, username, force=False,
                            annotation_rootdir=annotation_midbrainIncluded_rootdir)

    if sec in label_polygons.index:
        label_polygons_filtered = {name: vertices for name, vertices in label_polygons.loc[sec].dropna().to_dict().iteritems()
                                if name in set(labels_unsided) - {'outerContour'}}
        indices_allLandmarks = locate_patches(grid_spec=grid_spec, mask_tb=mask_tb, polygons=label_polygons_filtered)
    else:
        sys.stderr.write('Section %d has no labelings.\n' % sec)

indices_aggregated = np.concatenate([indices_roi] + indices_allLandmarks.values())
sample_locs_aggregated = sample_locations[indices_aggregated]
hists_arr0, hists_arr1, hists_arr2 = compute_spm_histograms(labelmap, sample_locs_aggregated, patch_size=patch_size, M=M)

component_lengths = [len(indices_roi)] + map(len, indices_allLandmarks.values())
roll_sum = np.cumsum(component_lengths)

if args.test:
    test_dir = create_if_not_exists(os.path.join(output_dir, 'test', stack))
    bp.pack_ndarray_file(hists_arr0[:roll_sum[0]], os.path.join(test_dir, '%(stack)s_%(sec)04d_roi1_histograms_l0.bp' % {'stack': stack, 'sec': sec}))
    bp.pack_ndarray_file(hists_arr1[:roll_sum[0]], os.path.join(test_dir, '%(stack)s_%(sec)04d_roi1_histograms_l1.bp' % {'stack': stack, 'sec': sec}))
    bp.pack_ndarray_file(hists_arr2[:roll_sum[0]], os.path.join(test_dir, '%(stack)s_%(sec)04d_roi1_histograms_l2.bp' % {'stack': stack, 'sec': sec}))

if args.train:
    train_dir = create_if_not_exists(os.path.join(output_dir, 'train', stack))
    names = indices_allLandmarks.keys()
    for i in range(len(roll_sum)-1):
        name = names[i]
        bp.pack_ndarray_file(hists_arr0[roll_sum[i]:roll_sum[i+1]], os.path.join(train_dir, '%(stack)s_%(sec)04d_%(name)s_histograms_l0.bp' % {'name': name, 'stack':stack, 'sec': sec}))
        bp.pack_ndarray_file(hists_arr1[roll_sum[i]:roll_sum[i+1]], os.path.join(train_dir, '%(stack)s_%(sec)04d_%(name)s_histograms_l1.bp' % {'name': name, 'stack':stack, 'sec': sec}))
        bp.pack_ndarray_file(hists_arr2[roll_sum[i]:roll_sum[i+1]], os.path.join(train_dir, '%(stack)s_%(sec)04d_%(name)s_histograms_l2.bp' % {'name': name, 'stack':stack, 'sec': sec}))
