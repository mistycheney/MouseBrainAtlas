#! /usr/bin/env python

import sys
import os
import time

import numpy as np
from joblib import Parallel, delayed

sys.path.append('/oasis/projects/nsf/csd181/yuncong/opencv-2.4.9/release/lib/python2.7/site-packages')
import cv2

from skimage.filters.rank import entropy
from skimage.morphology import remove_small_objects, disk, remove_small_holes, binary_dilation, disk, binary_closing
from skimage.measure import label, regionprops
from skimage.color import rgb2gray
from skimage.io import imread, imsave
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from skimage.filter import threshold_adaptive
from skimage.util import img_as_ubyte

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from annotation_utilities import contours_to_mask
from registration_utilities import find_contour_points

import morphsnakes
from collections import deque

from skimage.segmentation import slic
from skimage.future.graph import rag_mean_color, cut_normalized
from skimage.feature import canny

#######################################################################################

import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Generate mask for thumbnail images')
parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("input_dir", type=str, help="input dir")
parser.add_argument("filenames", type=str, help="image filenames, json encoded, no extensions")
parser.add_argument("output_dir", type=str, help="output dir")
parser.add_argument("--tb_fmt", type=str, help="thumbnail format (tif or png)", default='tif')
args = parser.parse_args()

stack = args.stack_name
input_dir = args.input_dir
output_dir = create_if_not_exists(args.output_dir)
tb_fmt = args.tb_fmt

output_viz_dir = create_if_not_exists('/home/yuncong/CSHL_data_processed/%(stack)s/%(stack)s_maskContourViz_unsorted' % dict(stack=stack))

import json

filenames = json.loads(args.filenames)

def generate_mask(fn, tb_fmt='tif'):
    try:
        img_rgb = imread(os.path.join(input_dir, fn + '.' + tb_fmt))

        # Auto-level

        img_rgb_bf = np.zeros_like(img_rgb, np.uint8)

        for i in range(3):
            img = img_rgb[..., i].copy()

            img_flattened = img.flatten()
            vmax = np.percentile(img_flattened, 99)
            vmin = np.percentile(img_flattened, 1)
            img[(img <= vmax) & (img >= vmin)] = (255./(vmax-vmin)*(img[(img <= vmax) & (img >= vmin)]-vmin)).astype(np.uint8)
            img[img > vmax] = 255
            img[img < vmin] = 0

            img_rgb_bf[..., i] = img

        img_bf = np.max(img_rgb_bf, axis=2)
        img = 255 - img_bf

        # Option 1: Graph cut based method.

        slic_labels = slic(img.astype(np.float), sigma=3, compactness=10, n_segments=1000,
                           multichannel=False, max_iter=100);

        sim_graph = rag_mean_color(img, slic_labels, mode='similarity', sigma=200.)

        # Normalized cut - merge superpixels.

        superpixel_merge_sim_thresh = .2

        ncut_labels = cut_normalized(slic_labels, sim_graph, in_place=False, thresh=superpixel_merge_sim_thresh)

        # Find background superpixels.

        background_labels = np.unique(np.concatenate([ncut_labels[:,0],
                                              ncut_labels[:,-1],
                                              ncut_labels[0,:],
                                              ncut_labels[-1,:]]))

        # background_labels = np.unique(np.concatenate([ncut_labels[:20,0], ncut_labels[-20:,0],
        #                                               ncut_labels[:20,-1], ncut_labels[-20:,-1],
        #                                               ncut_labels[0,:20], ncut_labels[0,-20:],
        #                                               ncut_labels[-1,:20], ncut_labels[-1,-20:]]))

        # Collect training superpixels.

        train_histos = []
        for b in background_labels:
            histo = np.histogram(img[ncut_labels == b], bins=np.arange(0,256,5))[0].astype(np.float)
            histo = histo/np.sum(histo)
            train_histos.append(histo)

        # Compute superpixel distances to training superpixels.

        histos = {}
        for l in np.unique(ncut_labels):
            histo = np.histogram(img[ncut_labels == l], bins=np.arange(0,256,5))[0].astype(np.float)
            histo = histo/np.sum(histo)
            histos[l] = histo

        TRAIN_DISTANCES_AS_DISTANCE_PERCENTILE = 10

        hist_distances = {}
        for l, h in histos.iteritems():
            # hist_distances[l] = np.min([chi2(h, th) for th in train_histos])
            hist_distances[l] = np.percentile([chi2(h, th) for th in train_histos], TRAIN_DISTANCES_AS_DISTANCE_PERCENTILE)

        # Generate mask for snake's initial contours.

        chi2_threshold = .5
        MIN_SIZE = 4000

        entropy_mask = np.zeros_like(img, np.bool)
        for l, d in hist_distances.iteritems():
            if d > chi2_threshold:
                entropy_mask[ncut_labels == l] = 1

        dilated_entropy_mask = binary_dilation(entropy_mask, disk(10))
        dilated_entropy_mask = remove_small_holes(dilated_entropy_mask, min_size=2000)
        dilated_entropy_mask = remove_small_objects(dilated_entropy_mask, min_size=MIN_SIZE)

        # Enhance intensity.

        cc = np.concatenate([img[ncut_labels == b] for b in background_labels])
        bg_high = np.percentile(cc, 99)
        bg_low = np.percentile(cc, 10)

        foreground_mask = img < bg_low
        foreground_mask[~dilated_entropy_mask] = 0

        foreground_mask = remove_small_objects(foreground_mask, min_size=2000)
        foreground_mask = remove_small_holes(foreground_mask, min_size=2000)

        # Enhance edges.

        is_edge = canny(img, sigma=1, low_threshold=.2, high_threshold=.5)
        is_edge[~foreground_mask] = 0

        # Enhance image - combine intensity and edge enhancement.

        img_enhanced = img.copy()
        img_enhanced[is_edge] = 0


        # Find contours from mask.

        init_contours = [xys for xys in find_contour_points(dilated_entropy_mask.astype(np.int), sample_every=1)[1]
                         if len(xys) > 50]

        # assert len(init_contours) > 0, 'No contour is detected from entropy mask %s' % fn
        print 'Extracted %d contours from mask.' % len(init_contours)

        # Option 1: Morphsnake approach

        # Create initial levelset

        init_levelsets = []
        for cnt in init_contours:
            init_levelset = np.zeros_like(img, np.float)
            init_levelset[contours_to_mask([cnt], img.shape[:2])] = 1.
            init_levelset[:10, :] = 0
            init_levelset[-10:, :] = 0
            init_levelset[:, :10] = 0
            init_levelset[:, -10:] = 0

            # val_entropy = entropy(img_rgb[init_levelset.astype(np.bool), 2])
            # sys.stderr.write('Entropy: %.2f.\n' % val_entropy)
            #
            # val_std = np.std(img_rgb[init_levelset.astype(np.bool), 2])
            # sys.stderr.write('Contrast: %.2f.\n' % val_std)
            # #
            # if val_std < 60:
            #     sys.stderr.write('Contrast too low, ignore.\n')
            #     continue

            init_levelsets.append(init_levelset)

        # Evolve morphsnake

        final_levelsets = []

        for init_levelset in init_levelsets:

            t = time.time()

            msnake = morphsnakes.MorphACWE(img_enhanced.astype(np.float), smoothing=3, lambda1=1., lambda2=2.)

            msnake.levelset = init_levelset.copy()

            dq = deque([None, None])
            for i in range(200):

                # at stable stage, the levelset (thus contour) will oscilate,
                # so instead of comparing to previous levelset, must compare to the one before the previous
                oneBefore_levelset = dq.popleft()

                if i > 10:
                    if np.count_nonzero(msnake.levelset - oneBefore_levelset) < 3:
                        break

                dq.append(msnake.levelset)

                msnake.step()

            sys.stderr.write('Snake finished at iteration %d.\n' % i)
            sys.stderr.write('Snake: %.2f seconds\n' % (time.time()-t)) # 72s

            mask = msnake.levelset.astype(np.bool)
            final_levelsets.append(mask)

        final_mask = np.any(final_levelsets, axis=0)


        submask_dir = create_if_not_exists('/home/yuncong/CSHL_data_processed/MD635/MD635_submasks/%(fn)s' % {'fn':fn})
        i = 0
        for final_levelset in final_levelsets:
            l = label(final_levelset)
            for ll in np.unique(l):
                if ll != 0:
                    mask = l == ll
                    # vals = img_rgb[mask, 2]
                    # val_entropy = entropy(vals)
                    # sys.stderr.write('Entropy: %.2f.\n' % val_entropy)
                    # val_std = np.std(vals)
                    # sys.stderr.write('Contrast: %.2f.\n' % val_std)
                    #
                    i += 1
                    imsave(submask_dir + '/%(fn)s_submasks_%(i)d.png' % {'fn': fn, 'i':i}, img_as_ubyte(mask))

        # Save binary mask as png
        mask_fn = os.path.join(output_dir, '%(fn)s_mask.png' % dict(fn=fn))
        if os.path.exists(mask_fn):
            sys.stderr.write('Mask exists, overwrite: %s\n' % mask_fn)
        imsave(mask_fn, img_as_ubyte(final_mask))

        # Save outline overlayed image

        viz = gray2rgb(img)
        for final_levelset, init_levelset in zip(final_levelsets, init_levelsets):

            for cnt in find_contour_points(final_levelset)[1]:
                cv2.polylines(viz, [cnt.astype(np.int)], True, (255,0,0), 1) # red

            for cnt in find_contour_points(init_levelset)[1]:
                cv2.polylines(viz, [cnt.astype(np.int)], True, (0,0,255), 1) # blue

        contour_fn = os.path.join(output_viz_dir, '%(fn)s_mask_contour_viz.tif' % dict(fn=fn))
        if os.path.exists(contour_fn):
            execute_command('rm -rf %s' % contour_fn)
        imsave(contour_fn, viz)

            # viz = gray2rgb(img)
            # final_contours = find_contour_points(final_mask, sample_every=1)[1]
            # for cnt in final_contours:
            #     cv2.polylines(viz, [cnt.astype(np.int)], True, (255,0,0), 1)
            # contour_fn = os.path.join(output_dir, '%(fn)s_mask_contour_viz.tif' % dict(fn=fn))
            # if os.path.exists(contour_fn):
            #     execute_command('rm -rf %s' % contour_fn)
            # imsave(contour_fn, viz)

    except Exception as e:
        sys.stderr.write('%s\n' % e)
        sys.stderr.write('Mask error: %s\n' % fn)
        return

_ = Parallel(n_jobs=15)(delayed(generate_mask)(fn, tb_fmt=tb_fmt) for fn in filenames)
