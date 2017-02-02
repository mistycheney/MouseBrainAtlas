#! /usr/bin/env python

import os
import time
import sys

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from annotation_utilities import contours_to_mask, points_inside_contour
from registration_utilities import find_contour_points

import numpy as np

from skimage.morphology import remove_small_objects, disk, remove_small_holes, binary_dilation, disk, binary_closing
from skimage.io import imread, imsave
from skimage.util import img_as_ubyte
from skimage.segmentation import slic, mark_boundaries
from skimage.future.graph import rag_mean_color, cut_normalized, cut_threshold
from skimage.feature import canny
from skimage.color import label2rgb

import morphsnakes
from collections import deque

from multiprocess import Pool

#######################################################################################

import argparse
import json

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Generate mask for thumbnail images - fluorescent')
parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("input_dir", type=str, help="input dir")
parser.add_argument("filenames", type=str, help="image filenames, json encoded, no extensions")
parser.add_argument("output_dir", type=str, help="output dir")
parser.add_argument("modality", type=str, help="nissl or fluorescent")
parser.add_argument("--tb_fmt", type=str, help="thumbnail format (tif or png)", default='png')
parser.add_argument("--border_dissim_percentile", type=int, help="distance percentile", default=30)
parser.add_argument("--fg_dissim_thresh", type=float, \
help="Superpixels more dissimilar to border sp. than this threshold is foreground.", default=.4)
parser.add_argument("--min_size", type=int, help="minimum submask size", default=1000)
args = parser.parse_args()

stack = args.stack_name
input_dir = args.input_dir
filenames = json.loads(args.filenames)
output_dir = create_if_not_exists(args.output_dir)
tb_fmt = args.tb_fmt
modality = args.modality

VMAX_PERCENTILE = 99
VMIN_PERCENTILE = 1
# GAMMA = 2. # if gamma > 1, image is darker.

SLIC_SIGMA = 3
SLIC_COMPACTNESS = 10
SLIC_N_SEGMENTS = 1000
SLIC_MAXITER = 100

SUPERPIXEL_SIMILARITY_SIGMA = 200.
SUPERPIXEL_MERGE_SIMILARITY_THRESH = .2
GRAPHCUT_NUM_CUTS = 20

BORDER_DISSIMILARITY_PERCENTILE = args.border_dissim_percentile
FOREGROUND_DISSIMILARITY_THRESHOLD = args.fg_dissim_thresh
MIN_SIZE = args.min_size
sys.stderr.write('BORDER_DISSIMILARITY_PERCENTILE: %d\n' % BORDER_DISSIMILARITY_PERCENTILE)
sys.stderr.write('FOREGROUND_DISSIMILARITY_THRESHOLD: %.2f\n' % FOREGROUND_DISSIMILARITY_THRESHOLD)
sys.stderr.write('MIN_SIZE: %.2f\n' % MIN_SIZE)

# INTENSITY_FOREGROUND_PERCENTILE = 10
# INTENSITY_BACKGROUND_PERCENTILE = 99
#
# CANNY_EDGE_SIGMA = 1
# CANNY_LOW_THRESH = .2
# CANNY_HIGH_THRESH = .5

INIT_CONTOUR_MINLEN = 50

MORPHSNAKE_SMOOTHING = 2
MORPHSNAKE_LAMBDA1 = .1
MORPHSNAKE_LAMBDA2 = 2.
MORPHSNAKE_MAXITER = 200
MORPHSNAKE_MINITER = 10
PIXEL_CHANGE_TERMINATE_CRITERIA = 3
AREA_CHANGE_RATIO_TERMINATE_CRITERIA = 2.

##############################################

def generate_mask(fn, tb_fmt='png'):

    try:
        img_rgb = imread(os.path.join(input_dir, '%(fn)s.%(tb_fmt)s' % dict(fn=fn, tb_fmt=tb_fmt))) # uint16 for fluoro, uint8 for nissl.

        # Don't use blue channel, dim bright at the background; the other channels are in fact more contrasty.

        img = img_rgb[..., 1].copy()      # G is better than R as R sometimes are contaminated on fluorescent slices.

        if modality == 'fluorescent':
            img = img.max() - img # invert, make tissue dark on bright background

        # Stretch contrast

        img_flattened = img.flatten()

        vmax_perc = VMAX_PERCENTILE
        while vmax_perc > 80:
            vmax = np.percentile(img_flattened, vmax_perc)
            if vmax < 255:
                break
            else:
                vmax_perc -= 1

        vmin_perc = VMIN_PERCENTILE
        while vmin_perc < 20:
            vmin = np.percentile(img_flattened, vmin_perc)
            if vmin > 0:
                break
            else:
                vmin_perc += 1

        print '%d(%d percentile), %d(%d percentile)' % (vmin, vmin_perc, vmax, vmax_perc)

        img[(img <= vmax) & (img >= vmin)] = 255./(vmax-vmin)*(img[(img <= vmax) & (img >= vmin)]-vmin)
        img[img > vmax] = 255
        img[img < vmin] = 0
        img = img.astype(np.uint8)

        #############################
        ## Graph cut based method. ##
        #############################

        # Input to slic() must be float
        slic_labels = slic(img.astype(np.float), sigma=SLIC_SIGMA, compactness=SLIC_COMPACTNESS, n_segments=SLIC_N_SEGMENTS,
                   multichannel=False, max_iter=SLIC_MAXITER)

        sim_graph = rag_mean_color(img, slic_labels, mode='similarity', sigma=SUPERPIXEL_SIMILARITY_SIGMA)

        # Normalized cut - merge superpixels.

        ncut_labels = cut_normalized(slic_labels, sim_graph, in_place=False, thresh=SUPERPIXEL_MERGE_SIMILARITY_THRESH,
                                    num_cuts=GRAPHCUT_NUM_CUTS)

        # Find background superpixels.

        background_labels = np.unique(np.concatenate([ncut_labels[:,0],
                                                  ncut_labels[:,-1],
                                                  ncut_labels[0,:],
                                                  ncut_labels[-1,:]]))

        # Collect training superpixels.

        train_histos = []
        for b in background_labels:
            histo = np.histogram(img[ncut_labels == b], bins=np.arange(0,256,5))[0].astype(np.float)
            histo = histo/np.sum(histo)
            train_histos.append(histo)

        histos = {}
        for l in np.unique(ncut_labels):
            histo = np.histogram(img[ncut_labels == l], bins=np.arange(0,256,5))[0].astype(np.float)
            histo = histo/np.sum(histo)
            histos[l] = histo

        hist_distances = {}
        for l, h in histos.iteritems():
            hist_distances[l] = np.percentile([chi2(h, th) for th in train_histos], BORDER_DISSIMILARITY_PERCENTILE)
            # min is too sensitive if there is a blob at the border

        # Generate mask for snake's initial contours.

        superpixel_mask = np.zeros_like(img, np.bool)
        for l, d in hist_distances.iteritems():
            if d > FOREGROUND_DISSIMILARITY_THRESHOLD:
                superpixel_mask[ncut_labels == l] = 1

        # Dilate mask to make room for snake operation.

        dilated_superpixel_mask = binary_dilation(superpixel_mask, disk(10))
        dilated_superpixel_mask = remove_small_holes(dilated_superpixel_mask, min_size=2000)
        dilated_superpixel_mask = remove_small_objects(dilated_superpixel_mask, min_size=MIN_SIZE)

        # # Enhance intensity.
        #
        # cc = np.concatenate([img[ncut_labels == b] for b in background_labels])
        #
        # bg_high = np.percentile(cc, INTENSITY_BACKGROUND_PERCENTILE)
        # bg_low = np.percentile(cc, INTENSITY_FOREGROUND_PERCENTILE)
        #
        # foreground_mask = img < bg_low
        #
        # foreground_mask[~dilated_superpixel_mask] = 0
        #
        # foreground_mask = remove_small_objects(foreground_mask, min_size=2000)
        # foreground_mask = remove_small_holes(foreground_mask, min_size=2000)
        #
        # # Enhance edges.
        #
        # is_edge = canny(img, sigma=CANNY_EDGE_SIGMA, low_threshold=CANNY_LOW_THRESH, high_threshold=CANNY_HIGH_THRESH)
        # is_edge[~foreground_mask] = 0

        # Enhanced image - combine intensity and edge enhancement.

        # img_enhanced = img.copy()

        # Enhance edges.

        # img_enhanced[is_edge] = 0

        img_enhanced = img

        # Find contours from mask.

        init_contours = [xys for xys in find_contour_points(dilated_superpixel_mask.astype(np.int), sample_every=1)[1]
                         if len(xys) > INIT_CONTOUR_MINLEN]

        # assert len(init_contours) > 0, 'No contour is detected from entropy mask %s' % fn
        print 'Extracted %d contours from mask.' % len(init_contours)

        # Create initial levelset

        init_levelsets = []
        for cnt in init_contours:
            init_levelset = np.zeros_like(img, np.float)
            init_levelset[contours_to_mask([cnt], img.shape[:2])] = 1.
            init_levelset[:10, :] = 0
            init_levelset[-10:, :] = 0
            init_levelset[:, :10] = 0
            init_levelset[:, -10:] = 0

            init_levelsets.append(init_levelset)

        #####################
        # Evolve morphsnake #
        #####################

        final_masks = []

        for init_levelset in init_levelsets:

            discard = False
            init_area = np.count_nonzero(init_levelset)

            t = time.time()

            msnake = morphsnakes.MorphACWE(img_enhanced.astype(np.float), smoothing=int(MORPHSNAKE_SMOOTHING),
                                           lambda1=MORPHSNAKE_LAMBDA1, lambda2=MORPHSNAKE_LAMBDA2)

            msnake.levelset = init_levelset.copy()

            dq = deque([None, None])
            for i in range(MORPHSNAKE_MAXITER):

                # At stable stage, the levelset (thus contour) will oscilate,
                # so instead of comparing to previous levelset, must compare to the one before the previous
                oneBefore_levelset = dq.popleft()

                # If less than 3 pixels are changed, stop.
                if i > MORPHSNAKE_MINITER:
                    if np.count_nonzero(msnake.levelset - oneBefore_levelset) < PIXEL_CHANGE_TERMINATE_CRITERIA:
                        break

                area = np.count_nonzero(msnake.levelset)

                if area < MIN_SIZE:
                    discard = True
                    sys.stderr.write('Too small, stop iteration.\n')
                    break

                # If area changes more than 2, stop.
                area_change_ratio = area/float(init_area)
        #         sys.stderr.write('Area change: %.2f.\n' % area_change_ratio)
                if np.abs(np.log2(area_change_ratio)) > AREA_CHANGE_RATIO_TERMINATE_CRITERIA:
                    discard = True
                    sys.stderr.write('Area change too much, stop iteration.\n')
                    break

                dq.append(msnake.levelset)

        #         t = time.time()
                msnake.step()
        #         sys.stderr.write('Step: %f seconds\n' % (time.time()-t)) # 0.6 second / step

            sys.stderr.write('Snake finished at iteration %d.\n' % i)
            sys.stderr.write('Snake: %.2f seconds\n' % (time.time()-t)) # 72s

            if discard:
                sys.stderr.write('Discarded.\n')
                continue
            else:
                # Handles the case that a single initial contour morphs into multiple contours
                labeled_mask = label(msnake.levelset.astype(np.bool))
                for l in np.unique(labeled_mask):
                    if l != 0:
                        m = labeled_mask == l
                        if np.count_nonzero(m) > MIN_SIZE:
                            final_masks.append(m)
                            sys.stderr.write('Final masks added.\n')

            # Sort submasks by size.
            final_masks = sorted(final_masks, key=lambda x: len(x), reverse=True)

            submask_dir = create_if_not_exists(os.path.join(output_dir, fn))
            execute_command('rm -f %s/*' % submask_dir)
            # submask_viz_dir = create_if_not_exists('/home/yuncong/CSHL_data_processed/%(stack)s/%(stack)s_submask_overlayViz/%(fn)s' % dict(stack=stack, fn=fn))
            # execute_command('rm %s/*' % submask_viz_dir)

            img_rgb = imread(os.path.join(input_dir, fn + '.png'))

            for i, mask in enumerate(final_masks):
                imsave(os.path.join(submask_dir,'%(fn)s_submask_%(i)d.png' % {'fn': fn, 'i':i+1}), img_as_ubyte(mask))

                viz = img_rgb.copy()
                for cnt in find_contour_points(mask)[1]:
                    cv2.polylines(viz, [cnt.astype(np.int)], True, (255,0,0), 3) # red
                viz_fn = os.path.join(submask_dir, '%(fn)s_submask_%(i)d_overlayViz.tif' % dict(fn=fn, i=i+1))
                sys.stderr.write(viz_fn + '\n')
                imsave(viz_fn, viz)

        ################################################
        # Automatically judge the goodness of each mask
        ################################################

        n = len(final_masks)
        decisions = [0 for _ in range(n)]
        undecided = []
        negatives = []
        positives = []

        if n == 1: # If only one mask, assign it positive.
            decisions[0] = 1
            undecided.append(0)
        else:
            for i in range(n):
                if np.count_nonzero(final_masks[i]) < 10000:
                    decisions[i] = -1
                    negatives.append(i)
                else:
                    decisions[i] = 0
                    positives.append(i)

        # If only one undecided and the rest are negative, make the undecided positive.
        if len(undecided) == 1 and len(negatives) == n - len(undecided):
            decisions[undecided[0]] = 1

        # If there are more than one positives, make all other undecided negative.
        if len(positives) >= 1:
            for u in undecided:
                decisions[u] = -1

        decisions = consolidate_alg_submask_review(dict(zip(range(1, 1+len(decisions)), decisions)))

        # Write to file. Slot index starts with 1.
        with open(os.path.join(submask_dir, '%(fn)s_submasksAlgReview.txt' % dict(fn=fn)), 'w') as f:
            for i, dec in decisions.iteritems():
                f.write('%d %d\n' % (i, dec))

    except Exception as e:
        sys.stderr.write('%s\n' % e)
        sys.stderr.write('Mask error: %s\n' % fn)
        return


def consolidate_alg_submask_review(alg_decisions):
    """
    Return {submask_ind: bool}
    """

    consolidated_decisions = {}

    if len(alg_decisions) == 0:
        return {}
    else:
        alg_positives = [index for index, l in alg_decisions.iteritems() if l == 1]
        if len(alg_positives) > 0:
            assert len(alg_positives) == 1
            correct_index = alg_positives[0]
        else:
            correct_index = 1

        consolidated_decisions[correct_index] = 1
        for idx in alg_decisions:
            if idx != correct_index:
                consolidated_decisions[idx] = -1
        return {i: r == 1 for i, r in consolidated_decisions.iteritems()}

pool = Pool(15)
pool.map(lambda fn: generate_mask(fn, tb_fmt=tb_fmt), filenames)
pool.close()
pool.join()
