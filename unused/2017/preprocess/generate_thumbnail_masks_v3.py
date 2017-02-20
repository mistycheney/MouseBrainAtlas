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
from skimage.color import rgb2gray, gray2rgb
from skimage.io import imread, imsave
from skimage.segmentation import active_contour
from skimage.filters import gaussian, threshold_adaptive
from skimage.util import img_as_ubyte

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from annotation_utilities import contours_to_mask, points_inside_contour
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
    description='Generate mask for thumbnail images - fluorescent')
parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("input_dir", type=str, help="input dir")
parser.add_argument("filenames", type=str, help="image filenames, json encoded, no extensions")
parser.add_argument("output_dir", type=str, help="output dir")
parser.add_argument("method", type=str, help="nissl or fluorescent- use the entropy method for Nissl or Graphcut method for Fluorescent")
parser.add_argument("--tb_fmt", type=str, help="thumbnail format (tif or png)", default='tif')
parser.add_argument("--train_distance_percentile", type=int, help="distance percentile; used for fluorescent", default=10)
parser.add_argument("--chi2_threshold", type=float, help="chi2 threshold; used for fluorescent", default=.2)
parser.add_argument("--min_size", type=int, help="minimum submask size", default=1000)
parser.add_argument("--vmax_percentile", type=int, help="percentile of graylevel values of vmax", default=100)
args = parser.parse_args()

stack = args.stack_name
input_dir = args.input_dir
output_dir = create_if_not_exists(args.output_dir)
tb_fmt = args.tb_fmt
method = args.method

# Used for fluorescent
TRAIN_DISTANCES_AS_DISTANCE_PERCENTILE = args.train_distance_percentile
CHI2_THRESHOLD = args.chi2_threshold
VMAX_PERCENTILE = args.vmax_percentile
MIN_SIZE = args.min_size
AREA_CHANGE_RATIO_TERMINATE_CRITERIA = 2. # When area changes more than this ratio, stop iteration.

sys.stderr.write('Train distance percentile: %d\n' % TRAIN_DISTANCES_AS_DISTANCE_PERCENTILE)
sys.stderr.write('Chi2 threshold: %.2f\n' % CHI2_THRESHOLD)
sys.stderr.write('Minimum size: %.2f\n' % MIN_SIZE)
sys.stderr.write('Vmax percentile: %.2f\n' % VMAX_PERCENTILE)


# output_viz_dir = create_if_not_exists('/home/yuncong/CSHL_data_processed/%(stack)s/%(stack)s_maskContourViz_unsorted' % dict(stack=stack))

import json

filenames = json.loads(args.filenames)

##############################################

def generate_entropy_mask(img):

    h, w = img.shape
    e = entropy(img, disk(5))

    x = np.atleast_2d(e[e > .1]).T

    bics = []
    clfs = []
    for nc in [2,3]:
        clf = GaussianMixture(n_components=nc, covariance_type='full')
        clf.fit(x)
        bic = clf.bic(x)
        bics.append(bic)
        clfs.append(clf)

    print 'num. components', [2,3][np.argsort(bics)[0]]

    clf = clfs[np.argsort(bics)[0]]
    covars = np.atleast_1d(np.squeeze(clf.covariances_))

    if clf.n_components == 3:
        invalid_component = np.where(covars > .4)[0]
        if len(invalid_component) == 1:
            clf = clfs[0]
            covars = np.atleast_1d(np.squeeze(clf.covariances_))

    means = np.atleast_1d(np.squeeze(clf.means_))
    order = np.argsort(means)
    weights = clf.weights_

    new_means = []
    new_covs = []
    new_weights = []

    # Force into 2 classes: foreground and background
    km = KMeans(2)
    km.fit([[x]for x in means])
    for l in set(km.labels_):
        new_mean = means[km.labels_ == l].mean()
        new_cov = covars[km.labels_ == l].mean()
        new_weight = weights[km.labels_ == l].sum()

        new_means.append(new_mean)
        new_covs.append(new_cov)
        new_weights.append(new_weight)

    order = np.argsort(new_means)
    new_means = np.array(new_means)[order]
    new_covs = np.array(new_covs)[order]
    new_weights = np.array(new_weights)[order]

    counts, bins = np.histogram(e.flat, bins=100, density=True);

    # ignore small components
    gs = np.array([w * 1./np.sqrt(2*np.pi*c) * np.exp(-(bins-m)**2/(2*c)) for m, c, w in zip(new_means, new_covs, new_weights)])

    thresh = bins[np.where(gs[-1] - gs[-2] < 0)[0][-1]]

    mask = e > thresh

    max_area = np.max([p.area for p in regionprops(label(mask))])
    min_size = min(max_area, MIN_SIZE)

    mask_rso = remove_small_objects(mask, min_size=min_size, connectivity=8)
    mask_rsh = remove_small_holes(mask_rso, min_size=20000, connectivity=8)
    return mask_rsh


def generate_mask(fn, tb_fmt='tif'):
    try:
        if method == 'fluorescent':

            img_rgb = imread(os.path.join(input_dir, fn + '.' + tb_fmt))

            # Auto-level

            img_rgb_bf = np.zeros_like(img_rgb, np.uint8)

            for i in range(3):
                img = img_rgb[..., i].copy()

                img_flattened = img.flatten()
                vmax = np.percentile(img_flattened, VMAX_PERCENTILE)

                vmin_perc = 1
                while True:
                    vmin = np.percentile(img_flattened, vmin_perc)
                    if vmin > 0:
                        break
                    else:
                        vmin_perc += 1

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

            ncut_labels = cut_normalized(slic_labels, sim_graph, in_place=False, thresh=superpixel_merge_sim_thresh, num_cuts=20)

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

            # TRAIN_DISTANCES_AS_DISTANCE_PERCENTILE = 10

            hist_distances = {}
            for l, h in histos.iteritems():
                # hist_distances[l] = np.min([chi2(h, th) for th in train_histos])
                hist_distances[l] = np.percentile([chi2(h, th) for th in train_histos], TRAIN_DISTANCES_AS_DISTANCE_PERCENTILE)

            # Generate mask for snake's initial contours.

            entropy_mask = np.zeros_like(img, np.bool)
            for l, d in hist_distances.iteritems():
                if d > CHI2_THRESHOLD:
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

            final_masks = []

            for init_levelset in init_levelsets:

                discard = False
                init_area = np.count_nonzero(init_levelset)

                t = time.time()

                msnake = morphsnakes.MorphACWE(img_enhanced.astype(np.float), smoothing=3, lambda1=1., lambda2=2.)

                msnake.levelset = init_levelset.copy()

                dq = deque([None, None])
                for i in range(200):

                    # at stable stage, the levelset (thus contour) will oscilate,
                    # so instead of comparing to previous levelset, must compare to the one before the previous
                    oneBefore_levelset = dq.popleft()

                    # If less than 3 pixels are changed, stop.
                    if i > 10:
                        if np.count_nonzero(msnake.levelset - oneBefore_levelset) < 3:
                            break

                    # If area changes more than 2, stop.
                    area_change_ratio = np.count_nonzero(msnake.levelset)/float(init_area)
            #         sys.stderr.write('Area change: %.2f.\n' % area_change_ratio)
                    if np.abs(np.log2(area_change_ratio)) > AREA_CHANGE_RATIO_TERMINATE_CRITERIA:
                        discard = True
                        sys.stderr.write('Area change too much, ignore.\n')
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
                    labeled_mask = label(msnake.levelset.astype(np.bool))
                    for l in np.unique(labeled_mask):
                        if l != 0:
                            final_masks.append(labeled_mask == l)

        elif method == 'nissl':

            img_rgb = imread(os.path.join(input_dir, fn + '.' + tb_fmt))
            img = rgb2gray(img_rgb)

            entropy_mask = generate_entropy_mask(img)

            init_contours = [xys for xys in find_contour_points(entropy_mask.astype(np.int), sample_every=1)[1] if len(xys) > 50]
            assert len(init_contours) > 0, 'No contour is detected from entropy mask %s' % fn

            img_adap = threshold_adaptive(img, 51)
            img_adap[~entropy_mask] = 1

            final_masks = []

            img_adap_gauss = gaussian(img_adap.astype(np.float), 1)

            for init_cnt in init_contours:

                snake = active_contour(img_adap_gauss, init_cnt.astype(np.float),
                                       alpha=1., beta=1000., gamma=1.,
                                       w_line=0., w_edge=10.,
                                       max_iterations=1000)

                bg = np.zeros(img.shape[:2], bool)
                xys = points_inside_contour(snake.astype(np.int))
                bg[np.minimum(xys[:,1], bg.shape[0]-1), np.minimum(xys[:,0], bg.shape[1]-1)] = 1

                final_mask = bg & entropy_mask

                if np.count_nonzero(final_mask) < 100:
                    continue

                final_masks.append(final_mask)

        else:
            raise 'Method not recognized.'

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

        # Save binary mask as png
        # mask_fn = os.path.join(output_dir, '%(fn)s_mask.png' % dict(fn=fn))
        # if os.path.exists(mask_fn):
        #     sys.stderr.write('Mask exists, overwrite: %s\n' % mask_fn)
        # imsave(mask_fn, img_as_ubyte(final_mask))

        # submask_viz_dir = create_if_not_exists('/home/yuncong/CSHL_data_processed/%(stack)s/%(stack)s_submask_overlayViz/%(fn)s' % dict(stack=stack, fn=fn))

        # img_rgb = imread(os.path.join(input_dir, fn + '.png'))

        # Save partial mask images
        # for i in range(1, 7):
        #     try:
        #         mask = imread(os.path.join(submask_dir, '%(fn)s_submasks_%(i)d.png' % {'fn': fn, 'i':i})).astype(np.uint8)/255
        #     except:
        #         continue
        #
        #     viz = img_rgb.copy()
        #     for cnt in find_contour_points(mask)[1]:
        #         cv2.polylines(viz, [cnt.astype(np.int)], True, (255,0,0), 3) # red
        #
        #     viz_fn = os.path.join(submask_viz_dir, '%(fn)s_submask_%(i)d_overlayViz.tif' % dict(fn=fn, i=i))
        #     imsave(viz_fn, viz)

        # Save outline overlayed image

        # viz = gray2rgb(img)
        # for final_levelset, init_levelset in zip(final_levelsets, init_levelsets):
        #
        #     for cnt in find_contour_points(final_levelset)[1]:
        #         cv2.polylines(viz, [cnt.astype(np.int)], True, (255,0,0), 1) # red
        #
        #     for cnt in find_contour_points(init_levelset)[1]:
        #         cv2.polylines(viz, [cnt.astype(np.int)], True, (0,0,255), 1) # blue
        #
        # contour_fn = os.path.join(output_viz_dir, '%(fn)s_mask_contour_viz.tif' % dict(fn=fn))
        # if os.path.exists(contour_fn):
        #     execute_command('rm -rf %s' % contour_fn)
        # imsave(contour_fn, viz)
        #
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
        # import socket
        # hostname = socket.gethostname()
        # with open('/home/yuncong/CSHL_data_processed/%(stack)s/%(stack)s_maskError_%(hostname)s.txt' % dict(stack=stack, hostname=hostname), 'w') as f:
        #     f.write(fn + '\n')
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

_ = Parallel(n_jobs=15)(delayed(generate_mask)(fn, tb_fmt=tb_fmt) for fn in filenames)
