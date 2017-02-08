#! /usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import time
import sys

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from annotation_utilities import contours_to_mask
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
from preprocess2_utilities import *

from scipy.sparse.linalg import ArpackError

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
# parser.add_argument("modality", type=str, help="nissl or fluorescent")
parser.add_argument("--tb_fmt", type=str, help="thumbnail format (tif or png)", default='png')
parser.add_argument("--border_dissim_percentile", type=int, help="distance percentile", default=DEFAULT_BORDER_DISSIMILARITY_PERCENTILE)
parser.add_argument("--fg_dissim_thresh", type=float, help="Superpixels more dissimilar to border sp. than this threshold is foreground.", default=None)
parser.add_argument("--min_size", type=int, help="minimum submask size", default=DEFAULT_MINSIZE)
args = parser.parse_args()

stack = args.stack_name
input_dir = args.input_dir
filenames = json.loads(args.filenames)
output_dir = create_if_not_exists(args.output_dir)
tb_fmt = args.tb_fmt
# modality = args.modality

VMAX_PERCENTILE = 99
VMIN_PERCENTILE = 1
# GAMMA = 2. # if gamma > 1, image is darker.

SLIC_SIGMA = 3
SLIC_COMPACTNESS = 10
SLIC_N_SEGMENTS = 200
SLIC_MAXITER = 100

SUPERPIXEL_SIMILARITY_SIGMA = 50.
SUPERPIXEL_MERGE_SIMILARITY_THRESH = .2
GRAPHCUT_NUM_CUTS = 20

BORDER_DISSIMILARITY_PERCENTILE = args.border_dissim_percentile
sys.stderr.write('BORDER_DISSIMILARITY_PERCENTILE: %d\n' % BORDER_DISSIMILARITY_PERCENTILE)
MIN_SIZE = args.min_size
sys.stderr.write('MIN_SIZE: %.2f\n' % MIN_SIZE)
FOREGROUND_DISSIMILARITY_THRESHOLD = args.fg_dissim_thresh

FOREGROUND_DISSIMILARITY_THRESHOLD_MAX = 1.5
INIT_CONTOUR_COVERAGE_MAX = .5

INIT_CONTOUR_MINLEN = 50

MORPHSNAKE_SMOOTHING = 2
MORPHSNAKE_LAMBDA1 = .1
MORPHSNAKE_LAMBDA2 = 2.
MORPHSNAKE_MAXITER = 600
MORPHSNAKE_MINITER = 10
PIXEL_CHANGE_TERMINATE_CRITERIA = 3
AREA_CHANGE_RATIO_MAX = 1.1
AREA_CHANGE_RATIO_MIN = .1

##############################################

def generate_mask(fn, tb_fmt='png'):

    try:
        submask_dir = create_if_not_exists(os.path.join(output_dir, fn))
        execute_command('rm -f %s/*' % submask_dir)

        img_rgb = imread(os.path.join(input_dir, '%(fn)s.%(tb_fmt)s' % dict(fn=fn, tb_fmt=tb_fmt)))

        # Don't use blue channel, dim bright at the background; the other channels are in fact more contrasty.
        rmax = img_rgb[..., 0].max()
        rmin = img_rgb[..., 0].min()
        rrange = rmax - rmin
        gmax = img_rgb[..., 1].max()
        gmin = img_rgb[..., 1].min()
        grange = gmax - gmin
        bmax = img_rgb[..., 2].max()
        bmin = img_rgb[..., 2].min()
        brange = bmax - bmin

        rborder = np.median(np.r_[img_rgb[:10, :, 0].flatten(),
                                 img_rgb[-10:, :, 0].flatten(),
                                 img_rgb[:, :10, 0].flatten(),
                                 img_rgb[:, -10:, 0].flatten()])
        gborder = np.median(np.r_[img_rgb[:10, :, 1].flatten(),
                                 img_rgb[-10:, :, 1].flatten(),
                                 img_rgb[:, :10, 1].flatten(),
                                 img_rgb[:, -10:, 1].flatten()])
        bborder = np.median(np.r_[img_rgb[:10, :, 2].flatten(),
                                 img_rgb[-10:, :, 2].flatten(),
                                 img_rgb[:, :10, 2].flatten(),
                                 img_rgb[:, -10:, 2].flatten()])

        r_std = np.std(img_rgb[..., 0])
        g_std = np.std(img_rgb[..., 1])
        b_std = np.std(img_rgb[..., 2])

        if r_std > g_std:
            img = img_rgb[..., 0].copy() # G channel is better than R as R sometimes are contaminated on fluorescent slices.
            print 'Using RED channel.'
        else:
            img = img_rgb[..., 1].copy() # G channel is better than R as R sometimes are contaminated on fluorescent slices.
            print 'Using GREEN channel.'

        if rborder < 123 and gborder < 123 and bborder < 123:
            # dark background, fluorescent
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

        sys.stderr.write('%d(%d percentile), %d(%d percentile)\n' % (vmin, vmin_perc, vmax, vmax_perc) )

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

        for _ in range(3):
            try:
                t = time.time()
                ncut_labels = cut_normalized(slic_labels, sim_graph, in_place=False, thresh=SUPERPIXEL_MERGE_SIMILARITY_THRESH,
                                            num_cuts=GRAPHCUT_NUM_CUTS)
                sys.stderr.write('Normalized Cut: %.2f seconds.\n' % (time.time() - t))
                break
            except ArpackError as e:
                sys.stderr.write('ArpackError encountered.\n')
                continue

        ncut_boundaries_viz = mark_boundaries(img, label_img=ncut_labels, background_label=-1)
        # imsave(os.path.join(submask_dir, '%(fn)s_.png' % dict(fn=fn)), ncut_boundaries_viz)

        # Find background superpixels.

        background_labels = np.unique(np.concatenate([ncut_labels[:,0], ncut_labels[:,-1], ncut_labels[0,:], ncut_labels[-1,:]]))

        # Collect training superpixels.

        border_histos = []
        for b in background_labels:
            histo = np.histogram(img[ncut_labels == b], bins=np.arange(0,256,5))[0].astype(np.float)
            histo = histo/np.sum(histo)
            border_histos.append(histo)

        histos = {}
        for l in np.unique(ncut_labels):
            histo = np.histogram(img[ncut_labels == l], bins=np.arange(0,256,5))[0].astype(np.float)
            histo = histo/np.sum(histo)
            histos[l] = histo

        hist_distances = {}
        for l, h in histos.iteritems():
            hist_distances[l] = np.percentile([chi2(h, th) for th in border_histos], BORDER_DISSIMILARITY_PERCENTILE)
            # min is too sensitive if there is a blob at the border

        global FOREGROUND_DISSIMILARITY_THRESHOLD
        # sys.stderr.write('FOREGROUND_DISSIMILARITY_THRESHOLD: %.2f\n' % BORDER_DISSIMILARITY_PERCENTILE2)
        if FOREGROUND_DISSIMILARITY_THRESHOLD is None:
            # Automatically determine this value

            dist_vals = np.asarray(hist_distances.values())
            ticks = np.linspace(0, dist_vals.max(), 100)
            percentages = [np.count_nonzero(dist_vals < th) / float(len(dist_vals)) for th in ticks]

            h = np.gradient(np.gradient(percentages, 3), 3)

            plt.figure();
            plt.plot(ticks, h);
            plt.title('Hessian - minima is the plateau point of cum. distr.');
            plt.xlabel('Dissimlarity threshold');
            plt.savefig(os.path.join(submask_dir, '%(fn)s_spDissimCumDistHessian.png' % dict(fn=fn)));
            # plt.close();

            ticks_sorted = ticks[h.argsort()]
            ticks_sorted_reduced = ticks_sorted[ticks_sorted < FOREGROUND_DISSIMILARITY_THRESHOLD_MAX]

            init_contour_percentages = np.asarray([np.sum([np.count_nonzero(ncut_labels == l)
                                                           for l, d in hist_distances.iteritems()
                                                           if d > th]) / float(img.size)
                                                   for th in ticks_sorted_reduced])

            threshold_candidates = ticks_sorted_reduced[(init_contour_percentages < INIT_CONTOUR_COVERAGE_MAX) &\
                                                              (init_contour_percentages > 0)]
            np.savetxt(os.path.join(submask_dir, '%(fn)s_spThreshCandidates.txt' % dict(fn=fn)), threshold_candidates, fmt='%.3f')
            print threshold_candidates[:10]
            FOREGROUND_DISSIMILARITY_THRESHOLD = threshold_candidates[0]

        sys.stderr.write('FOREGROUND_DISSIMILARITY_THRESHOLD: %.2f\n' % FOREGROUND_DISSIMILARITY_THRESHOLD)

        # Visualize superpixel border distance map
        superpixel_border_distances = np.zeros_like(img, np.float)
        for l, s in hist_distances.iteritems():
            superpixel_border_distances[ncut_labels == l] = s

        plt.figure();
        im = plt.imshow(superpixel_border_distances, vmin=0, vmax=2);
        plt.title('Superpixels distance to border');
        plt.colorbar(im, fraction=0.025, pad=0.02);
        plt.savefig(os.path.join(submask_dir, '%(fn)s_spBorderDissim.png' % dict(fn=fn)));
        # plt.close();

        # Generate mask for snake's initial contours.

        superpixel_mask = np.zeros_like(img, np.bool)
        for l, d in hist_distances.iteritems():
            if d > FOREGROUND_DISSIMILARITY_THRESHOLD:
                superpixel_mask[ncut_labels == l] = 1

        labelmap, n_submasks = label(superpixel_mask, return_num=True)

        superpixel_submasks = []
        dilated_superpixel_submasks = []

        for i in range(1, n_submasks+1):
            m = labelmap == i
            superpixel_submasks.append(m)

            dilated_m = binary_dilation(m, disk(10))
            dilated_m = remove_small_objects(remove_small_holes(dilated_m, min_size=2000), min_size=MIN_SIZE)
            dilated_superpixel_submasks.append(dilated_m)

        # Visualize

        viz = img_as_ubyte(ncut_boundaries_viz)
        for submask in superpixel_submasks:
            for cnt in find_contour_points(submask)[1]:
                cv2.polylines(viz, [cnt.astype(np.int)], True, (255,0,0), 1) # red
        for submask in dilated_superpixel_submasks:
            for cnt in find_contour_points(submask)[1]:
                cv2.polylines(viz, [cnt.astype(np.int)], True, (0,0,255), 1) # blue
        imsave(os.path.join(submask_dir, '%(fn)s_ncutSubmasks.png' % dict(fn=fn)), viz)

        #####################

        img_enhanced = img

        # Find contours from mask.

        init_contours = [xys for submask in dilated_superpixel_submasks
                 for xys in find_contour_points(submask.astype(np.int), sample_every=1)[1]
                 if len(xys) > INIT_CONTOUR_MINLEN]

        # assert len(init_contours) > 0, 'No contour is detected from entropy mask %s' % fn
        sys.stderr.write('Extracted %d contours from mask.\n' % len(init_contours))

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

                if area_change_ratio > AREA_CHANGE_RATIO_MAX:
                    discard = True
                    sys.stderr.write('Area increases too much, stop iteration.\n')
                    break
                elif area_change_ratio < AREA_CHANGE_RATIO_MIN:
                    discard = True
                    sys.stderr.write('Area shrinks too much, stop iteration.\n')
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

        ############
        ## Export ##
        ############

        # submask_viz_dir = create_if_not_exists('/home/yuncong/CSHL_data_processed/%(stack)s/%(stack)s_submask_overlayViz/%(fn)s' % dict(stack=stack, fn=fn))
        # execute_command('rm %s/*' % submask_viz_dir)

        img_rgb = imread(os.path.join(input_dir, fn + '.' + tb_fmt))

        all_init_cnts = []
        for i, init_levelset in enumerate(init_levelsets):
            all_init_cnts += find_contour_points(init_levelset)[1]

        for i, mask in enumerate(final_masks):
            imsave(os.path.join(submask_dir,'%(fn)s_submask_%(i)d.png' % {'fn': fn, 'i':i+1}), img_as_ubyte(mask))

            viz = img_rgb.copy()

            cnts = find_contour_points(mask)
            if len(cnts) == 0:
                raise
            for cnt in cnts[1]:
                cv2.polylines(viz, [cnt.astype(np.int)], True, (255,0,0), 3) # red

            for cnt in all_init_cnts:
                cv2.polylines(viz, [cnt.astype(np.int)], True, (0,255,0), 3) # green

            viz_fn = os.path.join(submask_dir, '%(fn)s_submask_%(i)d_overlayViz.png' % dict(fn=fn, i=i+1))
            sys.stderr.write(viz_fn + '\n')
            imsave(viz_fn, viz)

        ################################################
        # Automatically judge the goodness of each mask
        ################################################

        n = len(final_masks)

        rank1 = np.argsort([np.count_nonzero(m) for m in final_masks])[::-1]

        bbox_to_image_center_distance = []
        for m in final_masks:
            xmin, xmax, ymin, ymax = bbox_2d(m)
            dist = np.sqrt(np.sum((np.r_[img.shape[1]/2, img.shape[0]/2] - ((xmin + xmax)/2, (ymin+ymax)/2))**2))
            bbox_to_image_center_distance.append(dist)

        rank2 = np.argsort(bbox_to_image_center_distance)

        r1 = np.asarray([r for r, i in sorted(enumerate(rank1), key=lambda (r,i): i)])
        r2 = np.asarray([r for r, i in sorted(enumerate(rank2), key=lambda (r,i): i)])
        rank = np.argsort(r1 + 1.01 * r2) # weight being close to center a bit more to break tie
        best_mask_ind = rank[0]

        decisions = [0 for _ in range(n)]
        decisions[best_mask_ind] = 1

        # Write to file. Slot index starts with 1.
        with open(os.path.join(submask_dir, '%(fn)s_submasksAlgReview.txt' % dict(fn=fn)), 'w') as f:
            for i, dec in enumerate(decisions):
                f.write('%d %d\n' % (i+1, dec))

    except Exception as e:
        sys.stderr.write('%s\n' % e)
        sys.stderr.write('Mask error: %s\n' % fn)
        return

t = time.time()
pool = Pool(12)
pool.map(lambda fn: generate_mask(fn, tb_fmt=tb_fmt), filenames)
pool.close()
pool.join()
sys.stderr.write('Generate masks: %.2f\n' % (time.time()-t))
