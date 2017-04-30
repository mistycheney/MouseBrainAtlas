import os
import sys
import time
from collections import deque

import numpy as np
sys.path.append(os.path.join(os.environ['REPO_DIR'], 'preprocess'))
import morphsnakes

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from metadata import *
from registration_utilities import find_contour_points
from annotation_utilities import contours_to_mask

# DEFAULT_BORDER_DISSIMILARITY_PERCENTILE = 30
# DEFAULT_FOREGROUND_DISSIMILARITY_THRESHOLD = .2
# DEFAULT_FOREGROUND_DISSIMILARITY_THRESHOLD = None
# DEFAULT_MINSIZE = 1000 # If tissues are separate pieces, 1000 is not small enough to capture them.
# DEFAULT_MINSIZE = 100

DEFAULT_MASK_CHANNEL = 0
VMAX_PERCENTILE = 99
VMIN_PERCENTILE = 1
MIN_SUBMASK_SIZE = 2000
# INIT_CONTOUR_MINLEN = 50
MORPHSNAKE_SMOOTHING = 1
MORPHSNAKE_LAMBDA1 = 1 # imprtance of inside pixels
MORPHSNAKE_LAMBDA2 = 20 # imprtance of outside pixels
# Only relative lambda1/lambda2 matters, large = shrink, small = expand
MORPHSNAKE_MAXITER = 600
MORPHSNAKE_MINITER = 10
PIXEL_CHANGE_TERMINATE_CRITERIA = 3
# AREA_CHANGE_RATIO_MAX = 1.2
AREA_CHANGE_RATIO_MAX = 10.0
AREA_CHANGE_RATIO_MIN = .1

def brightfieldize_image(img):
    border = np.median(np.concatenate([img[:10, :].flatten(), img[-10:, :].flatten(), img[:, :10].flatten(), img[:, -10:].flatten()]))
    if border < 123:
        # dark background, fluorescent
        img = img.max() - img # invert, make tissue dark on bright background
    return img

def contrast_stretch_image(img):
    """
    Args:
        img (2D np.ndarray): single-channel image.
    """

    # Stretch contrast
    # img_flattened = img.flatten()
    img_flattened = img[(img > 0) & (img < 255)] # ignore 0 and 255 which are likely artificial background

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

    from skimage.exposure import rescale_intensity
    img = img_as_ubyte(rescale_intensity(img, in_range=(vmin, vmax)))

    # img[(img <= vmax) & (img >= vmin)] = 255./(vmax-vmin)*(img[(img <= vmax) & (img >= vmin)]-vmin)
    # img[img > vmax] = 255
    # img[img < vmin] = 0
    # img = img.astype(np.uint8)

    return img

def snake(img, init_submasks=None, init_contours=None, lambda1=MORPHSNAKE_LAMBDA1, return_masks=True, min_size=MIN_SUBMASK_SIZE, crop_margin=50):
    """
    Args:
        crop_margin (int): crop the image to fit the extent of all masks plus a margin of `margin`. Set to negative for no cropping.
    """

    if init_contours is None:
        assert init_submasks is not None, "If no init_contours is given, must provide init_submasks."
        init_submasks = [m for m in init_submasks if np.count_nonzero(init_submasks) <= min_size]
        assert len(init_submasks) > 0, "No initial submasks are above the minimal area of %d." % min_size
        assert all([(m.shape[0] == img.shape[0]) & (m.shape[1] == img.shape[1]) for m in init_submasks]), "Not all shapes of initial submasks match the input image."
        bbox_all_submasks = [bbox_2d(m) for m in init_submasks]
        xmin, ymin = np.min(bbox_all_submasks[:,[0,2]])
        xmax, ymax = np.max(bbox_all_submasks[:,[1,3]])
    else:
        init_contours = [c.astype(np.int) for c in init_contours]
        assert len(init_contours) > 0, "No initial contours are longer than the minimal contour length of %d pixels." % INIT_CONTOUR_MINLEN
        xmin, ymin = np.min([np.min(c, axis=0) for c in init_contours], axis=0)
        xmax, ymax = np.max([np.max(c, axis=0) for c in init_contours], axis=0)

    # Crop to fit the extent.
    if crop_margin >= 0:
        crop_xmin = max(0, xmin - crop_margin)
        crop_ymin = max(0, ymin - crop_margin)
        crop_xmax = min(img.shape[1] - 1, xmax + crop_margin)
        crop_ymax = min(img.shape[0] - 1, ymax + crop_margin)
        cropped_img = img[crop_ymin:crop_ymax+1, crop_xmin:crop_xmax+1]
    else:
        crop_xmin = 0
        crop_ymin = 0
        crop_xmax = img.shape[1] - 1
        crop_ymax = img.shape[0] - 1
        cropped_img = img

    # Form initial levelsets
    init_levelsets = []
    if init_contours is None:
        for m in init_submasks:
            init_levelset = m[crop_ymin:crop_ymax+1, crop_xmin:crop_xmax+1]
            init_levelset[:10, :] = 0
            init_levelset[-10:, :] = 0
            init_levelset[:, :10] = 0
            init_levelset[:, -10:] = 0
            init_levelsets.append(init_levelset)
    else:
        for cnt in init_contours:
            init_contours_on_cropped_img = cnt - (crop_xmin, crop_ymin)
            init_levelset = contours_to_mask([cnt], cropped_img.shape[:2])
            init_levelset[:10, :] = 0
            init_levelset[-10:, :] = 0
            init_levelset[:, :10] = 0
            init_levelset[:, -10:] = 0
            init_levelsets.append(init_levelset)

    sys.stderr.write('Found %d levelsets.\n' % len(init_levelsets))

    #####################
    # Evolve morphsnake #
    #####################

    final_masks = []

    for levelset_ind, init_levelset in enumerate(init_levelsets):

        sys.stderr.write('\nContour %d\n' % levelset_ind)

        discard = False
        init_area = np.count_nonzero(init_levelset)

        t = time.time()

        msnake = morphsnakes.MorphACWE(cropped_img.astype(np.float), smoothing=int(MORPHSNAKE_SMOOTHING),
                                       lambda1=lambda1, lambda2=MORPHSNAKE_LAMBDA2)

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
            #
            # area = np.count_nonzero(msnake.levelset)
            #
            # if area < min_size:
            #     discard = True
            #     sys.stderr.write('Too small, stop iteration.\n')
            #     break

            labeled_mask = label(msnake.levelset.astype(np.bool))
            component_sizes = []
            for l in np.unique(labeled_mask):
                if l != 0:
                    m = labeled_mask == l
                    component_area = np.count_nonzero(m)
                    component_sizes.append(component_area)
                    if component_area/float(init_area) > AREA_CHANGE_RATIO_MAX:
                        msnake.levelset[m] = 0
                        sys.stderr.write('Component area expands too much - nullified.\n')
                    elif component_area < min_size:
                        msnake.levelset[m] = 0
                        sys.stderr.write('Component area is too small - nullified.\n')
            # print component_sizes

            if  np.count_nonzero(msnake.levelset)/float(init_area) < AREA_CHANGE_RATIO_MIN:
                discard = True
                sys.stderr.write('Area shrinks too much, stop iteration.\n')
                break

            dq.append(msnake.levelset)

    #         t = time.time()
            msnake.step()
    #         sys.stderr.write('Step: %f seconds\n' % (time.time()-t)) # 0.6 second/step, roughly 200 steps takes 120s

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
                    if np.count_nonzero(m) > min_size:
                        final_masks.append(m)
                        sys.stderr.write('Final masks added.\n')

    if len(final_masks) == 0:
        sys.stderr.write('Snake return no valid submasks.\n')
        return []

    if return_masks:
        final_masks_uncropped = []
        for m in final_masks:
            uncropped_mask = np.zeros(img.shape[:2], np.bool)
            uncropped_mask[crop_ymin:crop_ymax+1, crop_xmin:crop_xmax+1] = m
            final_masks_uncropped.append(uncropped_mask)
        return final_masks_uncropped
    else:
        final_contours = []
        for m in final_masks:
            cnts = [cnt_on_cropped + (crop_xmin, crop_ymin) for cnt_on_cropped in find_contour_points(m)[1]]
            final_contours += cnts
        return final_contours

# def save_submasks_one_section(self, submask_decisions, submasks=None, submask_contour_vertices=None, fn=None, sec=None, img_shape=None):
#     """
#     Args:
#         submasks (dict):
#         submask_contour_vertices (dict):
#     """
#
#     if fn is None and sec is not None:
#         fn = metadata_cache['sections_to_filenames'][sec]
#
#     submasks_dir = create_if_not_exists(os.path.join(THUMBNAIL_DATA_DIR, self.stack, self.stack + '_alignedTo_' + self.anchor_fn + '_auto_submasks'))
#     submask_fn_dir = os.path.join(submasks_dir, fn)
#
#     if submasks is not None: # submasks provided
#         submask_contour_vertices = {}
#         for submask_ind, m in submasks.iteritems():
#             cnts = find_contour_points(m)[1]
#             assert len(cnts) == 1, "Must have exactly one contour per submask."
#             submask_contour_vertices[submask_ind] = cnts[0]
#     else:
#         assert submask_contour_vertices is not None
#         submasks = {submask_ind: contours_to_mask(contours=[submask_verts], img_shape=img_shape)
#         for submask_ind, submask_verts in submask_contour_vertices}
#
#     # Save submasks
#     for submask_ind, m in submasks.iteritems():
#         submask_fp = os.path.join(submask_fn_dir, fn + '_alignedTo_' + self.anchor_fn + '_submask_%d.png' % submask_ind)
#         imsave(submask_fp, np.uint8(m)*255)
#
#     # Save submask contour vertices
#     submask_contour_vertices_fp = os.path.join(submask_fn_dir, fn + '_alignedTo_' + self.anchor_fn + '_submask_contour_vertices.pkl')
#     submask_contour_vertices_dict = {}
#     for submask_ind, m in submasks[sec].iteritems():
#         cnts = find_contour_points(m)[1]
#         assert len(cnts) == 1, "Must have exactly one contour per submask."
#         submask_contour_vertices_dict[submask_ind] = cnts[0]
#     save_pickle(submask_contour_vertices_dict, submask_contour_vertices_fp)
#
#     # Save submask decisions
#     decisions_fp = os.path.join(submask_fn_dir, fn +'_alignedTo_' + self.anchor_fn +  '_submask_decisions.txt')
#     from pandas import Series
#     Series(submask_decisions).to_csv(decisions_fp)

# def generate_submask_review_results(stack, filenames):
#     sys.stderr.write('Generate submask review...\n')
#
#     mask_alg_review_results = {}
#     for img_fn in filenames:
#         decisions = generate_submask_review_results_one_section(stack=stack, fn=img_fn)
#         if decisions is None:
#             sys.stderr.write('No review results found: %s.\n' % img_fn)
#             mask_alg_review_results[img_fn] = {}
#         else:
#             mask_alg_review_results[img_fn] = decisions
#
#     return cleanup_mask_review(mask_alg_review_results)
#
#
# def generate_submask_review_results_one_section(stack, fn):
#     review_fp = os.path.join(THUMBNAIL_DATA_DIR, "%(stack)s/%(stack)s_submasks/%(img_fn)s/%(img_fn)s_submasksAlgReview.txt") % \
#                 dict(stack=stack, img_fn=fn)
#     return parse_submask_review_results_one_section_from_file(review_fp)
#
# def parse_submask_review_results_one_section_from_file(review_fp):
#
#     if not os.path.exists(review_fp):
#         return
#
#     decisions = {}
#     with open(review_fp, 'r') as f:
#         for line in f:
#             mask_ind, decision = map(int, line.split())
#             decisions[mask_ind] = decision == 1
#
#     if len(decisions) == 0:
#         return
#     else:
#         return decisions
#
# def cleanup_mask_review(d):
#     """
#     Return {filename: {submask_ind: bool}}
#     """
#
#     labels_reviewed = {}
#
#     for fn, alg_labels in d.iteritems():
#         if len(alg_labels) == 0:
#             labels_reviewed[fn] = {}
#         else:
#             alg_positives = [index for index, l in alg_labels.iteritems() if l == 1]
#             if len(alg_positives) > 0:
#                 assert len(alg_positives) == 1
#                 correct_index = alg_positives[0]
#             else:
#                 correct_index = 1
#
#             alg_labels[correct_index] = 1
#             for idx in alg_labels:
#                 if idx != correct_index:
#                     alg_labels[idx] = -1
#
#             labels_reviewed[fn] = {i: r == 1 for i, r in alg_labels.iteritems()}
