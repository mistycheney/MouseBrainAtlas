#! /usr/bin/env python

import sys
import os

import time

# t = time.time()

import numpy as np
from joblib import Parallel, delayed

from skimage.filters.rank import entropy
from skimage.morphology import remove_small_objects, disk, remove_small_holes
from skimage.measure import label, regionprops
from skimage.color import rgb2gray, gray2rgb
from skimage.io import imread, imsave
from skimage.segmentation import active_contour
from skimage.filters import gaussian, threshold_adaptive
from skimage.util import img_as_ubyte

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from annotation_utilities import points_inside_contour
from registration_utilities import find_contour_points
from utilities2015 import create_if_not_exists

# sys.stderr.write('%f seconds.\n' % (time.time() - t))
# sys.exit(0)

#######################################################################################

import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Generate mask for thumbnail images - nissl')
parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("input_dir", type=str, help="input dir")
parser.add_argument("filenames", type=str, help="image filenames, json encoded, no extensions")
parser.add_argument("output_dir", type=str, help="output dir")
parser.add_argument("--tb_fmt", type=str, help="thumbnail format (tif or png)", default='tif')
args = parser.parse_args()

stack = args.stack_name
input_dir = args.input_dir
output_dir = args.output_dir
tb_fmt = args.tb_fmt

create_if_not_exists(output_dir)

import json

sys.stderr.write('%s\n' % args.filenames)

filenames = json.loads(args.filenames)

sys.stderr.write('%s\n' % filenames)

# thumbnail_fmt = 'tif'

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
    min_size = min(max_area, 5000)

    mask_rso = remove_small_objects(mask, min_size=min_size, connectivity=8)
    mask_rsh = remove_small_holes(mask_rso, min_size=20000, connectivity=8)
    return mask_rsh


def generate_mask(fn, tb_fmt='tif'):
    try:
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
            final_masks.append(final_mask)

        final_mask = np.any(final_masks, axis=0)

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

    except Exception as e:
        sys.stderr.write(e.message + '\n')
        sys.stderr.write('Mask error: %s\n' % fn)
        return

_ = Parallel(n_jobs=15)(delayed(generate_mask)(fn, tb_fmt=tb_fmt) for fn in filenames)
