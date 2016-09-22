#! /usr/bin/env python

import sys
import os
import numpy as np
import cPickle as pickle
import argparse

from joblib import Parallel, delayed

from skimage.filters.rank import entropy
from skimage.morphology import remove_small_objects, disk, remove_small_holes
from skimage.measure import label, regionprops, find_contours
from skimage.color import rgb2gray
from skimage.io import imread, imsave
from skimage import img_as_float
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from skimage.filter import threshold_adaptive, canny

from sklearn import mixture
from sklearn.cluster import KMeans

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from annotation_utilities import *

#######################################################################################

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Generate mask for thumbnail images')
parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("input_dir", type=str, help="input dir")
parser.add_argument("filenames", type=str, help="image filenames, json encoded")
parser.add_argument("output_dir", type=str, help="output dir")
args = parser.parse_args()

stack = args.stack_name
input_dir = args.input_dir
output_dir = args.output_dir

import json
filenames = json.loads(args.filenames)
sys.stderr.write('%s\n' % filenames)

thumbnail_fmt = 'tif'

def generate_entropy_mask(img):

    h, w = img.shape
    e = entropy(img, disk(5))

    x = np.atleast_2d(e[e > .1]).T

    bics = []
    clfs = []
    for nc in [2,3]:
        clf = mixture.GMM(n_components=nc, covariance_type='full')
        clf.fit(x)
        bic = clf.bic(x)
        bics.append(bic)
        clfs.append(clf)

    print 'num. components', [2,3][np.argsort(bics)[0]]

    clf = clfs[np.argsort(bics)[0]]

    means = np.atleast_1d(np.squeeze(clf.means_))

    order = np.argsort(means)
    covars = np.atleast_1d(np.squeeze(clf.covars_))
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


def generate_mask(fn):
    try:
        img_rgb = imread(os.path.join(input_dir, fn + '.tif'))
        img = rgb2gray(img_rgb)

        entropy_mask = generate_entropy_mask(img)

        init_contours = [yxs[:,::-1] for yxs in find_contours(entropy_mask, .5) if len(yxs) > 50]
        assert len(init_contours) > 0, 'No contour is detected from entropy mask %s' % fn

        img_adap = threshold_adaptive(img, 51)
        img_adap[~entropy_mask] = 1

        final_masks = []

        for init_cnt in init_contours:

        # init_cnt = init_contours[0]
            snake = active_contour(img_adap, init_cnt,
                                   alpha=1., beta=10., gamma=0.01,
                                   w_line=0., w_edge=1.,
                                   max_iterations=100)

            bg = np.zeros(img.shape[:2], bool)
            xys = points_inside_contour(snake.astype(np.int))
            bg[np.minimum(xys[:,1], bg.shape[0]-1), np.minimum(xys[:,0], bg.shape[1]-1)] = 1

            final_mask = bg & entropy_mask
            final_masks.append(final_mask)

        final_mask = np.any(final_masks, axis=0)

        mask_fn = os.path.join(output_dir, '%(fn)s_mask.png' % dict(fn=fn))
        imsave(mask_fn, img_as_ubyte(final_mask))

    except Exception as e:
        sys.stderr.write(e.message + '\n')
        sys.stderr.write('%d, Mask error: %s\n' % (len(final_masks), fn))
        return


_ = Parallel(n_jobs=16)(delayed(generate_mask)(fn) for fn in filenames)
