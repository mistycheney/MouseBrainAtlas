#! /usr/bin/env python

import sys
import os
import numpy as np
import cPickle as pickle
import argparse

from joblib import Parallel, delayed

from skimage.filters.rank import entropy
from skimage.morphology import remove_small_objects, disk, remove_small_holes
from skimage.measure import label, regionprops
from skimage.color import rgb2gray
from skimage.io import imread, imsave
from skimage import img_as_float

from sklearn import mixture

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Generate mask for thumbnail images')
parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("input_dir", type=str, help="input dir")
parser.add_argument("first_sec", type=int, help="first section")
parser.add_argument("last_sec", type=int, help="last section")
args = parser.parse_args()

stack = args.stack_name
input_dir = args.input_dir
first_secind = args.first_sec
last_secind = args.last_sec

def generate_mask(img):

    h, w = img.shape
    e = entropy(img, disk(5))

#     plt.figure();
#     plt.title('distribution');
#     plt.hist(e.flatten(), bins=100);

    x = np.atleast_2d(e[e > .1]).T

    bics = []
    clfs = []
    # for nc in [2]:
    for nc in [2,3]:
        clf = mixture.GMM(n_components=nc, covariance_type='full')
        clf.fit(x)
        bic = clf.bic(x)
        bics.append(bic)
        clfs.append(clf)

    # print 'num. components', np.argsort(bics)[0] + 2

    clf = clfs[np.argsort(bics)[0]]

    means = np.squeeze(clf.means_)

    order = np.argsort(means)
    means = means[order]

    covars = np.squeeze(clf.covars_)
    covars = covars[order]

    weights = clf.weights_
    weights = weights[order]

    # consider only the largest two components
    if nc > 2:
        order = sorted(np.argsort(weights)[-2:])
        weights = weights[order]
        covars = covars[order]
        means = means[order]

    counts, bins = np.histogram(e.flat, bins=100, density=True);

    # ignore small components
    gs = np.array([w * 1./np.sqrt(2*np.pi*c) * np.exp(-(bins-m)**2/(2*c)) for m, c, w in zip(means, covars, weights)])

#     plt.figure();
#     plt.title('fitted guassians');
#     plt.plot(bins, gs.T);

    thresh = bins[np.where(gs[-1] - gs[-2] < 0)[0][-1]]
    # print thresh

    mask = e > thresh

    mask = remove_small_objects(mask, min_size=10000, connectivity=8)
    mask = remove_small_holes(mask, min_size=10000, connectivity=8)
    
    return mask


def f(stack, sec):

    img = rgb2gray(imread(input_dir+'/'+stack+'_%04d_'%sec + suffix + '.tif'))
    
    try:
        mask = generate_mask(img)
    except:
        raise Exception('%d'%sec)
    
    img2 = img.copy()
    img2[~mask] = 0

    imsave(mask_dir+'/'+stack+'_%04d_'%sec + suffix + '_mask.png', img_as_float(mask))
    imsave(masked_img_dir+'/'+stack+'_%04d_'%sec + suffix + '_masked.png', img2)


suffix = '_'.join(os.path.split(input_dir)[-1].split('_')[1:])
mask_dir = os.environ['DATA_DIR'] + '/' + stack + '_' + suffix + '_mask'
masked_img_dir = os.environ['DATA_DIR'] + '/' + stack + '_' + suffix + '_masked'

if not os.path.exists(mask_dir):
    try:
        os.makedirs(mask_dir)
    except:
        pass

if not os.path.exists(masked_img_dir):
    try:
        os.makedirs(masked_img_dir)
    except:
        pass
    
_ = Parallel(n_jobs=16)(delayed(f)(stack, sec) for sec in range(first_secind, last_secind+1))