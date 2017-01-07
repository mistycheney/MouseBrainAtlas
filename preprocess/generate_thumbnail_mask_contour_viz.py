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
from registration_utilities import find_contour_points

#######################################################################################

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Generate mask contour visualization for thumbnail images')
parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("image_dir", type=str, help="original image dir")
parser.add_argument("mask_dir", type=str, help="mask dir")
parser.add_argument("filenames", type=str, help="image filenames, json encoded")
parser.add_argument("output_dir", type=str, help="output dir")
parser.add_argument("--tb_fmt", type=str, help="thumbnail format (tif or png)", default='tif')
args = parser.parse_args()

stack = args.stack_name
image_dir = args.image_dir
mask_dir = args.mask_dir
output_dir = args.output_dir
tb_fmt = args.tb_fmt

import json
filenames = json.loads(args.filenames)

def generate_mask_contour_visualization(fn, tb_fmt):
    try:
        mask_fn = os.path.join(mask_dir, '%(fn)s_mask.png' % dict(fn=fn))
        mask = imread(mask_fn)

        contour_xys = find_contour_points(mask.astype(np.bool), sample_every=1)[1]

        image_fn = os.path.join(image_dir, '%(fn)s.%(tb_fmt)s' % dict(fn=fn, tb_fmt=tb_fmt))
        image = imread(image_fn)
        if image.ndim == 2:
            viz = gray2rgb(image)
        elif image.ndim == 3:
            viz = image.copy()
        else:
            raise

        for cnt in contour_xys:
            cv2.polylines(viz, [cnt.astype(np.int)], True, (255,0,0), 1)

        output_fn = os.path.join(output_dir, '%(fn)s_mask_contour_viz.tif' % dict(fn=fn))

        if os.path.exists(output_fn):
            execute_command('rm -rf %s' % output_fn)

        imsave(output_fn, viz)

    except Exception as e:
        sys.stderr.write('%s\n'%e.message)
        sys.stderr.write('Mask error: %s\n' % fn)
        return


_ = Parallel(n_jobs=16)(delayed(generate_mask_contour_visualization)(fn, tb_fmt) for fn in filenames)
