import numpy as np
import sys
import os

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from metadata import *
from data_manager import *

import matplotlib.pyplot as plt
from matplotlib.path import Path
from shapely.geometry import Polygon

import pandas as pd
try:
    import mxnet as mx
except:
    sys.stderr.write('Cannot import mxnet.\n')

from joblib import Parallel, delayed
import time

def grid_parameters_to_sample_locations(grid_spec=None, patch_size=None, stride=None, w=None, h=None):
    # patch_size, stride, w, h = grid_parameters.tolist()

    if grid_spec is not None:
        patch_size, stride, w, h = grid_spec

    half_size = patch_size/2
    ys, xs = np.meshgrid(np.arange(half_size, h-half_size, stride), np.arange(half_size, w-half_size, stride),
                     indexing='xy')
    sample_locations = np.c_[xs.flat, ys.flat]
    return sample_locations

def locate_patches(grid_spec=None, stack=None, patch_size=224, stride=56, image_shape=None, mask_tb=None, polygons=None, bbox=None):
    """
    Return addresses of patches that are either in polygons or on mask.
    - If mask is given, the valid patches are those whose centers are True. bbox and polygons are ANDed with mask.
    - If bbox is given, valid patches are those entirely inside bbox.
    - If polygons is given, the valid patches are those whose bounding boxes
    if shrinked to 30%% are completely within the polygons.
    """

    if grid_spec is not None:
        patch_size, stride, image_width, image_height = grid_spec
    elif image_shape is not None :
        image_width, image_height = image_shape
    else:
        image_width, image_height = DataManager.get_image_dimension(stack)

    sample_locations = grid_parameters_to_sample_locations(patch_size=patch_size, stride=stride, w=image_width, h=image_height)
    half_size = patch_size/2

    indices_fg = np.where(mask_tb[sample_locations[:,1]/32, sample_locations[:,0]/32])[0]
    indices_bg = np.setdiff1d(range(sample_locations.shape[0]), indices_fg)

    if bbox is not None:
        assert polygons is None, 'Can only supply one of bbox or polygons.'

        box_x, box_y, box_w, box_h = detect_bbox_lookup[stack] if bbox is None else bbox

        xmin = max(half_size, box_x*32)
        xmax = min(image_width-half_size-1, (box_x+box_w)*32)
        ymin = max(half_size, box_y*32)
        ymax = min(image_height-half_size-1, (box_y+box_h)*32)

        indices_roi = np.where(np.all(np.c_[sample_locations[:,0] > xmin, sample_locations[:,1] > ymin,
                                            sample_locations[:,0] < xmax, sample_locations[:,1] < ymax], axis=1))[0]

        indices_roi = np.setdiff1d(indices_roi, indices_bg)
        print len(indices_roi), 'patches in ROI'

        return indices_roi

    else:
        assert polygons is not None, 'Can only supply one of bbox or polygons.'

        # This means we require a patch to have 30% of its radius to be within the landmark boundary to be considered inside the landmark
        margin = int(.3*half_size)

        sample_locations_ul = sample_locations - (margin, margin)
        sample_locations_ur = sample_locations - (-margin, margin)
        sample_locations_ll = sample_locations - (margin, -margin)
        sample_locations_lr = sample_locations - (-margin, -margin)

        indices_inside = {}
        indices_surround = {}

        for label, poly in polygons.iteritems():

            path = Path(poly)
            indices_inside[label] =  np.where(path.contains_points(sample_locations_ll) &\
                                              path.contains_points(sample_locations_lr) &\
                                              path.contains_points(sample_locations_ul) &\
                                              path.contains_points(sample_locations_ur))[0]

        indices_allInside = np.concatenate(indices_inside.values())

        for label, poly in polygons.iteritems():

            surround = Polygon(poly).buffer(500, resolution=2)

            path = Path(list(surround.exterior.coords))
            indices_sur =  np.where(path.contains_points(sample_locations_ll) &\
                                    path.contains_points(sample_locations_lr) &\
                                    path.contains_points(sample_locations_ul) &\
                                    path.contains_points(sample_locations_ur))[0]

            # surround classes do not include patches of any no-surround class
            indices_surround[label] = np.setdiff1d(indices_sur, np.r_[indices_bg, indices_allInside])


        indices_allLandmarks = {}
        for l, inds in indices_inside.iteritems():
            indices_allLandmarks[l] = inds
            print len(inds), 'patches in', l
        for l, inds in indices_surround.iteritems():
            indices_allLandmarks[l+'_surround'] = inds
            print len(inds), 'patches in', l+'_surround'
        indices_allLandmarks['bg'] = indices_bg
        print len(indices_bg), 'patches in', 'bg'

    return indices_allLandmarks


def visualize_filters(model, name, input_channel=0, title=''):

    filters = model.arg_params[name].asnumpy()

    n = len(filters)

    ncol = 16
    nrow = n/ncol

    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*.5, nrow*.5), sharex=True, sharey=True)

    fig.suptitle(title)

    axes = axes.flatten()
    for i in range(n):
        axes[i].matshow(filters[i][input_channel], cmap=plt.cm.gray)
        axes[i].tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            left='off',
            right='off',
            labelbottom='off',
            labeltop='off',
            labelright='off',
            labelleft='off') # labels along the bottom edge are off
        axes[i].axis('equal')
    plt.show()
