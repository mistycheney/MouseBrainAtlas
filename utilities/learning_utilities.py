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

from collections import defaultdict
from visualization_utilities import *
from annotation_utilities import *

def compute_accuracy(predictions, true_labels, exclude_abstained=True, abstain_label=-1):

    n = len(predictions)
    if exclude_abstained:
        predicted_indices = predictions != abstain_label
        acc = np.count_nonzero(predictions[predicted_indices] == true_labels[predicted_indices]) / float(len(predicted_indices))
    else:
        acc = np.count_nonzero(predictions == true_labels) / float(n)

    return acc

def compute_predictions(H, abstain_label=-1):
    predictions = np.argmax(H, axis=1)
    no_decision_indices = np.where(np.all(H == 0, axis=1))[0]

    if abstain_label == 'random':
        predictions[no_decision_indices] = np.random.randint(0, H.shape[1], len(no_decision_indices)).astype(np.int)
    else:
        predictions[no_decision_indices] = abstain_label

    return predictions, no_decision_indices

def compute_confusion_matrix(probs, labels, soft=False, normalize=True, abstain_label=-1):

    n_labels = len(np.unique(labels))

    pred_is_hard = np.array(probs).ndim == 1

    if pred_is_hard:
        soft = False

    M = np.zeros((n_labels, n_labels))
    for probs0, tl in zip(probs, labels):
        if soft:
            M[tl] += probs0
        else:
            if pred_is_hard:
                if probs0 != abstain_label:
                    M[tl, probs0] += 1
            else:
                hard = np.zeros((n_labels, ))
                hard[np.argmax(probs0)] = 1.
                M[tl] += hard

    if normalize:
        M_normalized = M.astype(np.float)/M.sum(axis=1)[:, np.newaxis]
        return M_normalized
    else:
        return M

def plot_confusion_matrix(cm, labels, title='Confusion matrix', cmap=plt.cm.Blues, figsize=(4,4), text=True, axis=None, **kwargs):

    if axis is None:
        fig = plt.figure(figsize=figsize)
        axis = fig.add_subplot(1,1,1)

    axis.imshow(cm, interpolation='nearest', cmap=cmap, **kwargs)
    axis.set_title(title)
#     plt.colorbar()
    axis.set_xticks(np.arange(len(labels)))
    axis.set_yticks(np.arange(len(labels)))
    axis.set_xticklabels(labels)
    axis.set_yticklabels(labels)

    axis.set_ylabel('True label')
    axis.set_xlabel('Predicted label')

    if cm.dtype.type is np.int_:
        fmt = '%d'
    else:
        fmt = '%.2f'

    if text:
        for x in xrange(len(labels)):
            for y in xrange(len(labels)):
                if not np.isnan(cm[y,x]):
                    axis.text(x,y, fmt % cm[y,x],
                             horizontalalignment='center',
                             verticalalignment='center');

def export_images_given_patch_addresses(addresses, downscale_factor, fn_template, name_to_color):
    """
    fn_template: a str including argument %stack and %sec
    """

    locations = locate_patches_given_addresses(addresses)

    locations_grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for list_index, (stack, sec, loc, name) in enumerate(locations):
        locations_grouped[stack][sec][name].append(loc)
    locations_grouped.default_factory = None

    for stack, locations_allSections in locations_grouped.iteritems():
        for sec, locations_allNames in locations_allSections.iteritems():
            clrs = [name_to_color[name] for name in locations_allNames.iterkeys()]
            locs = locations_allNames.values()
            viz = patch_boxes_overlay_on(bg='original', downscale_factor=downscale_factor,
                                        locs=locs, colors=clrs, patch_size=224, stack=stack, sec=sec)
            cv2.imwrite(fn_template % {'stack':stack, 'sec':sec}, viz[...,::-1])


def extract_patches_given_locations(stack, sec, locs=None, grid_spec=None, indices=None, sample_locations=None):

    img = imread(DataManager.get_image_filepath(stack, sec))

    if grid_spec is None:
        grid_spec = get_default_gridspec(stack)

    patch_size, stride, w, h = grid_spec
    half_size = patch_size/2

    if indices is not None:
        assert locs is None, 'Cannot specify both indices and locs.'
        if sample_locations is None:
            sample_locations = grid_parameters_to_sample_locations(grid_spec)
        locs = sample_locations[indices]

    patches = [img[y-half_size:y+half_size, x-half_size:x+half_size] for x, y in locs]
    return patches


def locate_patches_given_addresses(addresses):
    """
    addresses is a list of addresses.
    address: stack, section, structure_name, index
    """

    from collections import defaultdict
    addresses_grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for addressList_index, (stack, sec, name, i) in enumerate(addresses):
        addresses_grouped[stack][sec][name].append((i, addressList_index))

    patch_locations_all = []
    addressList_indices_all = []
    for stack, indices_allSections in addresses_grouped.iteritems():
        grid_spec = get_default_gridspec(stack)
        sample_locations = grid_parameters_to_sample_locations(grid_spec)
        indices_allLandmarks_allSections = locate_annotated_patches(stack, grid_spec)
        for sec, indices_allNames in indices_allSections.iteritems():
            for name, fwInd_addrLstInd_tuples in indices_allNames.iteritems():
                landmarkWise_indices_selected, addressList_indices = map(list, zip(*fwInd_addrLstInd_tuples))
                frameWise_indices_oneLandmark = indices_allLandmarks_allSections[sec][name]
                frameWise_indices_selected = frameWise_indices_oneLandmark[landmarkWise_indices_selected]

                patch_locations_all += [(stack, sec, loc, name) for loc in sample_locations[frameWise_indices_selected]]
                addressList_indices_all += addressList_indices

    patch_locations_all_inOriginalOrder = [patch_locations_all[i] for i in np.argsort(addressList_indices_all)]
    return patch_locations_all_inOriginalOrder


def extract_patches_given_addresses(addresses):

    locations = locate_patches_given_addresses(addresses)

    from collections import defaultdict
    locations_grouped = defaultdict(lambda: defaultdict(list))
    for list_index, (stack, sec, loc, name) in enumerate(locations):
        locations_grouped[stack][sec].append((loc, list_index))

    patches_all = []
    list_indices_all = []
    for stack, locations_allSections in locations_grouped.iteritems():
        for sec, loc_listInd_tuples in locations_allSections.iteritems():
            locs_thisSec, list_indices = map(list, zip(*loc_listInd_tuples))
            patches_all += extract_patches_given_locations(stack, sec, locs=locs_thisSec)
            list_indices_all += list_indices

    patch_all_inOriginalOrder = [patches_all[i] for i in np.argsort(list_indices_all)]
    return patch_all_inOriginalOrder


def get_default_gridspec(stack, patch_size=224, stride=56):
    image_width, image_height = DataManager.get_image_dimension(stack)
    return (patch_size, stride, image_width, image_height)

def locate_annotated_patches(stack, grid_spec=None, username='yuncong', force=False):
    """
    If exists, load from <patch_rootdir>/<stack>_indices_allLandmarks_allSection.h5
    """

    if not force:
        try:
            fn = os.path.join(patch_rootdir, '%(stack)s_indices_allLandmarks_allSection.h5' % {'stack':stack})
            indices_allLandmarks_allSections_df = pd.read_hdf(fn, 'framewise_indices')
            return indices_allLandmarks_allSections_df
        except Exception as e:
            sys.stderr.write(e.message)

    if grid_spec is None:
        grid_spec = get_default_gridspec(stack)

    label_polygons = load_label_polygons_if_exists(stack, username, force=force,
                        annotation_rootdir=annotation_midbrainIncluded_rootdir)

    first_sec, last_sec = detect_bbox_range_lookup[stack]
    # bar = show_progress_bar(first_sec, last_sec)

    indices_allLandmarks_allSections = {}
    for sec in range(first_sec, last_sec+1):
        # bar.value = sec

        if sec in label_polygons.index:

            mask_tb = DataManager.load_thumbnail_mask(stack, sec)
            indices_allLandmarks = locate_patches(grid_spec=grid_spec, mask_tb=mask_tb, polygons=label_polygons.loc[sec].dropna())
            indices_allLandmarks_allSections[sec] = indices_allLandmarks
        else:
            sys.stderr.write('Section %d has no labelings.\n' % sec)


    indices_allLandmarks_allSections_df = pd.DataFrame(indices_allLandmarks_allSections)

    return indices_allLandmarks_allSections_df


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
