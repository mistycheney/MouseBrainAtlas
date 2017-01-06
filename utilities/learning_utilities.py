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

# import pandas as pd
from pandas import read_hdf, DataFrame

from itertools import groupby

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
    """
    probs: n_example x n_class
    """

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

    axis.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1, **kwargs)
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

# def export_images_given_patch_addresses(addresses, downscale_factor, fn_template, name_to_color):
#     """
#     fn_template: a str including argument %stack and %sec
#     """
#
#     locations = locate_patches_given_addresses(addresses)
#
#     locations_grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
#     for list_index, (stack, sec, loc, name) in enumerate(locations):
#         locations_grouped[stack][sec][name].append(loc)
#     locations_grouped.default_factory = None
#
#     for stack, locations_allSections in locations_grouped.iteritems():
#         for sec, locations_allNames in locations_allSections.iteritems():
#             clrs = [name_to_color[name] for name in locations_allNames.iterkeys()]
#             locs = locations_allNames.values()
#             viz = patch_boxes_overlay_on(bg='original', downscale_factor=downscale_factor,
#                                         locs=locs, colors=clrs, patch_size=224, stack=stack, sec=sec)
#             cv2.imwrite(fn_template % {'stack':stack, 'sec':sec}, viz[...,::-1])


# def get_names_given_locations(stack, section, label_polygons=None, username=None, locations=None, indices=None, grid_spec=None, grid_locations=None):
#
#     if label_polygons is None:
#         label_polygons = load_label_polygons_if_exists(stack, username)
#
#     # if indices is not None:
#     #     assert locations is None, 'Cannot specify both indices and locs.'
#     #     if grid_locations is None:
#     #         grid_locations = grid_parameters_to_sample_locations(grid_spec)
#     #     locations = grid_locations[indices]
#
#     index_to_name_mapping = {}
#     for name, full_indices in label_polygons.loc[section].dropna().to_dict():
#         for i in full_indices:
#             index_to_name_mapping[i] = name
#
#     return [index_to_name_mapping[i] for i in indices]


def extract_patches_given_locations(stack, sec, locs=None, indices=None, grid_spec=None,
                                    grid_locations=None, version='rgb-jpg'):
    """

    """

    img = imread(DataManager.get_image_filepath(stack, sec, version=version))

    if grid_spec is None:
        grid_spec = get_default_gridspec(stack)

    patch_size, stride, w, h = grid_spec
    half_size = patch_size/2

    if indices is not None:
        assert locs is None, 'Cannot specify both indices and locs.'
        if grid_locations is None:
            grid_locations = grid_parameters_to_sample_locations(grid_spec)
        locs = grid_locations[indices]

    patches = [img[y-half_size:y+half_size, x-half_size:x+half_size].copy() for x, y in locs]
    return patches


# def locate_patches_given_addresses(addresses):
#     """
#     addresses is a list of addresses.
#     address: stack, section, structure_name, index
#     """
#
#     from collections import defaultdict
#     addresses_grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
#     for addressList_index, (stack, sec, name, i) in enumerate(addresses):
#         addresses_grouped[stack][sec][name].append((i, addressList_index))
#
#     patch_locations_all = []
#     addressList_indices_all = []
#     for stack, indices_allSections in addresses_grouped.iteritems():
#         grid_spec = get_default_gridspec(stack)
#         sample_locations = grid_parameters_to_sample_locations(grid_spec)
#         indices_allLandmarks_allSections = locate_annotated_patches(stack, grid_spec)
#         for sec, indices_allNames in indices_allSections.iteritems():
#             for name, fwInd_addrLstInd_tuples in indices_allNames.iteritems():
#                 landmarkWise_indices_selected, addressList_indices = map(list, zip(*fwInd_addrLstInd_tuples))
#                 frameWise_indices_oneLandmark = indices_allLandmarks_allSections[sec][name]
#                 frameWise_indices_selected = frameWise_indices_oneLandmark[landmarkWise_indices_selected]
#
#                 patch_locations_all += [(stack, sec, loc, name) for loc in sample_locations[frameWise_indices_selected]]
#                 addressList_indices_all += addressList_indices
#
#     patch_locations_all_inOriginalOrder = [patch_locations_all[i] for i in np.argsort(addressList_indices_all)]
#     return patch_locations_all_inOriginalOrder


def locate_patches_given_addresses_v2(addresses):
    """
    addresses is a list of addresses.
    address: stack, section, framewise_index
    """

    from collections import defaultdict
    addresses_grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for addressList_index, (stack, sec, i) in enumerate(addresses):
        addresses_grouped[stack][sec].append((i, addressList_index))

    patch_locations_all = []
    addressList_indices_all = []
    for stack, indices_allSections in addresses_grouped.iteritems():
        grid_spec = get_default_gridspec(stack)
        sample_locations = grid_parameters_to_sample_locations(grid_spec)
        for sec, fwInd_addrLstInd_tuples in indices_allSections.iteritems():
            frameWise_indices_selected, addressList_indices = map(list, zip(*fwInd_addrLstInd_tuples))
            patch_locations_all += [(stack, sec, loc, name) for loc in sample_locations[frameWise_indices_selected]]
            addressList_indices_all += addressList_indices

    patch_locations_all_inOriginalOrder = [patch_locations_all[i] for i in np.argsort(addressList_indices_all)]
    return patch_locations_all_inOriginalOrder

def extract_patches_given_locations_multiple_sections(addresses, location_or_grid_index='location', version='rgb-jpg'):
    """
    addresses is a list of addresses.
    address: stack, section, framewise_index
    """

    from collections import defaultdict
    # locations_grouped = defaultdict(lambda: defaultdict(list))
    # for list_index, (stack, sec, loc) in enumerate(addresses):
    #     locations_grouped[stack][sec].append((loc, list_index))

    locations_grouped = {}
    for stack_sec, list_index_and_address_grouper in groupby(sorted(enumerate(addresses), key=lambda (i, x): (x[0],x[1])),
        key=lambda (i,x): (x[0], x[1])):
        locations_grouped[stack_sec] = [(address[2], list_index) for list_index, address in list_index_and_address_grouper]

    # locations_grouped = {stack_sec: (x[2], list_index) \
    #                     for stack_sec, (list_index, x) in groupby(enumerate(addresses), lambda (i, x): (x[0],x[1]))}

    patches_all = []
    list_indices_all = []
    # for stack, locations_allSections in locations_grouped.iteritems():
    #     for sec, loc_listInd_tuples in locations_allSections.iteritems():
    for stack_sec, locations_allSections in locations_grouped.iteritems():
        stack, sec = stack_sec
        locs_thisSec, list_indices = map(list, zip(*locations_allSections))
        if location_or_grid_index == 'location':
            extracted_patches = extract_patches_given_locations(stack, sec, locs=locs_thisSec, version=version)
        else:
            extracted_patches = extract_patches_given_locations(stack, sec, indices=locs_thisSec, version=version)
        patches_all += extracted_patches
        list_indices_all += list_indices

    patch_all_inOriginalOrder = [patches_all[i] for i in np.argsort(list_indices_all)]
    return patch_all_inOriginalOrder

def get_names_given_locations_multiple_sections(addresses, location_or_grid_index='location', username=None):

    # from collections import defaultdict
    # locations_grouped = {stack_sec: (x[2], list_index) \
    #                     for stack_sec, (list_index, x) in group_by(enumerate(addresses), lambda i, x: (x[0],x[1]))}

    locations_grouped = defaultdict(lambda: defaultdict(list))
    for list_index, (stack, sec, loc) in enumerate(addresses):
        locations_grouped[stack][sec].append((loc, list_index))

    # if indices is not None:
    #     assert locations is None, 'Cannot specify both indices and locs.'
    #     if grid_locations is None:
    #         grid_locations = grid_parameters_to_sample_locations(grid_spec)
    #     locations = grid_locations[indices]

    names_all = []
    list_indices_all = []
    for stack, locations_allSections in locations_grouped.iteritems():

        structure_grid_indices = locate_annotated_patches(stack=stack, username=username, force=True,
                                                        annotation_rootdir=annotation_midbrainIncluded_v2_rootdir)

        for sec, loc_listInd_tuples in locations_allSections.iteritems():

            label_dict = structure_grid_indices[sec].dropna().to_dict()

            locs_thisSec, list_indices = map(list, zip(*loc_listInd_tuples))

            index_to_name_mapping = {}

            for name, full_indices in label_dict.iteritems():
                for i in full_indices:
                    index_to_name_mapping[i] = name

            if location_or_grid_index == 'location':
                raise Exception('Not implemented.')
            else:
                names = [index_to_name_mapping[i] if i in index_to_name_mapping else 'BackG' for i in locs_thisSec]

            names_all += names
            list_indices_all += list_indices

    names_all_inOriginalOrder = [names_all[i] for i in np.argsort(list_indices_all)]
    return names_all_inOriginalOrder



# def extract_patches_given_addresses(addresses):
#
#     locations = locate_patches_given_addresses(addresses)
#
#     from collections import defaultdict
#     locations_grouped = defaultdict(lambda: defaultdict(list))
#     for list_index, (stack, sec, loc, name) in enumerate(locations):
#         locations_grouped[stack][sec].append((loc, list_index))
#
#     patches_all = []
#     list_indices_all = []
#     for stack, locations_allSections in locations_grouped.iteritems():
#         for sec, loc_listInd_tuples in locations_allSections.iteritems():
#             locs_thisSec, list_indices = map(list, zip(*loc_listInd_tuples))
#             patches_all += extract_patches_given_locations(stack, sec, locs=locs_thisSec)
#             list_indices_all += list_indices
#
#     patch_all_inOriginalOrder = [patches_all[i] for i in np.argsort(list_indices_all)]
#     return patch_all_inOriginalOrder


def get_default_gridspec(stack, patch_size=224, stride=56):
    # image_width, image_height = DataManager.get_image_dimension(stack)
    image_width, image_height = metadata_cache['image_shape'][stack]
    return (patch_size, stride, image_width, image_height)

def locate_annotated_patches_v2(stack, grid_spec=None, annotation_rootdir=None):
    """
    If exists, load from <patch_rootdir>/<stack>_indices_allLandmarks_allSection.h5

    Return a DataFrame: indexed by structure names and section number, cell is the array of grid indices.
    """

    # if not force:
    #     try:
    #         fn = os.path.join(patch_rootdir, '%(stack)s_indices_allLandmarks_allSection.h5' % {'stack':stack})
    #         indices_allLandmarks_allSections_df = pd.read_hdf(fn, 'framewise_indices')
    #         return indices_allLandmarks_allSections_df
    #     except Exception as e:
    #         sys.stderr.write(e.message)

    if grid_spec is None:
        grid_spec = get_default_gridspec(stack)

    contours_df = read_hdf(annotation_midbrainIncluded_v2_rootdir + '/%(stack)s/%(stack)s_annotation_v3.h5' % dict(stack=stack), 'contours')
    contours = contours_df[(contours_df['orientation'] == 'sagittal') & (contours_df['downsample'] == 1)]
    contours = contours.drop_duplicates(subset=['section', 'name', 'side', 'filename', 'downsample', 'creator'])
    contours = convert_annotation_v3_original_to_aligned_cropped(contours, stack=stack)

    # filename_to_section, section_to_filename = DataManager.load_sorted_filenames(stack)
    sections_to_filenames = metadata_cache['sections_to_filenames'][stack]

    grouped = contours.groupby('section')

    patch_indices_allSections_allStructures = {}
    for sec, group in grouped:
        sys.stderr.write('Analyzing section %d..\n' % sec)
        if sections_to_filenames[sec] in ['Placeholder', 'Nonexisting, Rescan']:
            continue
        polygons_this_sec = [(contour['name'], contour['vertices']) for contour_id, contour in group.iterrows()]
        mask_tb = DataManager.load_thumbnail_mask_v2(stack, sec)
        patch_indices = locate_patches_v2(grid_spec=grid_spec, mask_tb=mask_tb, polygons=polygons_this_sec)
        patch_indices_allSections_allStructures[sec] = patch_indices

    patch_indices_allSections_allStructures_df = DataFrame(patch_indices_allSections_allStructures)

    # if dump:
    #     fn = os.path.join(patch_rootdir, '%(stack)s_indices_allLandmarks_allSection.h5' % {'stack':stack})
    #     indices_allLandmarks_allSections_df = pd.to_hdf(fn, 'framewise_indices')
    #     return indices_allLandmarks_allSections_df

    return patch_indices_allSections_allStructures_df


def sample_locations(grid_indices_lookup, structures, num_samples_per_polygon=None, num_samples_per_landmark=None):
    """
    Return address_list (section, grid_idx).
    """

    location_list = defaultdict(list)

    for name in structures:

        if name not in grid_indices_lookup.index:
            continue

        for sec, grid_indices in grid_indices_lookup.loc[name].dropna().to_dict().iteritems():

            n = len(grid_indices)

            if n == 0:
                sys.stderr.write('Cell is empty.\n')
                continue

            if num_samples_per_polygon is None:
                location_list[name] += [(sec, i) for i in grid_indices]

            else:
                random_sampled_indices = grid_indices[np.random.choice(range(n), min(n, num_samples_per_polygon), replace=False)]
                location_list[name] += [(sec, i) for i in random_sampled_indices]

    if num_samples_per_landmark is not None:

        sampled_location_list = {}
        for name_s, addresses in location_list.iteritems():
            n = len(addresses)
            random_sampled_indices = np.random.choice(range(n), min(n, num_samples_per_landmark), replace=False)
            sampled_location_list[name_s] = [addresses[i] for i in random_sampled_indices]
        return sampled_location_list

    else:
        location_list.default_factory = None
        return location_list


def grid_parameters_to_sample_locations(grid_spec=None, patch_size=None, stride=None, w=None, h=None):
    # patch_size, stride, w, h = grid_parameters.tolist()

    if grid_spec is not None:
        patch_size, stride, w, h = grid_spec

    half_size = patch_size/2
    ys, xs = np.meshgrid(np.arange(half_size, h-half_size, stride), np.arange(half_size, w-half_size, stride),
                     indexing='xy')
    sample_locations = np.c_[xs.flat, ys.flat]
    return sample_locations

def locate_patches_v2(grid_spec=None, stack=None, patch_size=224, stride=56, image_shape=None, mask_tb=None, polygons=None, bbox=None):
    """
    Return addresses of patches that are either in polygons or on mask.
    - If mask is given, the valid patches are those whose centers are True. bbox and polygons are ANDed with mask.
    - If bbox is given, valid patches are those entirely inside bbox. bbox = (x,y,w,h)
    - If polygons is given, the valid patches are those whose bounding boxes.
        - polygons can be a dict, keys are structure names, values are vertices (xy).
        - polygons can also be a list of (name, vertices) tuples.
    if shrinked to 30%% are completely within the polygons.

    scheme: 1 - negative does not include other positive classes that are in the surround
            2 - negative include other positive classes that are in the surround
    """

    if grid_spec is not None:
        patch_size, stride, image_width, image_height = grid_spec
    elif image_shape is not None :
        image_width, image_height = image_shape
    else:
        image_width, image_height = DataManager.get_image_dimension(stack)

    sample_locations = grid_parameters_to_sample_locations(patch_size=patch_size, stride=stride, w=image_width, h=image_height)
    half_size = patch_size/2

    indices_fg = np.where(mask_tb[sample_locations[:,1]/32, sample_locations[:,0]/32])[0] # patches in the foreground
    indices_bg = np.setdiff1d(range(sample_locations.shape[0]), indices_fg) # patches in the background

    if polygons is not None:
        if isinstance(polygons, dict):
            polygon_list = [(name, cnt) for name, cnts in polygons.iteritems() for cnt in cnts] # This is to deal with when one name has multiple contours
        elif isinstance(polygons, list):
            polygon_list = polygons
        else:
            raise Exception('Polygon must be either dict or list.')

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
        # indices_surround = {}

        indices_allLandmarks = {}

        for label, poly in polygon_list:

            path = Path(poly)
            indices_inside[label] = np.where(path.contains_points(sample_locations_ll) &\
                                              path.contains_points(sample_locations_lr) &\
                                              path.contains_points(sample_locations_ul) &\
                                              path.contains_points(sample_locations_ur))[0]

            indices_allLandmarks[label] = indices_inside[label]
            sys.stderr.write('%d patches in %s\n' % (len(indices_allLandmarks[label]), label))

        indices_allInside = np.concatenate(indices_inside.values())

        for label, poly in polygon_list:

            surround = Polygon(poly).buffer(500, resolution=2)

            path = Path(list(surround.exterior.coords))
            indices_sur =  np.where(path.contains_points(sample_locations_ll) &\
                                    path.contains_points(sample_locations_lr) &\
                                    path.contains_points(sample_locations_ul) &\
                                    path.contains_points(sample_locations_ur))[0]

            # surround classes do not include patches of any no-surround class
            indices_allLandmarks[label+'_surround_noclass'] = np.setdiff1d(indices_sur, np.r_[indices_bg, indices_allInside])
            sys.stderr.write('%d patches in %s\n' % (len(indices_allLandmarks[label+'_surround_noclass']), label+'_surround_noclass'))
            # print len(indices_allLandmarks[label+'_surround_noclass']), 'patches in', label+'_surround_noclass'

            for l, inds in indices_inside.iteritems():
                if l == label: continue
                indices = np.intersect1d(indices_sur, inds)
                if len(indices) > 0:
                    indices_allLandmarks[label+'_surround_'+l] = indices
                    sys.stderr.write('%d patches in %s\n' % (len(indices), label+'_surround_'+l))

            # # all foreground patches except the particular label's inside patches
            indices_allLandmarks[label+'_negative'] = np.setdiff1d(range(sample_locations.shape[0]), np.r_[indices_bg, indices_inside[label]])
            sys.stderr.write('%d patches in %s\n' % (len(indices_allLandmarks[label+'_negative']), label+'_negative'))

        # for l, inds in indices_inside.iteritems():
        #     indices_allLandmarks[l] = inds
        #     print len(inds), 'patches in', l
        # for l, inds in indices_surround.iteritems():
        #     indices_allLandmarks[l+'_surround_noclass'] = inds
        #     print len(inds), 'patches in', l+'_surround_noclass'
        indices_allLandmarks['bg'] = indices_bg
        sys.stderr.write('%d patches in %s\n' % (len(indices_bg), 'bg'))

    return indices_allLandmarks


# def locate_patches(grid_spec=None, stack=None, patch_size=224, stride=56, image_shape=None, mask_tb=None, polygons=None, bbox=None):
#     """
#     Return addresses of patches that are either in polygons or on mask.
#     - If mask is given, the valid patches are those whose centers are True. bbox and polygons are ANDed with mask.
#     - If bbox is given, valid patches are those entirely inside bbox. bbox = (x,y,w,h)
#     - If polygons is given, the valid patches are those whose bounding boxes. polygons is a dict, keys are structure names, values are vertices (xy).
#     if shrinked to 30%% are completely within the polygons.
#     """
#
#     if grid_spec is not None:
#         patch_size, stride, image_width, image_height = grid_spec
#     elif image_shape is not None :
#         image_width, image_height = image_shape
#     else:
#         image_width, image_height = DataManager.get_image_dimension(stack)
#
#     sample_locations = grid_parameters_to_sample_locations(patch_size=patch_size, stride=stride, w=image_width, h=image_height)
#     half_size = patch_size/2
#
#     indices_fg = np.where(mask_tb[sample_locations[:,1]/32, sample_locations[:,0]/32])[0]
#     indices_bg = np.setdiff1d(range(sample_locations.shape[0]), indices_fg)
#
#     if bbox is not None:
#         assert polygons is None, 'Can only supply one of bbox or polygons.'
#
#         box_x, box_y, box_w, box_h = detect_bbox_lookup[stack] if bbox is None else bbox
#
#         xmin = max(half_size, box_x*32)
#         xmax = min(image_width-half_size-1, (box_x+box_w)*32)
#         ymin = max(half_size, box_y*32)
#         ymax = min(image_height-half_size-1, (box_y+box_h)*32)
#
#         indices_roi = np.where(np.all(np.c_[sample_locations[:,0] > xmin, sample_locations[:,1] > ymin,
#                                             sample_locations[:,0] < xmax, sample_locations[:,1] < ymax], axis=1))[0]
#
#         indices_roi = np.setdiff1d(indices_roi, indices_bg)
#         print len(indices_roi), 'patches in ROI'
#
#         return indices_roi
#
#     else:
#         assert polygons is not None, 'Can only supply one of bbox or polygons.'
#
#         # This means we require a patch to have 30% of its radius to be within the landmark boundary to be considered inside the landmark
#         margin = int(.3*half_size)
#
#         sample_locations_ul = sample_locations - (margin, margin)
#         sample_locations_ur = sample_locations - (-margin, margin)
#         sample_locations_ll = sample_locations - (margin, -margin)
#         sample_locations_lr = sample_locations - (-margin, -margin)
#
#         indices_inside = {}
#         indices_surround = {}
#
#         for label, poly in polygons.iteritems():
#
#             path = Path(poly)
#             indices_inside[label] =  np.where(path.contains_points(sample_locations_ll) &\
#                                               path.contains_points(sample_locations_lr) &\
#                                               path.contains_points(sample_locations_ul) &\
#                                               path.contains_points(sample_locations_ur))[0]
#
#         indices_allInside = np.concatenate(indices_inside.values())
#
#         for label, poly in polygons.iteritems():
#
#             surround = Polygon(poly).buffer(500, resolution=2)
#
#             path = Path(list(surround.exterior.coords))
#             indices_sur =  np.where(path.contains_points(sample_locations_ll) &\
#                                     path.contains_points(sample_locations_lr) &\
#                                     path.contains_points(sample_locations_ul) &\
#                                     path.contains_points(sample_locations_ur))[0]
#
#             # surround classes do not include patches of any no-surround class
#             indices_surround[label] = np.setdiff1d(indices_sur, np.r_[indices_bg, indices_allInside])
#
#
#         indices_allLandmarks = {}
#         for l, inds in indices_inside.iteritems():
#             indices_allLandmarks[l] = inds
#             print len(inds), 'patches in', l
#         for l, inds in indices_surround.iteritems():
#             indices_allLandmarks[l+'_surround'] = inds
#             print len(inds), 'patches in', l+'_surround'
#         indices_allLandmarks['bg'] = indices_bg
#         print len(indices_bg), 'patches in', 'bg'
#
#     return indices_allLandmarks


def addresses_to_features(addresses):
    """
    If certain input address is outside the mask, the corresponding feature returned is None.
    """

    feature_list = []

    list_indices_all_stack_section = []

    invalid_list_indices = []

    for st_se, group in groupby(sorted(enumerate(addresses), key=lambda (i,(st,se,idx)): (st, se)),
           key=lambda (i,(st,se,idx)): (st, se)):

        print st_se

        list_indices, addrs = zip(*group)

        sampled_grid_indices = [idx for st, se, idx in addrs]

        stack, sec = st_se

        anchor_fn = metadata_cache['anchor_fn'][stack]
        fn = metadata_cache['sections_to_filenames'][stack][sec]

        if fn in ['Placeholder', 'Nonexisting', 'Rescan']:
            sys.stderr.write('Image file is %s.\n' % (fn))

            feature_list += [None for _ in sampled_grid_indices]
            list_indices_all_stack_section += list_indices

        else:

            # Load mapping grid index -> location
            locations_fn = PATCH_FEATURES_ROOTDIR + '/%(stack)s/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_patch_locations.txt' % dict(stack=stack, fn=fn, anchor_fn=anchor_fn)

            with open(locations_fn, 'r') as f:
                all_locations = [int(line.split()[0]) for line in f.readlines()]

#             all_locations = [idx for idx, x, y in locations]

            sampled_list_indices = []
#             list_indices2 = []
            for gi, lst_idx in zip(sampled_grid_indices, list_indices):
                if gi in all_locations:
                    sampled_list_indices.append(all_locations.index(gi))
#                     list_indices2.append(lst_idx)
                else:
                    sys.stderr.write('Patch in annotation but not in mask: %s %d %s alignedTo %s @%d\n' % (stack, sec, fn, anchor_fn, gi))
                    invalid_list_indices.append(lst_idx)
                    sampled_list_indices.append(None)
#                     list_indices2.append(lst_idx)

            feature_fn = PATCH_FEATURES_ROOTDIR + '/%(stack)s/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_features.hdf' % dict(stack=stack, fn=fn, anchor_fn=anchor_fn)

            features = load_hdf(feature_fn)
            # sampled_features = features[sampled_list_indices]
            # sampled_features = [features[i] if i is not None else None for i in sampled_list_indices]
            # feature_list += list(sampled_features)

            feature_list += [features[i].copy() if i is not None else None for i in sampled_list_indices]
            del features

#             list_indices_all_stack_section += list_indices2
            list_indices_all_stack_section += list_indices



    return [feature_list[i] for i in np.argsort(list_indices_all_stack_section)]


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
