import sys
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from shapely.geometry import Polygon
from pandas import read_hdf, DataFrame, read_csv
from itertools import groupby
from collections import defaultdict

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from metadata import *
from data_manager import *
from visualization_utilities import *
from annotation_utilities import *

def load_datasets(dataset_ids, labels_to_sample, clf_rootdir=CLF_ROOTDIR):
    
    merged_features = {}
    merged_addresses = {}

    for dataset_id in dataset_ids:

        # load training addresses

        addresses_fp = os.path.join(clf_rootdir, 'datasets', 'dataset_%d' % dataset_id, 'patch_addresses.pkl')
        download_from_s3_to_ec2(addresses_fp)
        addresses_curr_dataset = load_pickle(addresses_fp)
        
        # Load training features

        features_fp = os.path.join(clf_rootdir, 'datasets', 'dataset_%d' % dataset_id, 'patch_features.hdf')
        download_from_s3_to_ec2(features_fp)
        features_curr_dataset = load_hdf_v2(features_fp).to_dict()

        for label in labels_to_sample:
            try:
                if label not in merged_features:
                    merged_features[label] = features_curr_dataset[label]
                else:
                    merged_features[label] = np.vstack([merged_features[label], features_curr_dataset[label]])

                if label not in merged_addresses:
                    merged_addresses[label] = addresses_curr_dataset[label]
                else:
                    merged_addresses[label] += addresses_curr_dataset[label]

            except Exception as e:
                continue
                                
    return merged_features, merged_addresses


################
## Evaluation ##
################

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

def plot_confusion_matrix(cm, labels, title='Confusion matrix', cmap=plt.cm.Blues, figsize=(4,4), text=True, axis=None,
xlabel='Predicted label', ylabel='True label', **kwargs):

    if axis is None:
        fig = plt.figure(figsize=figsize)
        axis = fig.add_subplot(1,1,1)
        return_fig = True
    else:
        return_fig = False

    axis.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1, **kwargs)
    axis.set_title(title)
#     plt.colorbar()
    axis.set_xticks(np.arange(len(labels)))
    axis.set_yticks(np.arange(len(labels)))
    axis.set_xticklabels(labels)
    axis.set_yticklabels(labels)

    axis.set_ylabel(ylabel)
    axis.set_xlabel(xlabel)

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

    if return_fig:
        return fig

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

def extract_patches_given_locations(stack, sec, locs=None, indices=None, grid_spec=None,
                                    grid_locations=None, version='compressed'):
    """
    Return patches at given locations.

    Args:
        indices:
            grid indices
        locs:
            patch centers

    Returns:
        list of patches.
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

    #print [(y-half_size, y+half_size, x-half_size, x+half_size) for x, y in locs]
    patches = [img[y-half_size:y+half_size, x-half_size:x+half_size].copy() for x, y in locs]
    return patches

def extract_patches_given_locations_multiple_sections(addresses, location_or_grid_index='location', version='compressed'):
    """
    Args:
        addresses:
            list of (stack, section, framewise_index)
    Returns:
        list of patches
    """

    from collections import defaultdict

    locations_grouped = {}
    for stack_sec, list_index_and_address_grouper in groupby(sorted(enumerate(addresses), key=lambda (i, x): (x[0],x[1])),
        key=lambda (i,x): (x[0], x[1])):
        locations_grouped[stack_sec] = [(address[2], list_index) for list_index, address in list_index_and_address_grouper]

    patches_all = []
    list_indices_all = []
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


def label_regions_multisections(stack, region_contours, surround_margins=None):
    """
    Args:
        region_contours: dict {sec: list of contours (nx2 array)}
    """

    # contours_df = read_hdf(ANNOTATION_ROOTDIR + '/%(stack)s/%(stack)s_annotation_v3.h5' % dict(stack=stack), 'contours')
    contours_df, _ = DataManager.load_annotation_v3(stack=stack)
    contours = contours_df[(contours_df['orientation'] == 'sagittal') & (contours_df['downsample'] == 1)]
    contours = contours.drop_duplicates(subset=['section', 'name', 'side', 'filename', 'downsample', 'creator'])
    labeled_contours = convert_annotation_v3_original_to_aligned_cropped(contours, stack=stack)

    sections_to_filenames = metadata_cache['sections_to_filenames'][stack]

    labeled_region_indices_all_sections = {}

    for sec, region_contours_curr_sec in region_contours.iteritems():

        sys.stderr.write('Analyzing section %d..\n' % section)

        matching_contours = labeled_contours[labeled_contours['section'] == section]
        if len(matching_contours) == 0:
            continue
        polygons_this_sec = [(contour['name'], contour['vertices']) for contour_id, contour in matching_contours.iterrows()]
        mask_tb = DataManager.load_thumbnail_mask_v2(stack, section)
        labeled_region_indices = identify_regions_inside(region_contours=region_contours, mask_tb=mask_tb, polygons=polygons_this_sec, \
                                            surround_margins=surround_margins)

        labeled_region_indices_all_sections[sec] = labeled_region_indices

    return labeled_region_indices_all_sections


def label_regions(stack, section, region_contours, surround_margins=None, labeled_contours=None):
    """
    Identify regions that belong to each class.
    
    Returns:
        dict of {label: region indices in input region_contours}
    """

    if is_invalid(sec=section, stack=stack):
        raise Exception('Section %s, %d invalid.' % (stack, section))

    if labeled_contours is None:
        # contours_df = read_hdf(ANNOTATION_ROOTDIR + '/%(stack)s/%(stack)s_annotation_v3.h5' % dict(stack=stack), 'contours')
        contours_df, _ = DataManager.load_annotation_v3(stack=stack)
        contours = contours_df[(contours_df['orientation'] == 'sagittal') & (contours_df['downsample'] == 1)]
        contours = contours.drop_duplicates(subset=['section', 'name', 'side', 'filename', 'downsample', 'creator'])
        contours = convert_annotation_v3_original_to_aligned_cropped(contours, stack=stack)
        labeled_contours = contours[contours['section'] == section]

    sys.stderr.write('Analyzing section %d..\n' % section)

    if len(labeled_contours) == 0:
        return {}

    polygons_this_sec = [(contour['name'], contour['vertices']) for contour_id, contour in labeled_contours.iterrows()
                        if contour['name'] in all_known_structures]

    mask_tb = DataManager.load_thumbnail_mask_v2(stack, section)
    patch_indices = identify_regions_inside(region_contours=region_contours, mask_tb=mask_tb, polygons=polygons_this_sec, \
                                        surround_margins=surround_margins)
    return patch_indices

def identify_regions_inside(region_contours, stack=None, image_shape=None, mask_tb=None, polygons=None, surround_margins=None):
    """
    Return addresses of patches that are in polygons or on mask.
    - If mask is given, the valid patches are those whose centers are True. bbox and polygons are ANDed with mask.
    - If polygons is given, the valid patches are those whose bounding boxes.
        - polygons can be a dict, keys are structure names, values are x-y vertices (nx2 array).
        - polygons can also be a list of (name, vertices) tuples.
    if shrinked to 30%% are completely within the polygons.

    scheme: 1 - negative does not include other positive classes that are in the surround
            2 - negative include other positive classes that are in the surround
    """

    n_regions = len(region_contours)

    if isinstance(polygons, dict):
        polygon_list = [(name, cnt) for name, cnts in polygons.iteritems() for cnt in cnts] # This is to deal with when one name has multiple contours
    elif isinstance(polygons, list):
        assert isinstance(polygons[0], tuple)
        polygon_list = polygons
    else:
        raise Exception('Polygon must be either dict or list.')

    if surround_margins is None:
        surround_margins = [100,200,300,400,500,600,700,800,900,1000]

    # Identify patches in the foreground.
    region_centroids_tb = np.array([np.mean(cnt, axis=0)/32. for cnt in region_contours], np.int)
    indices_fg = np.where(mask_tb[region_centroids_tb[:,1], region_centroids_tb[:,0]])[0]
    # indices_fg = np.where(mask_tb[sample_locations[:,1]/32, sample_locations[:,0]/32])[0] # patches in the foreground
    # sys.stderr.write('%d patches in fg\n' % len(indices_fg))

    # Identify patches in the background.
    indices_bg = np.setdiff1d(range(n_regions), indices_fg)

    indices_inside = {}
    indices_allLandmarks = {}

    for label, poly in polygon_list:
        path = Path(poly)
        # indices_inside[label] = np.where([np.all(path.contains_points(cnt)) for cnt in region_contours])[0]

        # Being inside is defined as 70% of vertices are within the polygon
        indices_inside[label] = np.where([np.count_nonzero(path.contains_points(cnt)) >= len(cnt)*.7 for cnt in region_contours])[0]
        indices_allLandmarks[label] = indices_inside[label]
        # sys.stderr.write('%d patches in %s\n' % (len(indices_allLandmarks[label]), label))

    indices_allInside = np.concatenate(indices_inside.values())

    for label, poly in polygon_list:

        for margin in surround_margins:
        # margin = 500
            surround = Polygon(poly).buffer(margin, resolution=2)
            # 500 pixels (original resolution, ~ 250um) away from landmark contour

            path = Path(list(surround.exterior.coords))
            indices_sur = np.where([np.count_nonzero(path.contains_points(cnt)) >=  len(cnt)*.7  for cnt in region_contours])[0]

            # surround classes do not include patches of any no-surround class
            indices_allLandmarks[label+'_surround_'+str(margin)+'_noclass'] = np.setdiff1d(indices_sur, np.r_[indices_bg, indices_allInside])
            # sys.stderr.write('%d patches in %s\n' % (len(indices_allLandmarks[label+'_surround_'+str(margin)+'_noclass']), label+'_surround_'+str(margin)+'_noclass'))
            # print len(indices_allLandmarks[label+'_surround_noclass']), 'patches in', label+'_surround_noclass'

            for l, inds in indices_inside.iteritems():
                if l == label: continue
                indices = np.intersect1d(indices_sur, inds)
                if len(indices) > 0:
                    indices_allLandmarks[label+'_surround_'+str(margin)+'_'+l] = indices
                    # sys.stderr.write('%d patches in %s\n' % (len(indices), label+'_surround_'+str(margin)+'_'+l))

        # Identify all foreground patches except the particular label's inside patches
        indices_allLandmarks[label+'_negative'] = np.setdiff1d(range(n_regions), np.r_[indices_bg, indices_inside[label]])
        # sys.stderr.write('%d patches in %s\n' % (len(indices_allLandmarks[label+'_negative']), label+'_negative'))

    indices_allLandmarks['bg'] = indices_bg
    indices_allLandmarks['noclass'] = np.setdiff1d(range(n_regions), np.r_[indices_bg, indices_allInside])

    return indices_allLandmarks


def locate_annotated_patches_v2(stack, grid_spec=None, sections=None, surround_margins=None):
    """
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

    contours_df = read_hdf(ANNOTATION_ROOTDIR + '/%(stack)s/%(stack)s_annotation_v3.h5' % dict(stack=stack), 'contours')
    contours = contours_df[(contours_df['orientation'] == 'sagittal') & (contours_df['downsample'] == 1)]
    contours = contours.drop_duplicates(subset=['section', 'name', 'side', 'filename', 'downsample', 'creator'])
    contours = convert_annotation_v3_original_to_aligned_cropped(contours, stack=stack)

    # filename_to_section, section_to_filename = DataManager.load_sorted_filenames(stack)
    sections_to_filenames = metadata_cache['sections_to_filenames'][stack]

    grouped = contours.groupby('section')

    patch_indices_allSections_allStructures = {}
    for sec, group in grouped:
        if sections is not None:
            if sec not in sections:
                continue
        sys.stderr.write('Analyzing section %d..\n' % sec)
        if is_invalid(sec=sec, stack=stack):
            continue
        polygons_this_sec = [(contour['name'], contour['vertices']) for contour_id, contour in group.iterrows()]
        mask_tb = DataManager.load_thumbnail_mask_v2(stack, sec)
        patch_indices = locate_patches_v2(grid_spec=grid_spec, mask_tb=mask_tb, polygons=polygons_this_sec, \
                                            surround_margins=surround_margins)
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
    !!! This should be sped up! It is now taking 100 seconds for one stack !!!
    """

    location_list = defaultdict(list)

    for name in structures:

        if name not in grid_indices_lookup.index:
            continue

        # t = time.time()
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
                
        # print name, time.time()-t

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



def generate_dataset(num_samples_per_label, stacks, labels_to_sample, model_name='Inception-BN'):
    """
    Generate dataset.
    - Extract addresses
    - Map addresses to features
    - Remove None features
    
    Returns:
        test_features, test_addresses, labels_found
    """
    
    # Extract addresses
    
    test_addresses = defaultdict(list)

    labels_found = set([])

    t = time.time()
    
    for stack in stacks:
        
        t1 = time.time()
        annotation_grid_indices_fn = os.path.join(ANNOTATION_ROOTDIR, stack, stack + '_annotation_grid_indices.h5')
        grid_indices_per_label = read_hdf(annotation_grid_indices_fn, 'grid_indices')
        sys.stderr.write('Read: %.2f seconds\n' % (time.time() - t1))

        labels_this_stack = set(grid_indices_per_label.index) & set(labels_to_sample)
        labels_found = labels_found | labels_this_stack

        t1 = time.time()
        test_addresses_sec_idx = sample_locations(grid_indices_per_label, labels_this_stack, 
                                                  num_samples_per_landmark=num_samples_per_label/len(stacks))
        sys.stderr.write('Sample addresses (stack %s): %.2s seconds.\n' % (stack, time.time() - t1))

        for label, addresses in test_addresses_sec_idx.iteritems():
            test_addresses[label] += [(stack, ) + addr for addr in addresses]

    test_addresses.default_factory = None
    
    sys.stderr.write('Sample addresses: %.2f seconds\n' % (time.time() - t))
    
    # Map addresses to features
    
    t = time.time()
    # test_features = apply_function_to_dict(lambda x: addresses_to_features(x, model_name=model_name), test_addresses)
    test_features = apply_function_to_dict(lambda x: addresses_to_features_parallel(x, model_name=model_name, n_processes=4), test_addresses)
    sys.stderr.write('Map addresses to features: %.2f seconds\n' % (time.time() - t))
    
    # Remove features that are None

    for name in labels_found:
        valid = [(ftr, addr) for ftr, addr in zip(test_features[name], test_addresses[name])
                    if ftr is not None]
        res = zip(*valid)
        test_features[name] = np.array(res[0])
        test_addresses[name] = res[1]
        
    return test_features, test_addresses, labels_found
    

def grid_parameters_to_sample_locations(grid_spec=None, patch_size=None, stride=None, w=None, h=None):
    # patch_size, stride, w, h = grid_parameters.tolist()

    if grid_spec is not None:
        patch_size, stride, w, h = grid_spec

    half_size = patch_size/2
    ys, xs = np.meshgrid(np.arange(half_size, h-half_size, stride), np.arange(half_size, w-half_size, stride),
                     indexing='xy')
    sample_locations = np.c_[xs.flat, ys.flat]
    return sample_locations

def locate_patches_v2(grid_spec=None, stack=None, patch_size=None, stride=None, image_shape=None, mask_tb=None, polygons=None, bbox=None,
bbox_lossless=None, surround_margins=None):
    """
    Return addresses of patches that are either in polygons or on mask.
    - If mask is given, the valid patches are those whose centers are True. bbox and polygons are ANDed with mask.
    - If bbox is given, valid patches are those entirely inside bbox. bbox = (x,y,w,h) in thumbnail resol.
    - If polygons is given, the valid patches are those whose bounding boxes.
        - polygons can be a dict, keys are structure names, values are x-y vertices (nx2 array).
        - polygons can also be a list of (name, vertices) tuples.
    if shrinked to 30%% are completely within the polygons.

    scheme: 1 - negative does not include other positive classes that are in the surround
            2 - negative include other positive classes that are in the surround
    """

    if grid_spec is None:
        grid_spec = get_default_gridspec(stack)

    image_shape = grid_spec[2:]

    if image_shape is not None:
        image_width, image_height = image_shape
    else:
        image_width, image_height = metadata_cache['image_shape'][stack]

    if patch_size is None:
        patch_size = grid_spec[0]

    if stride is None:
        stride = grid_spec[1]

    if surround_margins is None:
        surround_margins = [100,200,300,400,500,600,700,800,900,1000]

    sample_locations = grid_parameters_to_sample_locations(patch_size=patch_size, stride=stride, w=image_width, h=image_height)
    half_size = patch_size/2

    indices_fg = np.where(mask_tb[sample_locations[:,1]/32, sample_locations[:,0]/32])[0] # patches in the foreground
    # sys.stderr.write('%d patches in fg\n' % len(indices_fg))

    indices_bg = np.setdiff1d(range(sample_locations.shape[0]), indices_fg) # patches in the background

    if polygons is None and bbox is None and bbox_lossless is None:
        return indices_fg

    if polygons is not None:
        if isinstance(polygons, dict):
            polygon_list = [(name, cnt) for name, cnts in polygons.iteritems() for cnt in cnts] # This is to deal with when one name has multiple contours
        elif isinstance(polygons, list):
            assert isinstance(polygons.values()[0], tuple)
            polygon_list = polygons
        else:
            raise Exception('Polygon must be either dict or list.')

    if bbox is not None or bbox_lossless is not None:
        assert polygons is None, 'Can only supply one of bbox or polygons.'

        if bbox is not None:

            box_x, box_y, box_w, box_h = bbox

            xmin = max(half_size, box_x*32)
            xmax = min(image_width-half_size-1, (box_x+box_w)*32)
            ymin = max(half_size, box_y*32)
            ymax = min(image_height-half_size-1, (box_y+box_h)*32)

        else:
            box_x, box_y, box_w, box_h = bbox_lossless
            # print box_x, box_y, box_w, box_h

            xmin = max(half_size, box_x)
            xmax = min(image_width-half_size-1, box_x+box_w-1)
            ymin = max(half_size, box_y)
            ymax = min(image_height-half_size-1, box_y+box_h-1)

        # print xmin, xmax, ymin, ymax

        indices_roi = np.where(np.all(np.c_[sample_locations[:,0] > xmin, sample_locations[:,1] > ymin,
                                            sample_locations[:,0] < xmax, sample_locations[:,1] < ymax], axis=1))[0]

        indices_roi = np.setdiff1d(indices_roi, indices_bg)
        # print len(indices_roi), 'patches in ROI'

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
            # sys.stderr.write('%d patches in %s\n' % (len(indices_allLandmarks[label]), label))

        indices_allInside = np.concatenate(indices_inside.values())

        for label, poly in polygon_list:

            ############# surrround = 500 #############
            for margin in surround_margins:
            # margin = 500
                surround = Polygon(poly).buffer(margin, resolution=2)
                # 500 pixels (original resolution, ~ 250um) away from landmark contour

                path = Path(list(surround.exterior.coords))
                indices_sur =  np.where(path.contains_points(sample_locations_ll) &\
                                        path.contains_points(sample_locations_lr) &\
                                        path.contains_points(sample_locations_ul) &\
                                        path.contains_points(sample_locations_ur))[0]

                # surround classes do not include patches of any no-surround class
                indices_allLandmarks[label+'_surround_'+str(margin)+'_noclass'] = np.setdiff1d(indices_sur, np.r_[indices_bg, indices_allInside])
                # sys.stderr.write('%d patches in %s\n' % (len(indices_allLandmarks[label+'_surround_'+str(margin)+'_noclass']), label+'_surround_'+str(margin)+'_noclass'))
                # print len(indices_allLandmarks[label+'_surround_noclass']), 'patches in', label+'_surround_noclass'

                for l, inds in indices_inside.iteritems():
                    if l == label: continue
                    indices = np.intersect1d(indices_sur, inds)
                    if len(indices) > 0:
                        indices_allLandmarks[label+'_surround_'+str(margin)+'_'+l] = indices
                        # sys.stderr.write('%d patches in %s\n' % (len(indices), label+'_surround_'+str(margin)+'_'+l))

            # # all foreground patches except the particular label's inside patches
            indices_allLandmarks[label+'_negative'] = np.setdiff1d(range(sample_locations.shape[0]), np.r_[indices_bg, indices_inside[label]])
            # sys.stderr.write('%d patches in %s\n' % (len(indices_allLandmarks[label+'_negative']), label+'_negative'))

        # for l, inds in indices_inside.iteritems():
        #     indices_allLandmarks[l] = inds
        #     print len(inds), 'patches in', l
        # for l, inds in indices_surround.iteritems():
        #     indices_allLandmarks[l+'_surround_noclass'] = inds
        #     print len(inds), 'patches in', l+'_surround_noclass'
        indices_allLandmarks['bg'] = indices_bg
        # sys.stderr.write('%d patches in %s\n' % (len(indices_bg), 'bg'))

        indices_allLandmarks['noclass'] = np.setdiff1d(range(sample_locations.shape[0]), np.r_[indices_bg, indices_allInside])
        # sys.stderr.write('%d patches in %s\n' % (len(indices_allLandmarks['noclass']), 'noclass'))

        return indices_allLandmarks

def addresses_to_structure_distances(addresses, structure_centers_all_stacks_all_secs_all_names):
    """
    Return a list of dict {structure_name: distance}.
    """
    augmented_addresses = [addr + (i,) for i, addr in enumerate(addresses)]

    all_stacks = set([ad[0] for ad in addresses])
    grid_locations = {st: grid_parameters_to_sample_locations(get_default_gridspec(st)) for st in all_stacks}

    sorted_addresses = sorted(augmented_addresses, key=lambda (st,sec,gp_ind,list_ind): (st,sec))

    distances_to_structures = defaultdict(list)

    for (st, sec), address_group_ in groupby(sorted_addresses, lambda (st,sec,gp_ind,list_ind): (st,sec)):
        address_group = list(address_group_) # otherwise it is an iteraror, which can only be used once
        locations_this_group = np.array([grid_locations[st][gp_ind] for st,sec,gp_ind,list_ind in address_group])
        list_indices_this_group = [list_ind for st,sec,gp_ind,list_ind in address_group]

        for structure_name in structure_centers_all_stacks_all_secs_all_names[st][sec].keys():
            distances = np.sqrt(np.sum((locations_this_group - structure_centers_all_stacks_all_secs_all_names[st][sec][structure_name])**2, axis=1))
            distances_to_structures[structure_name] += zip(list_indices_this_group, distances)

    distances_to_structures.default_factory = None

    d = {structure_name: dict(lst) for structure_name, lst in distances_to_structures.iteritems()}

    import pandas
    df = pandas.DataFrame(d)
    return df.T.to_dict().values()

def addresses_to_locations(addresses):
    """
    Take a list of (stack, section, gridpoint_index),
    return x,y coordinates (lossless).
    """

    augmented_addresses = [addr + (i,) for i, addr in enumerate(addresses)]
    sorted_addresses = sorted(augmented_addresses, key=lambda (st,sec,gp_ind,list_ind): st)

    locations = []
    for st, address_group in groupby(sorted_addresses, lambda (st,sec,gp_ind,list_ind): st):
        grid_locations = grid_parameters_to_sample_locations(get_default_gridspec(st))
        locations_this_group = [(list_inf, grid_locations[gp_ind]) for st,sec,gp_ind,list_ind in address_group]
        locations.append(locations_this_group)

    locations = list(chain(*locations))
    return [v for i, v in sorted(locations)]

def addresses_to_features_parallel(addresses, model_name='Sat16ClassFinetuned', n_processes=16):
    """
    If certain input address is outside the mask, the corresponding feature returned is None.
    """

    groups = [(st_se, list(group)) for st_se, group in groupby(sorted(enumerate(addresses), key=lambda (i,(st,se,idx)): (st, se)), key=lambda (i,(st,se,idx)): (st, se))]
    
    def f(x):
        st_se, group = x
        print st_se

        list_indices, addrs = zip(*group)
        sampled_grid_indices = [idx for st, se, idx in addrs]
        stack, sec = st_se
        fn = metadata_cache['sections_to_filenames'][stack][sec]

        if is_invalid(fn):
            sys.stderr.write('Image file is %s.\n' % (fn))
            features_ret = [None for _ in sampled_grid_indices]
            # feature_list += [None for _ in sampled_grid_indices]
            # list_indices_all_stack_section += list_indices
        else:
            # Load mapping grid index -> location
            all_grid_indices, _ = DataManager.load_dnn_feature_locations(stack=stack, model_name=model_name, fn=fn)
            all_grid_indices = all_grid_indices.tolist()

            sampled_list_indices = []
            for gi, lst_idx in zip(sampled_grid_indices, list_indices):
                if gi in all_grid_indices:
                    sampled_list_indices.append(all_grid_indices.index(gi))
                else:
                    sys.stderr.write('Patch in annotation but not in mask: %s %d %s @%d\n' % (stack, sec, fn, gi))
                    sampled_list_indices.append(None)

            features = DataManager.load_dnn_features(stack=stack, model_name=model_name, fn=fn)
            features_ret = [features[i].copy() if i is not None else None for i in sampled_list_indices]
            del features

            # list_indices_all_stack_section += list_indices
        return features_ret, list_indices
        
    from multiprocess import Pool
    pool = Pool(n_processes)
    res = pool.map(f, groups)    
    pool.close()
    pool.join()
    
    feature_list, list_indices = zip(*res)
    feature_list = list(chain(*feature_list))
    list_indices_all_stack_section = list(chain(*list_indices))

    return [feature_list[i] for i in np.argsort(list_indices_all_stack_section)]


def addresses_to_features(addresses, model_name='Sat16ClassFinetuned'):
    """
    If certain input address is outside the mask, the corresponding feature returned is None.
    """

    feature_list = []
    list_indices_all_stack_section = []

    for st_se, group in groupby(sorted(enumerate(addresses), key=lambda (i,(st,se,idx)): (st, se)), key=lambda (i,(st,se,idx)): (st, se)):

        print st_se

        list_indices, addrs = zip(*group)
        sampled_grid_indices = [idx for st, se, idx in addrs]

        stack, sec = st_se

        fn = metadata_cache['sections_to_filenames'][stack][sec]

        if is_invalid(fn):
            sys.stderr.write('Image file is %s.\n' % (fn))
            feature_list += [None for _ in sampled_grid_indices]
            list_indices_all_stack_section += list_indices
        else:
            # Load mapping grid index -> location
            all_grid_indices, _ = DataManager.load_dnn_feature_locations(stack=stack, model_name=model_name, fn=fn)
            all_grid_indices = all_grid_indices.tolist()

            sampled_list_indices = []
            for gi, lst_idx in zip(sampled_grid_indices, list_indices):
                if gi in all_grid_indices:
                    sampled_list_indices.append(all_grid_indices.index(gi))
                else:
                    sys.stderr.write('Patch in annotation but not in mask: %s %d %s @%d\n' % (stack, sec, fn, gi))
                    sampled_list_indices.append(None)

            features = DataManager.load_dnn_features(stack=stack, model_name=model_name, fn=fn)
            feature_list += [features[i].copy() if i is not None else None for i in sampled_list_indices]
            del features

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
