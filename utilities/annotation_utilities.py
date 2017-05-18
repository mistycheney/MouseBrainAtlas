import sys
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from skimage.measure import grid_points_in_poly

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from metadata import *
from data_manager import *


def contours_to_mask(contours, img_shape):
    """
    img_shape: h,w
    """

    final_masks = []

    for cnt in contours:

        bg = np.zeros(img_shape, bool)
        xys = points_inside_contour(cnt.astype(np.int))
        bg[np.minimum(xys[:,1], bg.shape[0]-1), np.minimum(xys[:,0], bg.shape[1]-1)] = 1

        final_masks.append(bg)

    final_mask = np.any(final_masks, axis=0)
    return final_mask


def get_surround_volume(vol, distance=5, valid_level=0):
    """
    Return the volume with voxels surrounding active voxels in the input volume set to 1.

    Args:
        valid_level (float):
            voxels with value above this level are regarded as active.
        distance (int):
            surrounding voxels are closer than distance (in unit of voxel) from any active voxels.
    """
    from scipy.ndimage.morphology import distance_transform_edt
    eps = 5
    xmin, xmax, ymin, ymax, zmin, zmax = bbox_3d(vol)
    ydim, xdim, zdim = vol.shape
    roi_xmin = max(0, xmin - distance - eps)
    roi_ymin = max(0, ymin - distance - eps)
    roi_zmin = max(0, zmin - distance - eps)
    roi_xmax = min(xdim-1, xmax + distance + eps)
    roi_ymax = min(ydim-1, ymax + distance + eps)
    roi_zmax = min(zdim-1, zmax + distance + eps)
    roi = (vol > valid_level)[roi_ymin:roi_ymax+1, roi_xmin:roi_xmax+1, roi_zmin:roi_zmax+1]

    dist_vol = distance_transform_edt(roi == 0)
    roi_surround_vol = (dist_vol > 0) & (dist_vol < distance)

    surround_vol = np.zeros_like(vol)
    surround_vol[roi_ymin:roi_ymax+1, roi_xmin:roi_xmax+1, roi_zmin:roi_zmax+1] = roi_surround_vol
    return surround_vol

def points_inside_contour(cnt, num_samples=None):
    xmin, ymin = cnt.min(axis=0)
    xmax, ymax = cnt.max(axis=0)
    h, w = (ymax-ymin+1, xmax-xmin+1)
    inside_ys, inside_xs = np.where(grid_points_in_poly((h, w), cnt[:, ::-1]-(ymin,xmin)))

    if num_samples is None:
        inside_points = np.c_[inside_xs, inside_ys] + (xmin, ymin)
    else:
        n = inside_ys.size
        random_indices = np.random.choice(range(n), min(1000, n), replace=False)
        inside_points = np.c_[inside_xs[random_indices], inside_ys[random_indices]]

    return inside_points


def assign_sideness(label_polygons, landmark_range_limits=None):
    """Assign left or right suffix to a label_polygons object.
    """

    if landmark_range_limits is None:
        landmark_range_limits = get_landmark_range_limits(label_polygons=label_polygons)

    label_polygons_dict = label_polygons.to_dict()
    label_polygons_sideAssigned_dict = defaultdict(dict)

    for name, v in label_polygons_dict.iteritems():
        if name not in singular_structures:
            name_u = convert_name_to_unsided(name)
            for sec, coords in v.iteritems():
                if np.any(np.isnan(coords)): continue

                lname = convert_to_left_name(name)
                rname = convert_to_right_name(name)

                if lname in landmark_range_limits and sec <= landmark_range_limits[lname][1]:
                    label_polygons_sideAssigned_dict[lname][sec] = coords
                elif rname in landmark_range_limits and sec >= landmark_range_limits[rname][0]:
                    label_polygons_sideAssigned_dict[rname][sec] = coords
                else:
                    print name, sec, landmark_range_limits[lname], landmark_range_limits[rname]
                    raise Exception('label_polygon has structure %s on section %d beyond range limits.' % (name, sec))

        else:
            label_polygons_sideAssigned_dict[name].update({sec:coords for sec, coords in v.iteritems()
                                                           if not np.any(np.isnan(coords))})

    from pandas import DataFrame
    label_polygons_sideAssigned = DataFrame(label_polygons_sideAssigned_dict)
    return label_polygons_sideAssigned



def get_annotation_on_sections(stack=None, username=None, label_polygons=None, filtered_labels=None):
    """
    Get a dictionary, whose keys are landmark names and
    values are indices of sections containing particular landmarks.

    Parameters
    ----------
    stack : str
    username : str
    label_polygon : pandas.DataFrame, optional
    filtered_labels : list of str, optional
    """

    assert stack is not None or label_polygons is not None

    if label_polygons is None:
        label_polygons = load_label_polygons_if_exists(stack, username)

    annotation_on_sections = {}

    if filtered_labels is None:
        labels = set(label_polygons.columns)
    else:
        labels = set(label_polygons.columns) & set(filtered_labels)

    for l in labels:
        annotation_on_sections[l] = list(label_polygons[l].dropna().keys())

    return annotation_on_sections


def get_landmark_range_limits_v2(stack=None, label_section_lookup=None, filtered_labels=None):
    """
    label_section_lookup is a dict, keys are labels, values are sections.
    """

    # first_sec, last_sec = section_range_lookup[stack]

    print label_section_lookup

    first_sec, last_sec = DataManager.load_cropbox(stack)[4:]
    mid_sec = (first_sec + last_sec)/2
    print mid_sec

    landmark_limits = {}

    if filtered_labels is None:
        d = set(label_section_lookup.keys())
    else:
        d = set(label_section_lookup.keys()) & set(filtered_labels)

    d_unsided = set(map(convert_name_to_unsided, d))

    for name_u in d_unsided:

        lname = convert_to_left_name(name_u)
        rname = convert_to_right_name(name_u)

        secs = []

        if name_u in label_section_lookup:
            secs += list(label_section_lookup[name_u])

        if lname in label_section_lookup:
            secs += list(label_section_lookup[lname])

        if rname in label_section_lookup:
            secs += list(label_section_lookup[rname])

        secs = np.array(sorted(secs))

        if name_u in singular_structures: # single
            landmark_limits[name_u] = (secs.min(), secs.max())
        else: # two sides

            if len(secs) == 1:
                sys.stderr.write('Structure %s has label on only one section.\n' % name_u)
                sec = secs[0]
                if sec < mid_sec:
                    landmark_limits[lname] = (sec, sec)
                else:
                    landmark_limits[rname] = (sec, sec)
                continue

            elif len(secs) == 0:
                raise
            else:

                inferred_Ls = secs[secs < mid_sec]
                if len(inferred_Ls) > 0:
                    inferred_maxL = np.max(inferred_Ls)
                else:
                    inferred_maxL = None

                inferred_Rs = secs[secs >= mid_sec]
                if len(inferred_Rs) > 0:
                    inferred_minR = np.min(inferred_Rs)
                else:
                    inferred_minR = None

                # diffs = np.diff(secs)
                # peak = np.argmax(diffs)
                #
                # inferred_maxL = secs[peak]
                # inferred_minR = secs[peak+1]

                if lname in label_section_lookup:
                    labeled_maxL = np.max(label_section_lookup[lname])
                    maxL = max(labeled_maxL, inferred_maxL if inferred_maxL is not None else 0)
                else:
                    maxL = inferred_maxL

                if rname in label_section_lookup:
                    labeled_minR = np.min(label_section_lookup[rname])
                    minR = min(labeled_minR, inferred_minR if inferred_minR is not None else 999)
                else:
                    minR = inferred_minR

                if maxL is not None:
                    landmark_limits[lname] = (np.min(secs), maxL)

                if minR is not None:
                    landmark_limits[rname] = (minR, np.max(secs))

                # if maxL >= minR:
                #     sys.stderr.write('Left and right labels for %s overlap.\n' % name_u)
                #     # sys.stderr.write('labeled_maxL=%d, inferred_maxL=%d, labeled_minR=%d, inferred_minR=%d\n' %
                #     #                  (labeled_maxL, inferred_maxL, labeled_minR, inferred_minR))
                #
                #     if inferred_maxL < inferred_minR:
                #         maxL = inferred_maxL
                #         minR = inferred_minR
                #         sys.stderr.write('[Resolved] using inferred maxL/minR.\n')
                #     elif labeled_maxL < labeled_minR:
                #         maxL = labeled_maxL
                #         minR = labeled_minR
                #         sys.stderr.write('[Resolved] using labeled maxL/minR.\n')
                #     else:
                #         sys.stderr.write('#### Cannot resolve.. ignored.\n')
                #         continue

            # landmark_limits[lname] = (np.min(secs), maxL)
            # landmark_limits[rname] = (minR, np.max(secs))

            # print 'label:', name_u
            # print 'secs:', secs
            # print 'inferred_maxL:', inferred_maxL, ', labeled_maxL:', labeled_maxL, ', inferred_minR:', inferred_minR, ', labeled_minR:', labeled_minR
            # print '\n'

    return landmark_limits


def get_landmark_range_limits(stack=None, username=None, label_polygons=None, label_section_lookup=None, filtered_labels=None):
    """
    Get a dictionary, whose keys are landmark names and
    values are tuples specifying the first and last sections that have the particular landmarks.

    Parameters
    ----------
    stack : str
    username : str
    label_polygon : pandas.DataFrame, optional
    filtered_labels : list of str, optional
    """

    assert stack is not None or label_polygons is not None

    if label_polygons is None:
        label_polygons = load_label_polygons_if_exists(stack, username)

    mid_sec = (label_polygons.index[0]+label_polygons.index[-1])/2

    landmark_limits = {}

    if filtered_labels is None:
        d = set(label_polygons.keys())
    else:
        d = set(label_polygons.keys()) & set(filtered_labels)

    d_unsided = set(map(convert_name_to_unsided, d))

    for name_u in d_unsided:

        lname = convert_to_left_name(name_u)
        rname = convert_to_right_name(name_u)

        secs = []

        if name_u in label_polygons:
            secs += list(label_polygons[name_u].dropna().keys())

        if lname in label_polygons:
            secs += list(label_polygons[lname].dropna().keys())

        if rname in label_polygons:
            secs += list(label_polygons[rname].dropna().keys())

        secs = np.array(sorted(secs))

        if name_u in singular_structures: # single
            landmark_limits[name_u] = (secs.min(), secs.max())
        else: # two sides

            if len(secs) == 1:
                sys.stderr.write('Structure %s has label on only one section.\n' % name_u)
                sec = secs[0]
                if sec < mid_sec:
                    landmark_limits[lname] = (sec, sec)
                else:
                    landmark_limits[rname] = (sec, sec)
                continue

            elif len(secs) == 0:
                raise
            else:

                diffs = np.diff(secs)
                peak = np.argmax(diffs)

                inferred_maxL = secs[peak]
                inferred_minR = secs[peak+1]

                if lname in label_polygons:
                    labeled_maxL = label_polygons[lname].dropna().keys().max()
                    maxL = max(labeled_maxL, inferred_maxL)
                else:
                    maxL = inferred_maxL

                if rname in label_polygons:
                    labeled_minR = label_polygons[rname].dropna().keys().min()
                    minR = min(labeled_minR, inferred_minR)
                else:
                    minR = inferred_minR

                if maxL >= minR:
                    sys.stderr.write('Left and right labels for %s overlap.\n' % name_u)
                    # sys.stderr.write('labeled_maxL=%d, inferred_maxL=%d, labeled_minR=%d, inferred_minR=%d\n' %
                    #                  (labeled_maxL, inferred_maxL, labeled_minR, inferred_minR))

                    if inferred_maxL < inferred_minR:
                        maxL = inferred_maxL
                        minR = inferred_minR
                        sys.stderr.write('[Resolved] using inferred maxL/minR.\n')
                    elif labeled_maxL < labeled_minR:
                        maxL = labeled_maxL
                        minR = labeled_minR
                        sys.stderr.write('[Resolved] using labeled maxL/minR.\n')
                    else:
                        sys.stderr.write('#### Cannot resolve.. ignored.\n')
                        continue

            landmark_limits[lname] = (secs.min(), maxL)
            landmark_limits[rname] = (minR, secs.max())

            # print 'label:', name_u
            # print 'secs:', secs
            # print 'inferred_maxL:', inferred_maxL, ', labeled_maxL:', labeled_maxL, ', inferred_minR:', inferred_minR, ', labeled_minR:', labeled_minR
            # print '\n'

    return landmark_limits

def generate_annotaion_list(stack, username, filepath=None):

    if filepath is None:
        filepath = os.path.join('/oasis/projects/nsf/csd395/yuncong/CSHL_data_labelings_losslessAlignCropped/',
                    '%(stack)s_%(username)s_latestAnnotationFilenames.txt' % {'stack': stack, 'username': username})

    f = open(filepath, 'w')

    fn_list = []

    for sec in range(first_bs_sec, last_bs_sec + 1):

        dm.set_slice(sec)

        ret = dm.load_review_result_path(username, 'latest', suffix='consolidated')
        if ret is not None:
            fn = ret[0]
            # print fn
            fn_list.append(fn)
            f.write(fn + '\n')

    f.close()
    return fn_list

def get_section_contains_labels(label_polygons, filtered_labels=None):

    section_contains_labels = defaultdict(set)

    if filtered_labels is None:
        labels = label_polygons.keys()
    else:
        labels = label_polygons.keys() & set(filtered_labels)

    for l in labels:
        for s in label_polygons[l].dropna().index:
            section_contains_labels[s].add(l)
    section_contains_labels.default_factory = None

    return section_contains_labels


def load_label_polygons_if_exists(stack, username, output_path=None, output=True, force=False,
                                downsample=None, orientation=None, annotation_rootdir=None,
                                side_assigned=False):
    """
    - assign sideness
    """

    if side_assigned:
        label_polygons_path = os.path.join(annotation_rootdir, '%(stack)s_%(username)s_annotation_polygons_sided.h5' % {'stack': stack, 'username': username})
    else:
        label_polygons_path = os.path.join(annotation_rootdir, '%(stack)s_%(username)s_annotation_polygons.h5' % {'stack': stack, 'username': username})

    if os.path.exists(label_polygons_path) and not force:
        label_polygons = pd.read_hdf(label_polygons_path, 'label_polygons')
    else:
        label_polygons = pd.DataFrame(generate_label_polygons(stack, username=username, orientation=orientation, annotation_rootdir=annotation_rootdir, downsample=downsample))

        if side_assigned:
            label_polygons = assign_sideness(label_polygons)

        if output:
            if output_path is None:
                label_polygons.to_hdf(label_polygons_path, 'label_polygons')
            else:
                label_polygons.to_hdf(output_path, 'label_polygons')

    return label_polygons


def generate_label_polygons(stack, username, orientation=None, downsample=None, timestamp='latest', output_path=None,
                            labels_merge_map={'SolM': 'Sol', 'LC2':'LC', 'Pn2': 'Pn', '7n1':'7n', '7n2':'7n', '7n3':'7n'},
                            annotation_rootdir=None, structure_names=None):
    """Read annotation file, and do the following processing:
    - merge labels
    - remove sideness tag if structure is singular
    """

    # dm = DataManager(stack=stack)

    # if structure_names is None:
    #     structure_names = labelMap_unsidedToSided.keys()

    label_polygons = defaultdict(lambda: {})
    section_bs_begin, section_bs_end = section_range_lookup[stack]

    labelings, usr, ts = DataManager.load_annotation_v2(stack=stack, username=username, orientation=orientation, downsample=downsample,
                                                        timestamp=timestamp, annotation_rootdir=annotation_rootdir)

    for sec in range(section_bs_begin, section_bs_end+1):

        # dm.set_slice(sec)
        if sec not in labelings['polygons']:
            sys.stderr.write('Section %d is not in labelings.\n' % sec)
            continue

        for ann in labelings['polygons'][sec]:
            label = ann['label']

            if label in labels_merge_map:
                label = labels_merge_map[label]

            if 'side' not in ann:
                ann['side'] = None

            if label in singular_structures:
                # assert ann['side'] is None, 'Structure %s is singular, but labeling says it has side property.' % label
                if ann['side'] is not None:
                    sys.stderr.write('Structure %s is singular, but labeling says it has side property... ignore side.\n' % label)
            else:
                if ann['side'] is not None:
                    label = label + '_' + ann['side']

            if label not in all_landmark_names_unsided: # 12N_L -> 12N
                sys.stderr.write('Label %s on Section %d is not recognized.\n' % (label, sec))
                    # continue

            label_polygons[label][sec] = np.array(ann['vertices']).astype(np.int)

    label_polygons.default_factory = None

    return label_polygons


# def generate_label_polygons(stack, username, output_path=None,
#                             labels_merge_map={'SolM': 'Sol', 'LC2':'LC', 'Pn2': 'Pn', '7n1':'7n', '7n2':'7n', '7n3':'7n'},
#                             annotation_rootdir=None):
#     """Convert annotation files to polygon objects.
#     """
#
#     # dm = DataManager(stack=stack)
#     label_polygons = defaultdict(lambda: {})
#     section_bs_begin, section_bs_end = section_range_lookup[stack]
#     for sec in range(section_bs_begin, section_bs_end+1):
#
#         # dm.set_slice(sec)
#         ret = DataManager.load_annotation(stack=stack, section=sec, username=username, timestamp='latest', annotation_rootdir=annotation_rootdir)
#         if ret is None:
#             continue
#         annotations, usr, ts = ret
#
#         # for ann in annotations:
#         for ann in annotations['polygons']:
#             label = ann['label']
#             if label in labels_merge_map:
#                 label = labels_merge_map[label]
#             elif '_' in label and label[:-2] in labels_merge_map: # strip off sideness suffix
#                 label = labels_merge_map[label[:-2]] + label[-2:]
#
#             if label not in labels_unsided + labels_sided: # 12N_L -> 12N
#                 if '_' in label and label[:-2] in labelMap_unsidedToSided and len(labelMap_unsidedToSided[label[:-2]]) == 1:
#                     label = label[:-2]
#                 else:
#                     # raise Exception('Label %s is not recognized.' % label)
#                     sys.stderr.write('Label %s on Section %d is not recognized.\n' % (label, sec))
#                     continue
#
#             label_polygons[label][sec] = np.array(ann['vertices']).astype(np.int)
#
#     label_polygons.default_factory = None
#
#     return label_polygons



from shapely.geometry import Polygon, Point, LinearRing

def closest_to(point, poly):
    pol_ext = LinearRing(poly.exterior.coords)
    d = pol_ext.project(point)
    p = pol_ext.interpolate(d)
    closest_point_coords = list(p.coords)[0]
    return closest_point_coords


def average_multiple_volumes(volumes, bboxes):

    overall_xmin, overall_ymin, overall_zmin = np.min([(xmin, ymin, zmin) for xmin, xmax, ymin, ymax, zmin, zmax in bboxes], axis=0)
    overall_xmax, overall_ymax, overall_zmax = np.max([(xmax, ymax, zmax) for xmin, xmax, ymin, ymax, zmin, zmax in bboxes], axis=0)
    overall_volume = np.zeros((overall_ymax-overall_ymin+1, overall_xmax-overall_xmin+1, overall_zmax-overall_zmin+1), np.bool)

    for (xmin, xmax, ymin, ymax, zmin, zmax), vol in zip(bboxes, volumes):
        overall_volume[ymin - overall_ymin:ymax - overall_ymin+1, \
                        xmin - overall_xmin:xmax - overall_xmin+1, \
                        zmin - overall_zmin:zmax - overall_zmin+1] += vol

    return overall_volume, (overall_xmin, overall_xmax, overall_ymin, overall_ymax, overall_zmin, overall_zmax)

def interpolate_contours_to_volume(contours_grouped_by_pos=None, interpolation_direction=None, contours_xyz=None, return_voxels=False,
                                    return_contours=False, len_interval=20):
    """Interpolate contours

    Returns
    -------
    volume: a 3D binary array
    bbox (tuple): (xmin, xmax, ymin, ymax, zmin, zmax)

    If interpolation_direction == 'z', the points should be (x,y)
    If interpolation_direction == 'x', the points should be (y,z)
    If interpolation_direction == 'y', the points should be (x,z)
    """

    if contours_grouped_by_pos is None:
        assert contours_xyz is not None
        contours_grouped_by_pos = defaultdict(list)
        all_points = np.concatenate(contours_xyz)
        if interpolation_direction == 'z':
            for x,y,z in all_points:
                contours_grouped_by_pos[z].append((x,y))
        elif interpolation_direction == 'y':
            for x,y,z in all_points:
                contours_grouped_by_pos[y].append((x,z))
        elif interpolation_direction == 'x':
            for x,y,z in all_points:
                contours_grouped_by_pos[x].append((y,z))

    else:
        # all_points = np.concatenate(contours_grouped_by_z.values())
        if interpolation_direction == 'z':
            all_points = np.array([(x,y,z) for z, xys in contours_grouped_by_pos.iteritems() for x,y in xys])
        elif interpolation_direction == 'y':
            all_points = np.array([(x,y,z) for y, xzs in contours_grouped_by_pos.iteritems() for x,z in xzs])
        elif interpolation_direction == 'x':
            all_points = np.array([(x,y,z) for x, yzs in contours_grouped_by_pos.iteritems() for y,z in yzs])

    xmin, ymin, zmin = np.floor(all_points.min(axis=0)).astype(np.int)
    xmax, ymax, zmax = np.ceil(all_points.max(axis=0)).astype(np.int)

    interpolated_contours = get_interpolated_contours(contours_grouped_by_pos, len_interval)

    if return_contours:

        # from skimage.draw import polygon_perimeter
        # dense_contour_points = {}
        # for i, contour_pts in interpolated_contours.iteritems():
        #     xs = contour_pts[:,0]
        #     ys = contour_pts[:,1]
        #     dense_contour_points[i] = np.array(polygon_perimeter(ys, xs)).T[:, ::-1]
        # return dense_contour_points

        return {i: contour_pts.astype(np.int) for i, contour_pts in interpolated_contours.iteritems()}

    interpolated_interior_points = {i: points_inside_contour(contour_pts.astype(np.int)) for i, contour_pts in interpolated_contours.iteritems()}
    if return_voxels:
        return interpolated_interior_points

    volume = np.zeros((ymax-ymin+1, xmax-xmin+1, zmax-zmin+1), np.bool)
    for i, pts in interpolated_interior_points.iteritems():
        if interpolation_direction == 'z':
            volume[pts[:,1]-ymin, pts[:,0]-xmin, i-zmin] = 1
        elif interpolation_direction == 'y':
            volume[i-ymin, pts[:,0]-xmin, pts[:,1]-zmin] = 1
        elif interpolation_direction == 'x':
            volume[pts[:,0]-ymin, i-xmin, pts[:,1]-zmin] = 1

    return volume, (xmin,xmax,ymin,ymax,zmin,zmax)


def get_interpolated_contours(contours_grouped_by_pos, len_interval):
    """
    Snap minimum z to minimum int
    Snap maximum z to maximum int
    Return contours at integer levels
    """

    contours_grouped_by_adjusted_pos = {}
    for i, (pos, contour) in enumerate(sorted(contours_grouped_by_pos.iteritems())):
        if i == 0:
            contours_grouped_by_adjusted_pos[int(np.ceil(pos))] = contour
        elif i == len(contours_grouped_by_pos)-1:
            contours_grouped_by_adjusted_pos[int(np.floor(pos))] = contour
        else:
            contours_grouped_by_adjusted_pos[int(np.round(pos))] = contour

    zs = sorted(contours_grouped_by_adjusted_pos.keys())
    n = len(zs)

    interpolated_contours = {}

    for i in range(n):
        z0 = zs[i]
        interpolated_contours[z0] = np.array(contours_grouped_by_adjusted_pos[z0])
        if i + 1 < n:
            z1 = zs[i+1]
            interp_cnts = interpolate_contours(contours_grouped_by_adjusted_pos[z0], contours_grouped_by_adjusted_pos[z1], nlevels=z1-z0+1, len_interval_0=len_interval)
            for zi, z in enumerate(range(z0+1, z1)):
                interpolated_contours[z] = interp_cnts[zi+1]

    return interpolated_contours


def resample_polygon(cnt, n_points=None, len_interval=20):

    polygon = Polygon(cnt)

    if n_points is None:
        contour_length = polygon.exterior.length
        n_points = max(3, int(np.round(contour_length / len_interval)))

    resampled_cnt = np.empty((n_points, 2))
    for i, p in enumerate(np.linspace(0, 1, n_points+1)[:-1]):
        pt = polygon.exterior.interpolate(p, normalized=True)
        resampled_cnt[i] = (pt.x, pt.y)
    return resampled_cnt


from shapely.geometry import Polygon


def signed_curvatures(s, d=7):
    """
    https://www.wikiwand.com/en/Curvature
    Return curvature and x prime, y prime along a curve.
    """

    xp = np.gradient(s[:, 0], d)
    xpp = np.gradient(xp, d)
    yp = np.gradient(s[:, 1], d)
    ypp = np.gradient(yp, d)
    curvatures = (xp * ypp - yp * xpp)/np.sqrt(xp**2+yp**2)**3
    return curvatures, xp, yp

def interpolate_contours(cnt1, cnt2, nlevels, len_interval_0 = 20):
    '''
    Returned arrays include cnt1 and cnt2 - length of array is nlevels.
    '''

    # poly1 = Polygon(cnt1)
    # poly2 = Polygon(cnt2)
    #
    # interpolated_cnts = np.empty((nlevels, len(cnt1), 2))
    # for i, p in enumerate(cnt1):
    #     proj_point = closest_to(Point(p), poly2)
    #     interpolated_cnts[:, i] = (np.column_stack([np.linspace(p[0], proj_point[0], nlevels),
    #                      np.linspace(p[1], proj_point[1], nlevels)]))
    #
    # print cnt1
    # print cnt2

    l1 = Polygon(cnt1).length
    l2 = Polygon(cnt2).length
    n1 = len(cnt1)
    n2 = len(cnt2)
    len_interval_1 = l1 / n1
    len_interval_2 = l2 / n2
    len_interval_interpolated = np.linspace(len_interval_1, len_interval_2, nlevels)

    # len_interval_0 = 20
    n_points = max(int(np.round(max(l1, l2) / len_interval_0)), n1, n2)

    s1 = resample_polygon(cnt1, n_points=n_points)
    s2 = resample_polygon(cnt2, n_points=n_points)

    # Make sure point sets are both clockwise or both anti-clockwise.

    # c1 = np.mean(s1, axis=0)
    # c2 = np.mean(s2, axis=0)
    # d1 = (s1 - c1)[0]
    # d1 = d1 / np.linalg.norm(d1)
    # d2s = s2 - c2
    # d2s = d2s / np.sqrt(np.sum(d2s**2, axis=1))[:,None]
    # s2_start_index = np.argmax(np.dot(d1, d2s.T))
    # print s2_start_index
    # s2 = np.r_[np.atleast_2d(s2[s2_start_index:]), np.atleast_2d(s2[:s2_start_index])]

    # s2i = np.r_[[s2[0]], s2[1:][::-1]]

    s2i = s2[::-1]

    # curv1, xp1, yp1 = signed_curvatures(s1)
    # curv2, xp2, yp2 = signed_curvatures(s2)
    # curv2i, xp2i, yp2i = signed_curvatures(s2i)

    d = 7
    xp1 = np.gradient(s1[:, 0], d)
    yp1 = np.gradient(s1[:, 1], d)
    xp2 = np.gradient(s2[:, 0], d)
    yp2 = np.gradient(s2[:, 1], d)
    xp2i = np.gradient(s2i[:, 0], d)
    yp2i = np.gradient(s2i[:, 1], d)


    # using correlation over curvature values directly is much better than using correlation over signs
    # sign1 = np.sign(curv1)
    # sign2 = np.sign(curv2)
    # sign2i = np.sign(curv2i)

    # conv_curv_1_2 = np.correlate(np.r_[curv2, curv2], curv1, mode='valid')
    conv_xp_1_2 = np.correlate(np.r_[xp2, xp2], xp1, mode='valid')
    conv_yp_1_2 = np.correlate(np.r_[yp2, yp2], yp1, mode='valid')

    # conv_1_2 = np.correlate(np.r_[sign2, sign2], sign1, mode='valid')

    # top, second = conv_1_2.argsort()[::-1][:2]
    # d2_top = (s2 - c2)[top]
    # d2_top = d2_top / np.linalg.norm(d2_top)
    # d2_second = (s2 - c2)[second]
    # d2_second = d2_second / np.linalg.norm(d2_second)
    # s2_start_index = [top, second][np.argmax(np.dot([d2_top, d2_second], d1))]

    # conv_curv_1_2i = np.correlate(np.r_[curv2i, curv2i], curv1, mode='valid')
    conv_xp_1_2i = np.correlate(np.r_[xp2i, xp2i], xp1, mode='valid')
    conv_yp_1_2i = np.correlate(np.r_[yp2i, yp2i], yp1, mode='valid')

    # conv_1_2i = np.correlate(np.r_[sign2i, sign2i], sign1, mode='valid')
    # top, second = conv_1_2i.argsort()[::-1][:2]
    # if xp1[top] * xp2i[top] + yp1[top] * yp2i[top] > xp1[top] * xp2i[top] + yp1[top] * yp2i[top] :
    #     s2i_start_index = top
    # else:
    #     s2i_start_index = second

    # d2_top = (s2i - c2)[top]
    # d2_top = d2_top / np.linalg.norm(d2_top)
    # d2_second = (s2i - c2)[second]
    # d2_second = d2_second / np.linalg.norm(d2_second)
    # s2i_start_index = [top, second][np.argmax(np.dot([d2_top, d2_second], d1))]

    # if conv_1_2[s2_start_index] > conv_1_2i[s2i_start_index]:
    #     s3 = np.r_[np.atleast_2d(s2[s2_start_index:]), np.atleast_2d(s2[:s2_start_index])]
    # else:
    #     s3 = np.r_[np.atleast_2d(s2i[s2i_start_index:]), np.atleast_2d(s2i[:s2i_start_index])]

    # from scipy.spatial import KDTree
    # tree = KDTree(s1)
    # nn_in_order_s2 = np.count_nonzero(np.diff(tree.query(s2)[1]) > 0)
    # nn_in_order_s2i = np.count_nonzero(np.diff(tree.query(s2i)[1]) > 0)

    # overall_s2 = conv_curv_1_2 / conv_curv_1_2.max() + conv_xp_1_2 / conv_xp_1_2.max() + conv_yp_1_2 / conv_yp_1_2.max()
    # overall_s2i = conv_curv_1_2i / conv_curv_1_2i.max() + conv_xp_1_2i / conv_xp_1_2i.max() + conv_yp_1_2i / conv_yp_1_2i.max()

    # overall_s2 =  conv_xp_1_2 / conv_xp_1_2.max() + conv_yp_1_2 / conv_yp_1_2.max()
    # overall_s2i =  conv_xp_1_2i / conv_xp_1_2i.max() + conv_yp_1_2i / conv_yp_1_2i.max()

    overall_s2 =  conv_xp_1_2 + conv_yp_1_2
    overall_s2i =  conv_xp_1_2i + conv_yp_1_2i

    if overall_s2.max() > overall_s2i.max():
        s2_start_index = np.argmax(overall_s2)
        s3 = np.roll(s2, -s2_start_index, axis=0)
    else:
        s2i_start_index = np.argmax(overall_s2i)
        s3 = np.roll(s2i, -s2i_start_index, axis=0)

    # plt.plot(overall)
    # plt.show();

    interpolated_contours = [(1-r) * s1 + r * s3 for r in np.linspace(0, 1, nlevels)]
    resampled_interpolated_contours = [resample_polygon(cnt, len_interval=len_interval_interpolated[i]) for i, cnt in enumerate(interpolated_contours)]

    return resampled_interpolated_contours

def convert_annotation_v3_original_to_aligned(contour_df, stack):

    # with open(thumbnail_data_dir + '/%(stack)s/%(stack)s_sorted_filenames.txt'%dict(stack=stack), 'r') as f:
    #     fn_idx_tuples = [line.strip().split() for line in f.readlines()]
    #     filename_to_section = {fn: int(idx) for fn, idx in fn_idx_tuples}
    filename_to_section, _ = DataManager.load_sorted_filenames(stack)

    # import cPickle as pickle
    # Ts = pickle.load(open(thumbnail_data_dir + '/%(stack)s/%(stack)s_elastix_output/%(stack)s_transformsTo_anchor.pkl' % dict(stack=stack), 'r'))
    Ts = DataManager.load_transforms(stack=stack, downsample_factor=1)

    for cnt_id, cnt in contour_df[(contour_df['orientation'] == 'sagittal') & (contour_df['downsample'] == 1)].iterrows():
        fn = cnt['filename']
        if fn not in filename_to_section:
            continue
        sec = filename_to_section[fn]
        contour_df.loc[cnt_id, 'section'] = sec

        # T = Ts[fn].copy()
        # T[:2, 2] = T[:2, 2]*32
        # Tinv = np.linalg.inv(T)
        Tinv = Ts[fn]

        n = len(cnt['vertices'])

        vertices_on_aligned_cropped = np.dot(Tinv, np.c_[cnt['vertices'], np.ones((n,))].T).T[:, :2]
        contour_df.set_value(cnt_id, 'vertices', vertices_on_aligned_cropped)

        label_position_on_aligned_cropped = np.dot(Tinv, np.r_[cnt['label_position'], 1])[:2]
        contour_df.set_value(cnt_id, 'label_position', label_position_on_aligned_cropped)

    return contour_df

def convert_annotation_v3_original_to_aligned_cropped(contour_df, stack):

    filename_to_section, _ = DataManager.load_sorted_filenames(stack)

    # with open(thumbnail_data_dir + '/%(stack)s/%(stack)s_sorted_filenames.txt'%dict(stack=stack), 'r') as f:
    #     fn_idx_tuples = [line.strip().split() for line in f.readlines()]
    #     filename_to_section = {fn: int(idx) for fn, idx in fn_idx_tuples}
    #     # sorted_filelist = {int(idx): fn for fn, idx in fn_idx_tuples}

    xmin, xmax, ymin, ymax, first_sec, last_sec = DataManager.load_cropbox(stack)

    # with open(thumbnail_data_dir + '/%(stack)s/%(stack)s_cropbox.txt'%dict(stack=stack), 'r') as f:
    #     xmin, xmax, ymin, ymax, first_sec, last_sec = map(int, f.readline().split())

    # import cPickle as pickle
    # Ts = pickle.load(open(thumbnail_data_dir + '/%(stack)s/%(stack)s_elastix_output/%(stack)s_transformsTo_anchor.pkl' % dict(stack=stack), 'r'))
    Ts = DataManager.load_transforms(stack=stack, downsample_factor=1, use_inverse=True)

    for cnt_id, cnt in contour_df[(contour_df['orientation'] == 'sagittal') & (contour_df['downsample'] == 1)].iterrows():
        fn = cnt['filename']
        if fn not in filename_to_section:
            continue
        sec = filename_to_section[fn]
        contour_df.loc[cnt_id, 'section'] = sec

        # T = Ts[fn].copy()
        # T[:2, 2] = T[:2, 2]*32
        # Tinv = np.linalg.inv(T)
        Tinv = Ts[fn]

        n = len(cnt['vertices'])

        vertices_on_aligned_cropped = np.dot(Tinv, np.c_[cnt['vertices'], np.ones((n,))].T).T[:, :2] - (xmin*32, ymin*32)
        # contour_df.loc[cnt_id, 'vertices'] = vertices_on_aligned_cropped
        contour_df.set_value(cnt_id, 'vertices', vertices_on_aligned_cropped)

        label_position_on_aligned_cropped = np.dot(Tinv, np.r_[cnt['label_position'], 1])[:2] - (xmin*32, ymin*32)
        contour_df.set_value(cnt_id, 'label_position', label_position_on_aligned_cropped)
        # contour_df.loc[cnt_id, 'label_position'] = label_position_on_aligned_cropped.tolist()

    return contour_df

def convert_annotation_v3_aligned_cropped_to_original(contour_df, stack):

    # with open(thumbnail_data_dir + '/%(stack)s/%(stack)s_sorted_filenames.txt'%dict(stack=stack), 'r') as f:
    #     fn_idx_tuples = [line.strip().split() for line in f.readlines()]
    #     # filename_to_section = {fn: int(idx) for fn, idx in fn_idx_tuples}
    #     sorted_filelist = {int(idx): fn for fn, idx in fn_idx_tuples}

    filename_to_section, section_to_filename = DataManager.load_sorted_filenames(stack)

    # with open(thumbnail_data_dir + '/%(stack)s/%(stack)s_cropbox.txt'%dict(stack=stack), 'r') as f:
    #     xmin, xmax, ymin, ymax, first_sec, last_sec = map(int, f.readline().split())

    xmin, xmax, ymin, ymax, first_sec, last_sec = DataManager.load_cropbox(stack)

    # import cPickle as pickle
    # Ts = pickle.load(open(thumbnail_data_dir + '/%(stack)s/%(stack)s_elastix_output/%(stack)s_transformsTo_anchor.pkl' % dict(stack=stack), 'r'))

    Ts = DataManager.load_transforms(stack=stack, downsample_factor=1)

    for cnt_id, cnt in contour_df[(contour_df['orientation'] == 'sagittal') & (contour_df['downsample'] == 1)].iterrows():
        sec = cnt['section']
        fn = section_to_filename[sec]
        if fn in ['Placeholder', 'Nonexisting', 'Rescan']:
            continue
        contour_df.loc[cnt_id, 'filename'] = fn

        # T = Ts[fn].copy()
        # T[:2, 2] = T[:2, 2]*32

        T = np.linalg.inv(Ts[fn])

        n = len(cnt['vertices'])

        vertices_on_aligned = np.array(cnt['vertices']) + (xmin*32, ymin*32)
        contour_df.set_value(cnt_id, 'vertices', np.dot(T, np.c_[vertices_on_aligned, np.ones((n,))].T).T[:, :2])

        label_position_on_aligned = np.array(cnt['label_position']) + (xmin*32, ymin*32)
        contour_df.set_value(cnt_id, 'label_position', np.dot(T, np.r_[label_position_on_aligned, 1])[:2])

    return contour_df
