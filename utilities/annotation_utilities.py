import numpy as np

import sys
import os

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from metadata import *
from data_manager import *

import pandas as pd

from collections import defaultdict

from skimage.measure import grid_points_in_poly

def fill_sparse_volume(volume_sparse):

    from registration_utilities import find_contour_points

    volume = np.zeros_like(volume_sparse, np.int8)

    for z in range(volume_sparse.shape[-1]):
        for ind, cnts in find_contour_points(volume_sparse[..., z]).iteritems():
            cnt = cnts[np.argsort(map(len, cnts))[-1]]
            pts = points_inside_contour(cnt)
            volume[pts[:,1], pts[:,0], z] = ind
    return volume

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
        if name in labels_unsided and len(labelMap_unsidedToSided[name]) == 2:
            for sec, coords in v.iteritems():

                if np.any(np.isnan(coords)): continue

                if sec <= landmark_range_limits[name + '_L'][1]:
                    label_polygons_sideAssigned_dict[name + '_L'][sec] = coords
                elif sec >= landmark_range_limits[name + '_R'][0]:
                    label_polygons_sideAssigned_dict[name + '_R'][sec] = coords
                else:
                    print name, sec, landmark_range_limits[name + '_L'], landmark_range_limits[name + '_R']
                    raise Exception('label_polygon contains %s beyond range limits.' % name)

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


def get_landmark_range_limits(stack=None, username=None, label_polygons=None, filtered_labels=None):
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

    d_unsided = set([labelMap_sidedToUnsided[ll] if ll in labelMap_sidedToUnsided else ll for ll in d])

    for name_u in d_unsided:

        secs = []
        if name_u in label_polygons:
            secs += list(label_polygons[name_u].dropna().keys())
        if name_u + '_L' in label_polygons:
            secs += list(label_polygons[name_u + '_L'].dropna().keys())
        if name_u + '_R' in label_polygons:
            secs += list(label_polygons[name_u + '_R'].dropna().keys())

        secs = np.array(sorted(secs))

        if len(secs) < 2:
            continue

        if len(labelMap_unsidedToSided[name_u]) == 1: # single
            landmark_limits[name_u] = (secs.min(), secs.max())
        else: # two sides
            diffs = np.diff(secs)
            peak = np.argmax(diffs)

            inferred_maxL = secs[peak]
            inferred_minR = secs[peak+1]

            if name_u + '_L' in label_polygons:
                labeled_maxL = label_polygons[name_u + '_L'].dropna().keys().max()
                maxL = max(labeled_maxL, inferred_maxL)
            else:
                maxL = inferred_maxL

            if name_u + '_R' in label_polygons:
                labeled_minR = label_polygons[name_u + '_R'].dropna().keys().min()
                minR = min(labeled_minR, inferred_minR)
            else:
                minR = inferred_minR

            if maxL >= minR:
                sys.stderr.write('Left and right labels for %s overlap.\n' % name_u)
                continue

            landmark_limits[name_u + '_L'] = (secs.min(), maxL)
            landmark_limits[name_u + '_R'] = (minR, secs.max())

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


def load_label_polygons_if_exists(stack, username, output_path=None, force=False, annotation_rootdir=annotation_midbrainIncluded_rootdir):

    label_polygons_path = os.path.join(annotation_rootdir, '%(stack)s_%(username)s_annotation_polygons.h5' % {'stack': stack, 'username': username})

    if os.path.exists(label_polygons_path) and not force:
        label_polygons = pd.read_hdf(label_polygons_path, 'label_polygons')
    else:
        label_polygons = pd.DataFrame(generate_label_polygons(stack, username, annotation_rootdir=annotation_rootdir))

        if output_path is None:
            label_polygons.to_hdf(label_polygons_path, 'label_polygons')
        else:
            label_polygons.to_hdf(output_path, 'label_polygons')

    return label_polygons


def generate_label_polygons(stack, username, output_path=None,
                            labels_merge_map={'SolM': 'Sol', 'LC2':'LC', 'Pn2': 'Pn', '7n1':'7n', '7n2':'7n', '7n3':'7n'},
                            annotation_rootdir=annotation_midbrainIncluded_rootdir):
    """Convert annotation files to polygon objects.
    """

    # dm = DataManager(stack=stack)
    label_polygons = defaultdict(lambda: {})
    section_bs_begin, section_bs_end = section_range_lookup[stack]
    for sec in range(section_bs_begin, section_bs_end+1):

        # dm.set_slice(sec)
        ret = DataManager.load_annotation(stack=stack, section=sec, username=username, timestamp='latest', annotation_rootdir=annotation_rootdir)

        if ret is None:
            continue

        annotations, usr, ts = ret

        for ann in annotations:
            label = ann['label']
            if label in labels_merge_map:
                label = labels_merge_map[label]
            elif '_' in label and label[:-2] in labels_merge_map: # strip off sideness suffix
                label = labels_merge_map[label[:-2]] + label[-2:]

            if label not in labels_unsided + labels_sided: # 12N_L -> 12N
                if '_' in label and label[:-2] in labelMap_unsidedToSided and len(labelMap_unsidedToSided[label[:-2]]) == 1:
                    label = label[:-2]
                else:
                    # raise Exception('Label %s is not recognized.' % label)
                    sys.stderr.write('Label %s on Section %d is not recognized.\n' % (label, sec))
                    continue

            label_polygons[label][sec] = np.array(ann['vertices']).astype(np.int)

    label_polygons.default_factory = None

    return label_polygons



from shapely.geometry import Polygon, Point, LinearRing

def closest_to(point, poly):
    pol_ext = LinearRing(poly.exterior.coords)
    d = pol_ext.project(point)
    p = pol_ext.interpolate(d)
    closest_point_coords = list(p.coords)[0]
    return closest_point_coords


def average_multiple_volumes(volumes):
    return np.maximum(*volumes)

def interpolate_contours_to_volume(contours_grouped_by_pos=None, interpolation_direction=None, contours_xyz=None):
    """Interpolate contours

    Returns
    -------
    volume: a 3D binary array
    bbox (tuple): (xmin, xmax, ymin, ymax, zmin, zmax)

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

    interpolated_contours = {}

    zs = sorted(contours_grouped_by_pos.keys())

    for i in range(len(zs)):
        z0 = zs[i]
        interpolated_contours[z0] = np.array(contours_grouped_by_pos[z0])
        if i + 1 < len(zs):
            z1 = zs[i+1]
            interp_cnts = interpolate_contours(contours_grouped_by_pos[z0], contours_grouped_by_pos[z0], z1-z0+1)
            for zi, z in enumerate(range(z0+1, z1)):
                interpolated_contours[z] = interp_cnts[zi+1]

    interpolated_interior_points = {i: points_inside_contour(contour_pts.astype(np.int)) for i, contour_pts in interpolated_contours.iteritems()}

    volume = np.zeros((ymax-ymin+1, xmax-xmin+1, zmax-zmin+1), np.bool)
    for i, pts in interpolated_interior_points.iteritems():
        if interpolation_direction == 'z':
            volume[pts[:,1]-ymin, pts[:,0]-xmin, i-zmin] = 1
        elif interpolation_direction == 'y':
            volume[i, pts[:,0]-xmin, pts[:,1]-zmin] = 1
        elif interpolation_direction == 'x':
            volume[pts[:,0]-ymin, i, pts[:,1]-zmin] = 1

    return volume, (xmin,xmax,ymin,ymax,zmin,zmax)


def interpolate_contours(cnt1, cnt2, nlevels):
    '''
    returned arrays include cnt1 and cnt2
    '''

    poly1 = Polygon(cnt1)
    poly2 = Polygon(cnt2)

    interpolated_cnts = np.empty((nlevels, len(cnt1), 2))
    for i, p in enumerate(cnt1):
        proj_point = closest_to(Point(p), poly2)
        interpolated_cnts[:, i] = (np.column_stack([np.linspace(p[0], proj_point[0], nlevels),
                         np.linspace(p[1], proj_point[1], nlevels)]))

    return interpolated_cnts
