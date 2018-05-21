"""Functions related to registration."""

import numpy as np
import sys
import os
import time

from skimage.morphology import binary_closing, disk, binary_dilation, binary_erosion
from skimage.measure import grid_points_in_poly, subdivide_polygon, approximate_polygon
from skimage.measure import find_contours, regionprops
from shapely.geometry import Polygon
try:
    import cv2
except:
    sys.stderr.write('Cannot find cv2\n')
import matplotlib.pyplot as plt
from multiprocess import Pool

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from distributed_utilities import download_from_s3
from metadata import *
from lie import matrix_exp_v

def plot_alignment_results(traj, scores, select_best):

    if select_best == 'last_value':
        best_score = scores[-1]
        best_param = traj[-1]
    elif select_best == 'max_value':
        best_score = np.max(scores)
        best_param = traj[np.argmax(scores)]
    else:
        raise

    print 'Best parameters ='
    print best_param.reshape((3,4))
    print 'Best score =', best_score, ", initial score =", scores[0]

    Ts = np.array(traj)

    plt.plot(Ts[:, [0,5,10]]);
    plt.title('rotational params');
    plt.xlabel('Iteration');
    plt.show();

    plt.plot(Ts[:, [1,2,4,6,8,9]]);
    plt.title('rotational params');
    plt.xlabel('Iteration');
    plt.show();

    plt.plot(Ts[:, [3,7,11]]);
    plt.title('translation params');
    plt.xlabel('Iteration');
    plt.show();

    plt.figure();
    plt.plot(scores);
    plt.title('Score');
    plt.show();

def parallel_where_binary(binary_volume, num_samples=None):
    """
    Returns:
        (n,3)-ndarray
    """

    w = np.where(binary_volume)

    if num_samples is not None:
        n = len(w[0])
        sample_indices = np.random.choice(range(n), min(num_samples, n), replace=False)
        return np.c_[w[1][sample_indices].astype(np.int16),
                     w[0][sample_indices].astype(np.int16),
                     w[2][sample_indices].astype(np.int16)]
    else:
        return np.c_[w[1].astype(np.int16), w[0].astype(np.int16), w[2].astype(np.int16)]


def parallel_where(atlas_volume, label_ind, num_samples=None):

    w = np.where(atlas_volume == label_ind)

    if num_samples is not None:
        n = len(w[0])
        sample_indices = np.random.choice(range(n), min(num_samples, n), replace=False)
        return np.c_[w[1][sample_indices].astype(np.int16),
                     w[0][sample_indices].astype(np.int16),
                     w[2][sample_indices].astype(np.int16)]
    else:
        return np.c_[w[1].astype(np.int16), w[0].astype(np.int16), w[2].astype(np.int16)]

def affine_components_to_vector(tx=0,ty=0,tz=0,theta_xy=0,theta_xz=0,theta_yz=0,c=(0,0,0)):
    """
    y = R(x-c)+t+c.

    Args:
        theta_xy (float): in radian.
    Returns:
        (12,)-ndarray:
    """
    # assert np.count_nonzero([theta_xy, theta_yz, theta_xz]) <= 1, \
    # "Current implementation is sound only if only one rotation is given."

    cos_theta_xy = np.cos(theta_xy)
    sin_theta_xy = np.sin(theta_xy)
    cos_theta_yz = np.cos(theta_yz)
    sin_theta_yz = np.sin(theta_yz)
    cos_theta_xz = np.cos(theta_xz)
    sin_theta_xz = np.sin(theta_xz)
    Rz = np.array([[cos_theta_xy, -sin_theta_xy, 0], [sin_theta_xy, cos_theta_xy, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, cos_theta_yz, -sin_theta_yz], [0, sin_theta_yz, cos_theta_yz]])
    Ry = np.array([[cos_theta_xz, 0, -sin_theta_xz], [0, 1, 0], [sin_theta_xz, 0, cos_theta_xz]])
    R = np.dot(Rx, np.dot(Ry, Rz))
    tt = np.r_[tx,ty,tz] + c - np.dot(R,c)
    return np.ravel(np.c_[R, tt])

def rotate_transform_vector(v, theta_xy=0,theta_yz=0,theta_xz=0,c=(0,0,0)):
    """
    v is 12-length parameter.
    """
    cos_theta_z = np.cos(theta_xy)
    sin_theta_z = np.sin(theta_xy)
    Rz = np.array([[cos_theta_z, -sin_theta_z, 0], [sin_theta_z, cos_theta_z, 0], [0, 0, 1]])
    cos_theta_x = np.cos(theta_yz)
    sin_theta_x = np.sin(theta_yz)
    Rx = np.array([[1, 0, 0], [0, cos_theta_x, -sin_theta_x], [0, sin_theta_x, cos_theta_x]])
    cos_theta_y = np.cos(theta_xz)
    sin_theta_y = np.sin(theta_xz)
    Ry = np.array([[cos_theta_y, 0, -sin_theta_y], [0, 1, 0], [sin_theta_y, 0, cos_theta_y]])

    R = np.zeros((3,3))
    R[0, :3] = v[:3]
    R[1, :3] = v[4:7]
    R[2, :3] = v[8:11]
    t = v[[3,7,11]]
    R_new = np.dot(Rx, np.dot(Ry, np.dot(Rz, R)))
    t_new = t + c - np.dot(R_new, c)
    return np.ravel(np.c_[R_new, t_new])


def compute_bspline_cp_contribution_to_test_pts(control_points, test_points):
    """
    Args:
        control_points (1d-array): normalized in the unit of spacing interval
        test_points (1d-array): normalized in the unit of spacing interval
    """

    test_points_x_normalized = test_points
    ctrl_point_x_normalized = control_points

    D = np.subtract.outer(test_points_x_normalized, ctrl_point_x_normalized) # (#testpts, #ctrlpts)

    in_1 = ((D >= 0) & (D < 1)).astype(np.int)
    in_2 = ((D >= 1) & (D < 2)).astype(np.int)
    in_3 = ((D >= 2) & (D < 3)).astype(np.int)
    in_4 = ((D >= 3) & (D < 4)).astype(np.int)
    F = in_1 * D**3/6. + \
    in_2 * (D**2*(2-D)/6. + D*(3-D)*(D-1)/6. + (4-D)*(D-1)**2/6.) + \
    in_3 * (D*(3-D)**2/6. + (4-D)*(3-D)*(D-1)/6. + (4-D)**2*(D-2)/6.) + \
    in_4 * (4-D)**3/6.

    return F.T # (#ctrl, #test)

# def bspline_N(i, t):
#     """
#     Cubic B-spline base functions.

#     Args:
#         i (int): control point index. Can be negative (?)
#         t (float): position.
#     """

#     d = t - i
#     in_1 = ((d >= 0) & (d < 1)).astype(np.int)
#     in_2 = ((d >= 1) & (d < 2)).astype(np.int)
#     in_3 = ((d >= 2) & (d < 3)).astype(np.int)
#     in_4 = ((d >= 3) & (d < 4)).astype(np.int)
#     return \
#     in_1 * d**3/6. + \
#     in_2 * (d**2*(2-d)/6. + d*(3-d)*(d-1)/6. + (4-d)*(d-1)**2/6.) + \
#     in_3 * (d*(3-d)**2/6. + (4-d)*(3-d)*(d-1)/6. + (4-d)**2*(d-2)/6.) + \
#     in_4 * (4-d)**3/6.

# def N(i, t):
#     """
#     Cubic B-spline base functions.

#     Args:
#         i (int): control point index. Can be negative (?)
#         t (float): position.
#     """
#     d = t - i

#     if d >= 0 and d < 1:
#         return d**3/6.
#     elif d >= 1 and d < 2:
#         return d**2*(2-d)/6. + d*(3-d)*(d-1)/6. + (4-d)*(d-1)**2/6.
#     elif d >= 2 and d < 3:
#         return d*(3-d)**2/6. + (4-d)*(3-d)*(d-1)/6. + (4-d)**2*(d-2)/6.
#     elif d >= 3 and d < 4:
#         return (4-d)**3/6.
#     else:
#         return 0

    # if i <= t and t < i+1:
    #     return (t-i)**3/6.
    # elif i+1 <= t and t < i+2:
    #     return (t-i)**2*(i+2-t)/6. + (t-i)*(i+3-t)*(t-i-1)/6. + (i+4-t)*(t-i-1)**2/6.
    # elif i+2 <= t and t < i+3:
    #     return (t-i)*(i+3-t)**2/6. + (i+4-t)*(i+3-t)*(t-i-1)/6. + (i+4-t)**2*(t-i-2)/6.
    # elif i+3 <= t and t < i+4:
    #     return (i+4-t)**3/6.
    # else:
    #     return 0


# def N(i,t):
#     """
#     Cubic B-spline base functions.

#     Args:
#         i (int): control point index. Can be negative (?)
#         t (float): position.
#     """

#     if i <= t and t < i+1:
#         return (t-i)**3/6.
#     elif i+1 <= t and t < i+2:
#         return (t-i)**2*(i+2-t)/6. + (t-i)*(i+3-t)*(t-i-1)/6. + (i+4-t)*(t-i-1)**2/6.
#     elif i+2 <= t and t < i+3:
#         return (t-i)*(i+3-t)**2/6. + (i+4-t)*(i+3-t)*(t-i-1)/6. + (i+4-t)**2*(t-i-2)/6.
#     elif i+3 <= t and t < i+4:
#         return (i+4-t)**3/6.
#     else:
#         return 0

#########################################################################

from scipy.ndimage.interpolation import zoom

def generate_aligner_parameters_v2(alignment_spec,
                                   structures_m=all_known_structures_sided,
                                   structures_f=None,
                                  fixed_structures_are_sided=False,
                                  fixed_surroundings_have_positive_value=False,
                                  fixed_use_surround=False):
    """
    Args:
        alignment_spec (dict):
        
        fixed_structures_are_sided (bool):
        fixed_surroundings_have_positive_value (bool): if true, fixed surroundings are represented by separate structures (in the case of Neurolucida annotations). If False, fixed brain only has score volumes whose values are existence probabilities of certain structures.
        fixed_use_surround (bool): whether fixed structures include surround.

    Returns:
        - 'volume_moving': dict {ind_m: 3d array},
        - 'volume_fixed': dict {ind_m: 3d array},
        - 'structure_to_label_moving': dict {str: int},
        - 'label_to_structure_moving': dict {int: str},
        - 'structure_to_label_fixed': dict {str: int},
        - 'label_to_structure_fixed': dict {int: str},
        - 'label_weights_m': dict {int: float},
        - 'label_mapping_m2f': dict {int: int},
    """

    stack_m_spec = alignment_spec['stack_m']
    stack_m = stack_m_spec['name']
    vol_type_m = stack_m_spec['vol_type']
    if 'structure' in stack_m_spec:
        structure_m = stack_m_spec['structure']
    if 'detector_id' in stack_m_spec:
        detector_id_m = stack_m_spec['detector_id']
    if 'prep_id' in stack_m_spec:
        prep_id_m = stack_m_spec['prep_id']
    resolution_m = stack_m_spec['resolution']

    stack_f_spec = alignment_spec['stack_f']
    stack_f = stack_f_spec['name']
    vol_type_f = stack_f_spec['vol_type']
    if 'structure' in stack_m_spec:
        structure_f = stack_f_spec['structure']
    if 'detector_id' in stack_m_spec:
        detector_id_f = stack_f_spec['detector_id']
    if 'prep_id' in stack_m_spec:
        prep_id_f = stack_f_spec['prep_id']
    resolution_f = stack_f_spec['resolution']

    warp_setting = alignment_spec['warp_setting']

    registration_settings = read_csv(REGISTRATION_SETTINGS_CSV, header=0, index_col=0)
    warp_properties = registration_settings.loc[warp_setting]
    print warp_properties

    ################################################################

    upstream_warp_setting = warp_properties['upstream_warp_id']
    if upstream_warp_setting == 'None':
        upstream_warp_setting = None
    transform_type = warp_properties['transform_type']
    grad_computation_sample_number = int(warp_properties['grad_computation_sample_number'])

    surround_weight = warp_properties['surround_weight']
    if isinstance(surround_weight, float) or isinstance(surround_weight, int):
        surround_weight = float(surround_weight)
        include_surround = surround_weight != 0 and not np.isnan(surround_weight)
    elif isinstance(surround_weight, str):
        surround_weight = str(surround_weight)
        # Setting surround_weight as inverse is very important. Using -1 often gives false peaks.
        include_surround = True

    print 'surround', surround_weight, include_surround

    positive_weight = 'size'
    # positive_weight = 'inverse'

    ############################################################################

    if include_surround:
        structures_m = set(structures_m) | set([convert_to_surround_name(s, margin='200um') for s in structures_m])

    if upstream_warp_setting is None:

        if stack_m_spec['name'].startswith('atlas'):
            # in_bbox_wrt='atlasSpace'
            in_bbox_wrt='canonicalAtlasSpace'
        else:
            in_bbox_wrt='wholebrain'

        volume_moving, structure_to_label_moving, label_to_structure_moving = \
        DataManager.load_original_volume_all_known_structures_v3(stack_spec=stack_m_spec,
                                                                 sided=True,
                                                      include_surround=include_surround,
                                                        return_label_mappings=True,
                                                         name_or_index_as_key='index',
                                                         common_shape=False,
                                                        structures=structures_m,
                                                         in_bbox_wrt=in_bbox_wrt,
                                                                 return_origin_instead_of_bbox=True
                                                                 # in_bbox_wrt='wholebrain',
                                                            # out_bbox_wrt='atlasSpace'
                                                        )
    else:

        initial_alignment_spec = alignment_spec['initial_alignment_spec']
        print initial_alignment_spec

        volume_moving, structure_to_label_moving, label_to_structure_moving = \
        DataManager.load_transformed_volume_all_known_structures_v3(alignment_spec=initial_alignment_spec,
                                                                    resolution=resolution_m,
                                                            structures=structures_m,
                                                            sided=True,
                                                            return_label_mappings=True,
                                                            name_or_index_as_key='index',
                                                            common_shape=False,
                                                           return_origin_instead_of_bbox=True,
#                                                                     in_bbox_wrt='wholebrain',
#                                                                     out_bbox_wrt='wholebrain'
                                                                )

    if len(volume_moving) == 0:
        sys.stderr.write("No moving volumes.\n")
    else:
        sys.stderr.write("Loaded moving volumes: %s.\n" % sorted(structure_to_label_moving.keys()))

    #############################################################################

    if structures_f is None:
        if fixed_structures_are_sided:
            structures_f = set([s for s in structures_m])
        else:
            structures_f = set([convert_to_original_name(s) for s in structures_m])

    if not fixed_use_surround:
        structures_f = [s for s in structures_f if not is_surround_label(s)]

    # if stack_f_spec['name'] in ['MD589', 'MD585', 'MD594', 'LM27', 'LM17']:
    in_bbox_wrt = 'wholebrain'
    # else:
    #     in_bbox_wrt = 'wholebrainXYcropped'

    volume_fixed, structure_to_label_fixed, label_to_structure_fixed = \
    DataManager.load_original_volume_all_known_structures_v3(stack_spec=stack_f_spec,
                                in_bbox_wrt=in_bbox_wrt,
                                                     # out_bbox_wrt='wholebrain',
                                                             structures=structures_f,
                                                    # sided=False,
                                                    # include_surround=include_surround,
                                                    include_surround=include_surround if fixed_use_surround else False,
                                                     return_label_mappings=True,
                                                     name_or_index_as_key='index',
                                                     common_shape=False,
                                                            return_origin_instead_of_bbox=True)

    if len(volume_fixed) == 0:
        sys.stderr.write("No fixed volumes.\n")
    else:
        sys.stderr.write("Loaded fixed volumes: %s.\n" % sorted(structure_to_label_fixed.keys()))

    ############################################################################

    # Make two volumes the same resolution.

    voxel_size_m = convert_resolution_string_to_voxel_size(resolution=resolution_m, stack=stack_m_spec['name'])
    print "voxel size for moving = %.2f um" % voxel_size_m
    voxel_size_f = convert_resolution_string_to_voxel_size(resolution=resolution_f, stack=stack_f_spec['name'])
    print "voxel size for fixed = %.2f um" % voxel_size_f
    ratio_m_to_f = voxel_size_m / voxel_size_f
    if ratio_m_to_f < 1:
        print 'Moving volume voxel size (%.2f um) is smaller than fixed volume (%.2f um); downsample moving volume to %.2f um.' % (voxel_size_m, voxel_size_f, voxel_size_f)
        # float16 is not supported by zoom()
        volume_moving = {k: (rescale_by_resampling(v, ratio_m_to_f), o * ratio_m_to_f) for k, (v, o) in volume_moving.iteritems()}
        unified_resolution = voxel_size_f
    elif ratio_m_to_f > 1:
        print 'Fixed volume voxel size (%.2f um) is smaller than moving volume (%.2f um); downsample fixed volume to %.2f um.' % (voxel_size_f, voxel_size_m, voxel_size_m)
        volume_fixed = {k: (rescale_by_resampling(v, 1./ratio_m_to_f), o / float(ratio_m_to_f)) for k, (v, o) in volume_fixed.iteritems()}
        unified_resolution = voxel_size_m
    else:
        unified_resolution = voxel_size_m

    #############################################################################

    structure_subset_m = all_known_structures_sided

    if include_surround:
        structure_subset_m = structure_subset_m + [convert_to_surround_name(s, margin='200um') for s in structure_subset_m]

    if any(map(is_sided_label, structures_f)): # fixed volumes have structures of both sides.
        
        if include_surround:
            label_mapping_m2f = {label_m: structure_to_label_fixed[convert_to_nonsurround_name(name_m)]
             for label_m, name_m in label_to_structure_moving.iteritems()
             if name_m in structure_subset_m and convert_to_nonsurround_name(name_m) in structure_to_label_fixed}

        else:
            label_mapping_m2f = {label_m: structure_to_label_fixed[convert_to_nonsurround_name(name_m)]
                     for label_m, name_m in label_to_structure_moving.iteritems()
                    if name_m in structure_subset_m and convert_to_nonsurround_name(name_m) in structure_to_label_fixed}
    else:
        label_mapping_m2f = {label_m: structure_to_label_fixed[convert_to_original_name(name_m)]
                     for label_m, name_m in label_to_structure_moving.iteritems()
                     if name_m in structure_subset_m and convert_to_original_name(name_m) in structure_to_label_fixed}

    print 'label_mapping_m2f', label_mapping_m2f

    if include_surround:
        if positive_weight == 'inverse' or surround_weight == 'inverse':
            t = time.time()
            cutoff = .5 # Structure size is defined as the number of voxels whose value is above this cutoff probability.
        #     pool = Pool(NUM_CORES)
        #     volume_moving_structure_sizes = dict(zip(volume_moving.keys(),
        #                                              pool.map(lambda l: np.count_nonzero(volume_moving[l] > cutoff),
        #                                                       volume_moving.keys())))
        #     pool.close()
        #     pool.join()
            volume_moving_structure_sizes = {k: np.count_nonzero(v > cutoff) for k, (v, o) in volume_moving.iteritems()}
            sys.stderr.write("Computing structure sizes: %.2f s\n" % (time.time() - t))

    label_weights_m = {}

    for label_m in label_mapping_m2f.iterkeys():
        name_m = label_to_structure_moving[label_m]
        if not is_surround_label(name_m):
            if positive_weight == 'size':
                label_weights_m[label_m] = 1.
            elif positive_weight == 'inverse':
                p = np.percentile(volume_moving_structure_sizes.values(), 50)
                label_weights_m[label_m] =  np.minimum(p / volume_moving_structure_sizes[label_m], 1.)
            else:
                sys.stderr.write("positive_weight %s is not recognized. Using the default.\n" % positive_weight)

    for label_m in label_mapping_m2f.iterkeys():
        name_m = label_to_structure_moving[label_m]
        if is_surround_label(name_m):
            label_ns = structure_to_label_moving[convert_to_nonsurround_name(name_m)]
            if surround_weight == 'inverse':
                if fixed_surroundings_have_positive_value:
                    # for Neurolucida annotation data
                    # fixed contains 7N, 7N_surround as separate map, each has positive values.
                    label_weights_m[label_m] = label_weights_m[label_ns] * volume_moving_structure_sizes[label_ns] / float(volume_moving_structure_sizes[label_m])
                else:
                    # fixed brain has only 7N prob. map
                    label_weights_m[label_m] = - label_weights_m[label_ns] * volume_moving_structure_sizes[label_ns] / float(volume_moving_structure_sizes[label_m])
            elif isinstance(surround_weight, int) or isinstance(surround_weight, float):
                if fixed_surroundings_have_positive_value:
                    label_weights_m[label_m] = surround_weight
                else:
                    label_weights_m[label_m] = - surround_weight
            else:
                sys.stderr.write("surround_weight %s is not recognized. Using the default.\n" % surround_weight)

    print label_weights_m

    ######################################################

    alinger_parameters = \
    {'label_weights_m': label_weights_m,
     'label_mapping_m2f': label_mapping_m2f,
     'volume_moving': volume_moving,
     'volume_fixed': volume_fixed,
     'structure_to_label_moving': structure_to_label_moving,
     'label_to_structure_moving': label_to_structure_moving,
     'structure_to_label_fixed': structure_to_label_fixed,
     'label_to_structure_fixed': label_to_structure_fixed,
     'transform_type': transform_type,
     'grad_computation_sample_number': grad_computation_sample_number,
     'resolution_um': unified_resolution
    }

    return alinger_parameters

# def compute_gradient(volumes, smooth_first=False):
#     """
#     Args:
#         volumes (dict {int: 3d-array}): dict of volumes
#         smooth_first (bool): If true, smooth each volume before computing gradients.
#         This is useful if volume is binary and gradients are only nonzero at structure borders.

#     Note:
#         # 3.3 second - re-computing is much faster than loading
#         # .astype(np.float32) is important;
#         # Otherwise the score volume is type np.float16, np.gradient requires np.float32 and will have to convert which is very slow
#         # 2s (float32) vs. 20s (float16)
#     """
#     gradients = {}

#     for ind, v in volumes.iteritems():
#         print "Computing gradient for", ind

#         t1 = time.time()

#         gradients[ind] = np.zeros((3,) + v.shape)

#         # t = time.time()
#         cropped_v, (xmin,xmax,ymin,ymax,zmin,zmax) = crop_volume_to_minimal(v, margin=5, return_origin_instead_of_bbox=False)
#         # sys.stderr.write("Crop: %.2f seconds.\n" % (time.time()-t))

#         if smooth_first:
#             # t = time.time()
#             cropped_v = gaussian(cropped_v, 3)
#             # sys.stderr.write("Smooth: %.2f seconds.\n" % (time.time()-t))

#         # t = time.time()
#         cropped_v_gy_gx_gz = np.gradient(cropped_v.astype(np.float32), 3, 3, 3)
#         # sys.stderr.write("Compute gradient: %.2f seconds.\n" % (time.time()-t))

#         gradients[ind][0][ymin:ymax+1, xmin:xmax+1, zmin:zmax+1] = cropped_v_gy_gx_gz[1]
#         gradients[ind][1][ymin:ymax+1, xmin:xmax+1, zmin:zmax+1] = cropped_v_gy_gx_gz[0]
#         gradients[ind][2][ymin:ymax+1, xmin:xmax+1, zmin:zmax+1] = cropped_v_gy_gx_gz[2]

#         # sys.stderr.write("Overall: %.2f seconds.\n" % (time.time()-t1))

#     return gradients

# def transform_parameters_to_init_T_v2(global_transform_parameters, local_aligner_parameters, centroid_m, centroid_f):
#     """
#     `init_T` is on top of shifting moving volume by `centroid_m` and shifting fixed volume by `centroid_f`.
#
#     Returns:
#         (3,4)-array:
#     """
#
#     T = alignment_parameters_to_transform_matrix_v2(global_transform_parameters)
#     R = T[:3,:3]
#     t = T[:3,3]
#
#     # lof = local_aligner_parameters['volume_fixed_origin_wrt_wholebrain']
#     # lom = local_aligner_parameters['volume_moving_origin_wrt_wholebrain']
#
#     init_T = np.column_stack([R, np.dot(R, centroid_m) + t - centroid_f])
#     return init_T

# def transform_parameters_relative_to_initial_shift(transform_parameters, centroid_m, centroid_f):
#     """
#     Returns:
#         (3,4)-array:
#     """
#
#     T = alignment_parameters_to_transform_matrix_v2(transform_parameters)
#     R = T[:3,:3]
#     t_composite = T[:3,3]
#
#     t = t_composite - centroid_f + np.dot(R, centroid_m)
#     return np.column_stack([R, t])
#
# from scipy.optimize import approx_fprime

def hessian(x0, f, epsilon=1.e-5, linear_approx=False, *args):
    """
    A numerical approximation to the Hessian matrix of cost function at
    location x0 (hopefully, the minimum)

    Args:
        x0: point
        f: cost function
    """

    # ``calculate_cost_function`` is the cost function implementation
    # The next line calculates an approximation to the first
    # derivative
    f1 = approx_fprime( x0, f, epsilon, *args)

    # This is a linear approximation. Obviously much more efficient
    # if cost function is linear
    if linear_approx:
        f1 = np.matrix(f1)
        return f1.transpose() * f1
    # Allocate space for the hessian
    n = x0.shape[0]
    hessian = np.zeros ( ( n, n ) )
    # The next loop fill in the matrix
    xx = x0
    for j in xrange( n ):
        xx0 = xx[j] # Store old value
        xx[j] = xx0 + epsilon[j] # Perturb with finite difference
        # Recalculate the partial derivatives for this new point
        f2 = approx_fprime( x0, f, epsilon, *args)
        hessian[:, j] = (f2 - f1)/epsilon[j] # scale...
        xx[j] = xx0 # Restore initial value of x0
    return hessian

def show_contours(cnts, bg, title):
    viz = bg.copy()
    for cnt in cnts:
        for c in cnt:
            cv2.circle(viz, tuple(c.astype(np.int)), 1, (0,255,0), -1)
        cv2.polylines(viz, [cnt.astype(np.int)], True, (0,255,0), 2)

    plt.figure(figsize=(10,10));
    plt.imshow(viz);
#     plt.title(title);
    plt.axis('off');
    plt.show();

def show_levelset(levelset, bg, title):
    if bg.ndim == 3:
        viz = bg.copy()
    elif bg.ndim == 2:
        viz = gray2rgb(bg)
    cnts = find_contours(levelset, level=.5)
    for cnt in cnts:
        for c in cnt[:,::-1]:
            cv2.circle(viz, tuple(c.astype(np.int)), 1, (0,255,0), -1)
    plt.figure(figsize=(10,10));
    plt.imshow(viz, cmap=plt.cm.gray);
    plt.title(title);
    plt.axis('off');
    plt.show();

# http://deparkes.co.uk/2015/02/01/find-concave-hull-python/
# http://blog.thehumangeo.com/2014/05/12/drawing-boundaries-in-python/

from shapely.ops import cascaded_union, polygonize
from shapely.geometry import MultiLineString
from scipy.spatial import Delaunay
import numpy as np

def alpha_shape(coords, alphas):
    """
    Compute the alpha shape (concave hull) of a set
    of points.
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """

    tri = Delaunay(coords)

    pa = coords[tri.vertices[:,0]]
    pb = coords[tri.vertices[:,1]]
    pc = coords[tri.vertices[:,2]]

    a = np.sqrt(np.sum((pa - pb)**2, axis=1))
    b = np.sqrt(np.sum((pb - pc)**2, axis=1))
    c = np.sqrt(np.sum((pc - pa)**2, axis=1))
    s = (a + b + c)/2.
    area = np.sqrt(s*(s-a)*(s-b)*(s-c))
    circum_r = a*b*c/(4.0*area)

    geoms = []

    for al in alphas:
        edges = tri.vertices[circum_r < 1.0 / al]

        edge_points = []
        for ia, ib, ic in edges:
            edge_points.append(coords[ [ia, ib] ])
            edge_points.append(coords[ [ib, ic] ])
            edge_points.append(coords[ [ic, ia] ])

        m = MultiLineString(edge_points)
        triangles = list(polygonize(m))
        r = cascaded_union(triangles)

        geoms.append(r)


#     edges = tri.vertices[circum_r < 1.0/alpha]

# # slightly slower than below
# #     edge_points = list(chain(*[[coords[ [ia, ib] ], coords[ [ib, ic] ], coords[ [ic, ia] ]]
# #                    for ia, ib, ic in edges]))

#     edge_points = []
#     for ia, ib, ic in edges:
#         edge_points.append(coords[ [ia, ib] ])
#         edge_points.append(coords[ [ib, ic] ])
#         edge_points.append(coords[ [ic, ia] ])

#     m = MultiLineString(edge_points)
#     triangles = list(polygonize(m))
#     r = cascaded_union(triangles)

    return geoms

def less(center):
    def less_helper(a, b):
        if (a[0] - center[0] >= 0 and b[0] - center[0] < 0):
            return 1;
        if (a[0] - center[0] < 0 and b[0] - center[0] >= 0):
            return -1;
        if (a[0] - center[0] == 0 and b[0] - center[0] == 0):
            if (a[1] - center[1] >= 0 or b[1] - center[1] >= 0):
                return 2*int(a[1] > b[1]) - 1;
            return 2*int(b[1] > a[1]) - 1

        # compute the cross product of vectors (center -> a) x (center -> b)
        det = (a[0] - center[0]) * (b[1] - center[1]) - (b[0] - center[0]) * (a[1] - center[1])
        if (det < 0):
            return 1;
        if (det > 0):
            return -1;

        # points a and b are on the same line from the center
        # check which point is closer to the center
        d1 = (a[0] - center[0]) * (a[0] - center[0]) + (a[1] - center[1]) * (a[1] - center[1])
        d2 = (b[0] - center[0]) * (b[0] - center[0]) + (b[1] - center[1]) * (b[1] - center[1])
        return 2*int(d1 > d2) - 1

    return less_helper

def sort_vertices_counterclockwise(cnt):
    # http://stackoverflow.com/a/6989383
    center = cnt.mean(axis=0)
    return sorted(cnt, cmp=less(center))


def contour_to_concave_hull(cnt, levelset, alphas):

    xmin, ymin = cnt.min(axis=0)
    xmax, ymax = cnt.max(axis=0)

#     if levelset is None:

#         h, w = (ymax-ymin+1, xmax-xmin+1)
#         inside_ys, inside_xs = np.where(grid_points_in_poly((h, w), cnt[:, ::-1]-(ymin,xmin)))
#         n = inside_ys.size
#         random_indices = np.random.choice(range(n), min(5000, n), replace=False)
#         inside_points = np.c_[inside_xs[random_indices], inside_ys[random_indices]] + (xmin, ymin)

#     else:

    xs, ys = np.meshgrid(np.arange(xmin, xmax+1), np.arange(ymin, ymax+1))
    gridpoints = np.c_[xs.flat, ys.flat]
    inside_indices = np.where(levelset[gridpoints[:,1], gridpoints[:,0]] > 0)[0]
    n = inside_indices.size
    random_indices = np.random.choice(range(n), min(3000, n), replace=False)
    inside_points = gridpoints[inside_indices[random_indices]]


    geoms = alpha_shape(inside_points, alphas)

    base_area = np.sum(levelset)
    errs = np.array([(r.area if r.type == 'Polygon' else max([rr.area for rr in r])) - base_area for r in geoms])

#     plt.plot(errs);
#     plt.xticks(range(len(errs)), alphas);
#     plt.show();

#     plt.plot(np.abs(errs));
#     plt.xticks(range(len(errs)), alphas);
#     plt.show();

    c = np.argmin(np.abs(errs))
    r = geoms[c]

#     num_comps = np.array([1 if r.type == 'Polygon' else len(r) for r in geoms])
#     n = num_comps[-1]
#     while True:
#         c = np.min(np.where((num_comps == n) & (errs > 0)))
#         if errs[c] < 1e5:
#             break
#         n += 1

    if r.type == 'Polygon':
        concave_hull = r
    else:
        concave_hull = r[np.argmax([rr.area for rr in r])]

    # the heuristic rule here is:
    # merge two parts into one if the loss of including extraneous area is not larger
    # than the loss of sacrificing all parts other than the largest one

    if not hasattr(concave_hull, 'exterior'):
        sys.stderr.write('No concave hull produced.\n')
        return None

    if concave_hull.exterior.length < 20 * 3:
        point_interval = concave_hull.exterior.length / 4
    else:
        point_interval = 20
    new_cnt_subsampled = np.array([concave_hull.exterior.interpolate(r, normalized=True).coords[:][0]
                         for r in np.arange(0, 1, point_interval/concave_hull.exterior.length)],
               dtype=np.int)

    return new_cnt_subsampled, alphas[c]


def pad_scoremap(stack, sec, l, scoremaps_rootdir, bg_size):

    scoremaps_dir = os.path.join(scoremaps_rootdir, stack, '%04d'%sec)

    try:
#         scoremap_whole = bp.unpack_ndarray_file(os.path.join(scoremaps_dir,
#                                                    '%(stack)s_%(sec)04d_roi1_denseScoreMapLossless_%(label)s.bp' % \
#                                                    {'stack': stack, 'sec': sec, 'label': l}))

        scoremap_whole = load_hdf(os.path.join(scoremaps_dir,
                                                   '%(stack)s_%(sec)04d_roi1_denseScoreMapLossless_%(label)s.hdf' % \
                                                   {'stack': stack, 'sec': sec, 'label': l}))

    except:
        sys.stderr.write('No scoremap of %s exists\n' % (l))
        return None


    dataset = stack + '_' + '%04d'%sec + '_roi1'

    interpolation_xmin, interpolation_xmax, \
    interpolation_ymin, interpolation_ymax = np.loadtxt(os.path.join(scoremaps_dir,
                                                                     '%(dataset)s_denseScoreMapLossless_%(label)s_interpBox.txt' % \
                                    {'dataset': dataset, 'label': l})).astype(np.int)

    h, w = bg_size

    dense_scoremap_lossless = np.zeros((h, w), np.float32)
    dense_scoremap_lossless[interpolation_ymin:interpolation_ymax+1,
                            interpolation_xmin:interpolation_xmax+1] = scoremap_whole

    return dense_scoremap_lossless


def find_z_section_map(stack, volume_zmin, downsample_factor = 16):

    # factor = section_thickness/xy_pixel_distance_lossless

    xy_pixel_distance_downsampled = xy_pixel_distance_lossless * downsample_factor
    z_xy_ratio_downsampled = section_thickness / xy_pixel_distance_downsampled

    section_bs_begin, section_bs_end = section_range_lookup[stack]

    map_z_to_section = {}
    for s in range(section_bs_begin, section_bs_end+1):
        for z in range(int(z_xy_ratio_downsampled*s) - volume_zmin,
                       int(z_xy_ratio_downsampled*(s+1)) - volume_zmin + 1):
            map_z_to_section[z] = s

    return map_z_to_section

from data_manager import *


def surr_points(vertices):
    poly = Polygon(vertices)
    p1 = points_in_polygon(list(poly.buffer(10, resolution=2).exterior.coords))
    p2 = points_in_polygon(list(poly.exterior.coords))
    surr_pts = pts_arr_setdiff(p1, p2)
    return surr_pts

def points_in_polygon(polygon):
    pts = np.array(polygon, np.int)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    nz_ys, nz_xs =np.where(grid_points_in_poly((ymax-ymin+1, xmax-xmin+1), pts[:, ::-1]-[ymin, xmin]))
    nz2 = np.c_[nz_xs + xmin, nz_ys + ymin]
    return nz2

def pts_arr_setdiff(nz1, nz2):
    # http://stackoverflow.com/a/11903368
    a1_rows = nz1.view([('', nz1.dtype)] * nz1.shape[1])
    a2_rows = nz2.view([('', nz2.dtype)] * nz2.shape[1])
    surr_nzs = np.setdiff1d(a1_rows, a2_rows).view(nz1.dtype).reshape(-1, nz1.shape[1])
    return surr_nzs


def get_surround_voxels(volume, fill=False, num_samples=10000):
    """
    This does not get surround voxels at both sides in z direction.
    """

    if fill:
        from annotation_utilities import fill_sparse_volume
        volume = fill_sparse_volume(volume)

    from collections import defaultdict

    surr_volume = defaultdict(list)
    for z in range(volume.shape[2]):
        cnts = find_contour_points(volume[..., z])
        for l, cnt_parts in cnts.iteritems():
            cnt = cnt_parts[0]
            if len(cnt) < 5:
                continue
            surr_p = surr_points(cnt)
            if num_samples is not None:
                n = len(surr_p)
                sample_indices = np.random.choice(range(n), min(num_samples, n), replace=False)
                surr_p = surr_p[sample_indices]
            surr_volume[l].append(np.c_[surr_p, z*np.ones(len(surr_p),)])
    surr_volume.default_factory = None

    surr_nzs = {l: np.concatenate(arr_list).astype(np.int16) for l, arr_list in surr_volume.iteritems()}
    # surr_nzs = [np.concatenate(surr_volume[l]).T.astype(np.int16) for l in range(1, n_labels)]
    del surr_volume, surr_p, cnts

    return surr_nzs

def transform_points_2d(T, pts=None, c=None, pts_centered=None, c_prime=0):
    '''
    T: 1x6 vector
    c: center of volume 1
    c_prime: center of volume 2
    pts: nx2
    '''
    if pts_centered is None:
        pts_centered = pts - c

    Tm = np.reshape(T, (2,3))
    t = Tm[:, 2]
    A = Tm[:, :2]

    pts_prime = np.dot(A, pts_centered.T) + (t + c_prime)[:,None]
    return pts_prime.T

def transform_points_bspline(buvwx, buvwy, buvwz,
                             volume_shape=None, interval=None,
                             ctrl_x_intervals=None,
                             ctrl_y_intervals=None,
                             ctrl_z_intervals=None,
                             pts=None, c=(0,0,0), pts_centered=None, c_prime=(0,0,0),
                            NuNvNw_allTestPts=None):
    """
    Transform points by a B-spline transform.

    Args:
        volume_shape ((3,)-ndarray of int): (xdim, ydim, zdim)
        interval (int): control point spacing in x,y,z directions.
        pts ((n,3)-ndarray): input point coordinates.
        NuNvNw_allTestPts ((n_test, n_ctrlx * n_ctrly * n_ctrlz)-array)

    Returns:
        transformed_pts ((n,3)-ndarray): transformed point coordinates.
    """

    if pts_centered is None:
        assert pts is not None
        pts_centered = pts - c

    if NuNvNw_allTestPts is None:

        xdim, ydim, zdim = volume_shape
        if ctrl_x_intervals is None:
            ctrl_x_intervals = np.arange(0, xdim, interval)
        if ctrl_y_intervals is None:
            ctrl_y_intervals = np.arange(0, ydim, interval)
        if ctrl_z_intervals is None:
            ctrl_z_intervals = np.arange(0, zdim, interval)

        ctrl_x_intervals_centered = ctrl_x_intervals - c[0]
        ctrl_y_intervals_centered = ctrl_y_intervals - c[1]
        ctrl_z_intervals_centered = ctrl_z_intervals - c[2]

        t = time.time()

        NuPx_allTestPts = compute_bspline_cp_contribution_to_test_pts(control_points=ctrl_x_intervals_centered/float(interval),
                                                                     test_points=pts_centered[:,0]/float(interval))
        NvPy_allTestPts = compute_bspline_cp_contribution_to_test_pts(control_points=ctrl_y_intervals_centered/float(interval),
                                                                     test_points=pts_centered[:,1]/float(interval))
        NwPz_allTestPts = compute_bspline_cp_contribution_to_test_pts(control_points=ctrl_z_intervals_centered/float(interval),
                                                                     test_points=pts_centered[:,2]/float(interval))

#         NuPx_allTestPts = np.array([[N(ctrl_x/float(interval), x/float(interval)) for testPt_i, (x, y, z) in enumerate(pts_centered)]
#                                     for ctrlXInterval_i, ctrl_x in enumerate(ctrl_x_intervals_centered)])

#         NvPy_allTestPts = np.array([[N(ctrl_y/float(interval), y/float(interval)) for testPt_i, (x, y, z) in enumerate(pts_centered)]
#                                     for ctrlYInterval_i, ctrl_y in enumerate(ctrl_y_intervals_centered)])

#         NwPz_allTestPts = np.array([[N(ctrl_z/float(interval), z/float(interval)) for testPt_i, (x, y, z) in enumerate(pts_centered)]
#                                     for ctrlZInterval_i, ctrl_z in enumerate(ctrl_z_intervals_centered)])

        sys.stderr.write("Compute NuPx/NvPy/NwPz: %.2f seconds.\n" % (time.time() - t))

        # print NuPx_allTestPts.shape, NvPy_allTestPts.shape, NwPz_allTestPts.shape
        # (9, 157030) (14, 157030) (8, 157030)
        # (n_ctrlx, n_test)

        t = time.time()

        NuNvNw_allTestPts = np.einsum('it,jt,kt->ijkt', NuPx_allTestPts, NvPy_allTestPts, NwPz_allTestPts).reshape((-1, NuPx_allTestPts.shape[-1])).T

        # NuNvNw_allTestPts = np.array([np.ravel(np.tensordot(np.tensordot(NuPx_allTestPts[:,testPt_i],
        #                                                                  NvPy_allTestPts[:,testPt_i], 0),
        #                                                     NwPz_allTestPts[:,testPt_i], 0))
        #                           for testPt_i in range(len(pts_centered))])
        sys.stderr.write("Compute NuNvNw: %.2f seconds.\n" % (time.time() - t))

    # the expression inside np.ravel gives array of shape (n_ctrlx, n_ctrly, nctrlz)

    # print NuNvNw_allTestPts.shape
    # (157030, 1008)
    # (n_test, n_ctrlx * n_ctrly * n_ctrlz)

    # t = time.time()
    sum_uvw_NuNvNwbuvwx = np.dot(NuNvNw_allTestPts, buvwx)
    sum_uvw_NuNvNwbuvwy = np.dot(NuNvNw_allTestPts, buvwy)
    sum_uvw_NuNvNwbuvwz = np.dot(NuNvNw_allTestPts, buvwz)
    # sys.stderr.write("Compute sum: %.2f seconds.\n" % (time.time() - t))

    # print sum_uvw_NuNvNwbuvwx.shape

    transformed_pts = pts_centered + np.c_[sum_uvw_NuNvNwbuvwx, sum_uvw_NuNvNwbuvwy, sum_uvw_NuNvNwbuvwz] + c_prime
    return transformed_pts

def transform_points(pts, transform):
    '''
    Transform points.

    Args:
        pts:
        transform: any representation
    '''

    T = convert_transform_forms(transform=transform, out_form=(3,4))

    t = T[:, 3]
    A = T[:, :3]

    if len(np.atleast_2d(pts)) == 1:
            pts_prime = np.dot(A, np.array(pts).T) + t
    else:
        pts_prime = np.dot(A, np.array(pts).T) + t[:,None]

    return pts_prime.T

def transform_points_affine(T, pts=None, c=(0,0,0), pts_centered=None, c_prime=(0,0,0)):
    '''
    Transform points by a rigid or affine transform.

    Args:
        T ((nparams,)-ndarray): flattened array of transform parameters.
        c ((3,)-ndarray): origin of input points
        c_prime((3,)-ndarray): origin of output points
        pts ((n,3)-ndararay): coodrinates of input points
    '''

    if pts_centered is None:
        assert pts is not None
        pts_centered = pts - c

    Tm = np.reshape(T, (3,4))
    t = Tm[:, 3]
    A = Tm[:, :3]
    pts_prime = np.dot(A, pts_centered.T) + (t + c_prime)[:,None]

    return pts_prime.T


def transform_slice(img, T, centroid_m, centroid_f, xdim_f, ydim_f):
    nz_ys, nz_xs = np.where(img > 0)
    nzpixels_m_temp = np.c_[nz_xs, nz_ys]
    # "_temp" is appended to avoid name conflict with module level variable defined in registration.py

    nzs_m_aligned_to_f = transform_points_2d(T, pts=nzpixels_m_temp,
                            c=centroid_m, c_prime=centroid_f).astype(np.int16)

    img_m_aligned_to_f = np.zeros((ydim_f, xdim_f), img.dtype)

    xs_f, ys_f = nzs_m_aligned_to_f.T

    valid = (xs_f >= 0) & (ys_f >= 0) & \
            (xs_f < xdim_f) & (ys_f < ydim_f)

    xs_m, ys_m = nzpixels_m_temp.T

    img_m_aligned_to_f[ys_f[valid], xs_f[valid]] = img[ys_m[valid], xs_m[valid]]

    del nzs_m_aligned_to_f

    if np.issubdtype(img_m_aligned_to_f.dtype, np.float):
        # score volume
        dense_img = fill_sparse_score_image(img_m_aligned_to_f)
    # elif np.issubdtype(volume_m_aligned_to_f.dtype, np.integer):
    #     dense_volume = fill_sparse_volume(volume_m_aligned_to_f)
    else:
        raise Exception('transform_slice: Slice image must be float.')

    return dense_img

def transform_points_inverse(T, pts_prime=None, c_prime=np.array((0,0,0)), pts_prime_centered=None, c=np.array((0,0,0))):
    '''
    T: 1x12 vector, transform that maps pts to pts_prime
    c: center of volume 1
    c_prime: center of volume 2
    pts_prime: nx3
    '''

    Tm = np.reshape(T, (3,4))
    t = Tm[:, 3]
    A = Tm[:, :3]

    if pts_prime_centered is None:
        pts_prime_centered = pts_prime - c_prime

    pts = np.dot(np.linalg.inv(A), (pts_prime_centered-t).T) + c[:,None]

    return pts.T

def mahalanobis_distance_sq(nzs, mu, sigma):
    sigma_inv = np.linalg.inv(sigma)
    ds = nzs - mu
    dms = np.array([np.dot(d, np.dot(sigma_inv, d)) for d in ds])
    return dms

def transform_points_polyrigid_inverse_v2(pts_prime, rigid_param_list, anchor_points, sigmas, weights):
    """
    Transform points by the inverse of a weighted-average transform.

    Args:
        pts_prime ((n,3)-ndarray): points to transform
        rigid_param_list (list of (12,)-ndarrays): list of rigid transforms

    Returns:
        ((n,3)-ndarray): transformed points.
    """

    n_comp = len(rigid_param_list)
    n_voxels = len(pts_prime)

    Rs = [r.reshape((3,4))[:3,:3] for r in rigid_param_list]
    ts = [r.reshape((3,4))[:, 3] for r in rigid_param_list]
    Rs_inverse = [np.linalg.inv(R) for R in Rs]
    ts_inverse = [-np.dot(Rinv, t) for Rinv, t in zip(Rs_inverse, ts)]

    anchor_points_prime = np.array([np.dot(R, a) + t for R, t, a in zip(Rs, ts, anchor_points)])
    # print zip(anchor_points, anchor_points_prime)

    if sigmas[0].ndim == 2: # sigma is covariance matrix
        nzvoxels_weights = np.array([w*np.exp(-mahalanobis_distance_sq(pts_prime, ap, sigma))
                            for ap, sigma, w in zip(anchor_points_prime, sigmas, weights)])
    elif sigmas[0].ndim == 1: # sigma is a single scalar
        # nzvoxels_weights = np.array([w*np.exp(-np.sum((pts_prime - ap)**2, axis=1)/sigma**2) \
        #                     for ap, sigma, w in zip(anchor_points_prime, sigmas, weights)])
        nzvoxels_weights = np.array([w*1./(np.sum((pts_prime - ap)**2, axis=1)/sigma**2 + 1e-6) \
                    for ap, sigma, w in zip(anchor_points_prime, sigmas, weights)])
    nzvoxels_weights = nzvoxels_weights/nzvoxels_weights.sum(axis=0)
    # nzvoxels_weights[nzvoxels_weights < 1e-1] = 0
    # nzvoxels_weights = nzvoxels_weights/nzvoxels_weights.sum(axis=0)
    # n_components x n_voxels

    # print nzvoxels_weights

    nzs_m_aligned_to_f = np.array([np.sum([w * (np.dot(Rinv, p) + tinv) for w, Rinv,tinv in zip(ws, Rs_inverse,ts_inverse)], axis=0)
                                   for p, ws in zip(pts_prime, nzvoxels_weights.T)]).astype(np.int)
    return nzs_m_aligned_to_f

# def transform_points_polyrigid_inverse(pts_prime, rigid_param_list, anchor_points, sigmas, weights):
#     """
#     Transform points by the inverse of a weighted-average transform.

#     Args:
#         pts_prime ((n,3)-ndarray): points to transform
#         rigid_param_list (list of (12,)-ndarrays): list of rigid transforms

#     Returns:
#         ((n,3)-ndarray): transformed points.
#     """

#     n_comp = len(rigid_param_list)
#     n_voxels = len(pts_prime)

#     Rs = [r.reshape((3,4))[:3,:3] for r in rigid_param_list]
#     ts = [r.reshape((3,4))[:, 3] for r in rigid_param_list]
#     Rs_inverse = [np.linalg.inv(R) for R in Rs]
#     ts_inverse = [-np.dot(Rinv, t) for Rinv, t in zip(Rs_inverse, ts)]

#     anchor_points_prime = np.array([np.dot(R, a) + t for R, t, a in zip(Rs, ts, anchor_points)])
#     # print zip(anchor_points, anchor_points_prime)

#     if sigmas[0].ndim == 2: # sigma is covariance matrix
#         nzvoxels_weights = np.array([w*np.exp(-mahalanobis_distance_sq(pts_prime, ap, sigma))
#                             for ap, sigma, w in zip(anchor_points_prime, sigmas, weights)])
#     elif sigmas[0].ndim == 1: # sigma is a single scalar
#         nzvoxels_weights = np.array([w*np.exp(-np.sum((pts_prime - ap)**2, axis=1)/sigma**2) \
#                             for ap, sigma, w in zip(anchor_points_prime, sigmas, weights)])
#     nzvoxels_weights = nzvoxels_weights/nzvoxels_weights.sum(axis=0)
#     # nzvoxels_weights[nzvoxels_weights < 1e-1] = 0
#     # nzvoxels_weights = nzvoxels_weights/nzvoxels_weights.sum(axis=0)
#     # n_components x n_voxels

#     # print nzvoxels_weights

#     nzs_m_aligned_to_f = np.array([np.sum([w * (np.dot(Rinv, p) + tinv) for w, Rinv,tinv in zip(ws, Rs_inverse,ts_inverse)], axis=0)
#                                    for p, ws in zip(pts_prime, nzvoxels_weights.T)]).astype(np.int16)
#     return nzs_m_aligned_to_f


def transform_points_polyrigid(pts, rigid_param_list, anchor_points, sigmas, weights):
    """
    Transform points by a weighted-average transform.

    Args:
        rigid_param_list (list of (12,)-ndarrays): list of rigid transforms
        weights (list of float): weights of component transforms. Only the relative magnitude matters.

    Returns:
        ((n,3)-ndarray): transformed points.
    """

    # for ap, sigma, w in zip(anchor_points, sigmas, weights):
        # print w, -mahalanobis_distance_sq(pts, ap, sigma)

    if sigmas[0].ndim == 2: # sigma is covariance matrix
        nzvoxels_weights = np.array([w*np.exp(-mahalanobis_distance_sq(pts, ap, sigma))
                            for ap, sigma, w in zip(anchor_points, sigmas, weights)])
    elif sigmas[0].ndim == 1: # sigma is a single scalar
        nzvoxels_weights = np.array([w*np.exp(-np.sum((pts - ap)**2, axis=1)/(sigma**2)) \
                            for ap, sigma, w in zip(anchor_points, sigmas, weights)])
        # nzvoxels_weights = np.array([w*1./(np.sum((pts - ap)**2, axis=1)/sigma**2 + 1e-10) \
        #                     for ap, sigma, w in zip(anchor_points, sigmas, weights)])
        # nzvoxels_weights = np.array([w*1./(np.sqrt(np.sum((pts - ap)**2, axis=1))/sigma + 1e-10) \
        #                     for ap, sigma, w in zip(anchor_points, sigmas, weights)])
        # nzvoxels_weights = np.array([w*((np.sqrt(np.sum((pts - ap)**2, axis=1)) < sigma).astype(np.int) + 1e-10) \
        #                     for ap, sigma, w in zip(anchor_points, sigmas, weights)])
    # add a small constant to prevent from being rounded to 0.

    nzvoxels_weights = nzvoxels_weights / (nzvoxels_weights.sum(axis=0))
    nzvoxels_weights[nzvoxels_weights < 1e-3] = 0
    # nzvoxels_weights = nzvoxels_weights / (nzvoxels_weights.sum(axis=0) + 1e-10)

    # print nzvoxels_weights.sum(axis=0)

    # for x in nzvoxels_weights:
    #     print x

    # n_components x n_voxels

    nzs_m_aligned_to_f = np.zeros((len(pts), 3), dtype=np.float)

    for i, rigid_params in enumerate(rigid_param_list):
        nzs_m_aligned_to_f += nzvoxels_weights[i][:,None] * transform_points_affine(rigid_params, pts=pts).astype(np.float)

    nzs_m_aligned_to_f = nzs_m_aligned_to_f.astype(np.int32)
    return nzs_m_aligned_to_f


def transform_volume_polyrigid_v2(vol, rigid_param_list, anchor_points, sigmas, weights, out_bbox, fill_holes=True):
    """
    NEEDS REVIEW.

    Transform volume by weighted-average transform.

    Args:
        rigid_param_list (list of (12,)-ndarrays): list of rigid transforms, with initial shifts incorporated.
        weights (list of float): weights of component transforms. Only the relative magnitude matters.
    """

    nzvoxels_m_temp = parallel_where_binary(vol > 0)
    # "_temp" is appended to avoid name conflict with module level variable defined in registration.py

    nzs_m_aligned_to_f = transform_points_polyrigid(nzvoxels_m_temp, rigid_param_list, anchor_points, sigmas, weights).astype(np.int16)

    nzs_m_xmin_f, nzs_m_ymin_f, nzs_m_zmin_f = np.min(nzs_m_aligned_to_f, axis=0)
    nzs_m_xmax_f, nzs_m_ymax_f, nzs_m_zmax_f = np.max(nzs_m_aligned_to_f, axis=0)

    xdim_f = nzs_m_xmax_f - nzs_m_xmin_f + 1
    ydim_f = nzs_m_ymax_f - nzs_m_ymin_f + 1
    zdim_f = nzs_m_zmax_f - nzs_m_zmin_f + 1

    volume_m_aligned_to_f = np.zeros((ydim_f, xdim_f, zdim_f), vol.dtype)
    xs_f_wrt_bbox, ys_f_wrt_bbox, zs_f_wrt_inbbox = (nzs_m_aligned_to_f - (nzs_m_xmin_f, nzs_m_ymin_f, nzs_m_zmin_f)).T
    xs_m, ys_m, zs_m = nzvoxels_m_temp.T
    volume_m_aligned_to_f[ys_f_wrt_bbox, xs_f_wrt_bbox, zs_f_wrt_inbbox] = vol[ys_m, xs_m, zs_m]

    del nzs_m_aligned_to_f

    if np.issubdtype(volume_m_aligned_to_f.dtype, np.float):
        dense_volume = fill_sparse_score_volume(volume_m_aligned_to_f)
    elif np.issubdtype(volume_m_aligned_to_f.dtype, np.integer):
        dense_volume = fill_sparse_volume(volume_m_aligned_to_f)
        #dense_volume = volume_m_aligned_to_f
    else:
        raise Exception('transform_volume: Volume must be either float or int.')

    return dense_volume


def transform_volume_polyrigid(vol, rigid_param_list, anchor_points, sigmas, weights, out_bbox, fill_holes=True):
    """
    Transform volume by weighted-average transform.

    Args:
        rigid_param_list (list of (12,)-ndarrays): list of rigid transforms
        weights (list of float): weights of component transforms. Only the relative magnitude matters.

    """

    xmin_f, xmax_f, ymin_f, ymax_f, zmin_f, zmax_f = out_bbox
    xdim_f = xmax_f + 1 - xmin_f
    ydim_f = ymax_f + 1 - ymin_f
    zdim_f = zmax_f + 1 - zmin_f

    nzvoxels_m_temp = parallel_where_binary(vol > 0)
    nzs_m_aligned_to_f = transform_points_polyrigid(nzvoxels_m_temp, rigid_param_list, anchor_points, sigmas, weights)
    xs_f, ys_f, zs_f = nzs_m_aligned_to_f.T

    volume_m_aligned_to_f = np.zeros((ydim_f, xdim_f, zdim_f), vol.dtype)

    valid = (xs_f >= xmin_f) & (ys_f >= ymin_f) & (zs_f >= zmin_f) & \
    (xs_f < xmax_f) & (ys_f < ymax_f) & (zs_f < zmax_f)

    xs_m, ys_m, zs_m = nzvoxels_m_temp.T

    volume_m_aligned_to_f[ys_f[valid], xs_f[valid], zs_f[valid]] = \
    vol[ys_m[valid], xs_m[valid], zs_m[valid]]

    if fill_holes:
        if np.issubdtype(volume_m_aligned_to_f.dtype, np.float):
            dense_volume = fill_sparse_score_volume(volume_m_aligned_to_f)
        elif np.issubdtype(volume_m_aligned_to_f.dtype, np.integer):
            dense_volume = fill_sparse_volume(volume_m_aligned_to_f)
        else:
            raise Exception('transform_volume: Volume must be either float or int.')
    else:
        dense_volume = volume_m_aligned_to_f

    return dense_volume


def get_weighted_average_rigid_parameters(stack_m, stack_f, structures, alpha=100.):
    """
    Generate weighting parameters for the weighted averaging transform.

    Args:

    Returns:
        tuple (rigid_parameters (dict of (3,4)-array), anchor_points (dict of (3,)-array), sigmas (dict of float), weights (dict of float))
    """

    from data_manager import DataManager
    from utilities2015 import consolidate

    cutoff = .5 # Structure size is defined as the number of voxels whose value is above this cutoff probability.

    volumes = {}
    for structure in structures:
        try:
            volumes[structure] = \
            DataManager.load_transformed_volume(stack_m=stack_m,
                                                stack_f=stack_f,
                                                warp_setting=20,
                                                vol_type_f='score',
                                                vol_type_m='score',
                                                prep_id_f=2,
                                                detector_id_f=15,
                                                # vol_type_f='annotationAsScore',
                                                # vol_type_m='annotationAsScore',
                                                                    downscale=32,
                                                 structure=structure)
        except Exception as e:
            sys.stderr.write("Error loading volume for %s: %s\n" % (structure, str(e)))

    structures = volumes.keys()

    volume_moving_structure_sizes = {}
    for structure in structures:
        volume_moving_structure_sizes[structure] = np.count_nonzero(volumes[structure] > cutoff)

    total_size = sum(volume_moving_structure_sizes[s] for s in structures)
    structure_sizes_percent = {s: float(volume_moving_structure_sizes[s])/total_size
                               for s in structures}
    # each structure's size as a percetage of all structures' size.
    ################

    structure_covars = {}
    for s in structures:
        v = volumes[s]
        ys, xs, zs = np.where(v)
        nzs = np.c_[xs, ys, zs]
        nzsc = nzs - nzs.mean(axis=0)
        C = np.dot(nzsc.T, nzsc)/float(len(nzsc))
        S, V = np.linalg.eigh(C)
        structure_covars[s] = C

    ##################

    # Read Transform of each structure, do polyrigid transform

    rigid_parameters = {}
    anchor_points = {}
    weights = {}
    sigmas = {}

    for structure in structures:
        try:
            local_params, cm_rel2ann, cf_rel2ann, xdim_m, ydim_m, zdim_m, xdim_f, ydim_f, zdim_f = \
    DataManager.load_alignment_parameters(stack_m=stack_m, stack_f=stack_f,
                                                            warp_setting=17,
                                                                 # vol_type_m='annotationAsScore',
                                                                 # vol_type_f='annotationAsScore',
                                                              vol_type_m='score',
                                                                 vol_type_f='score',
                                                              prep_id_f=2,
                                                              detector_id_f=15,
                                          downscale=32,
                                             structure_m=structure,
                                             structure_f=structure)
            rigid_parameters[structure] = consolidate(local_params, cm_rel2ann, cf_rel2ann)[:3].flatten()
            anchor_points[structure] = cm_rel2ann
#             if structure == '7N_R':
#                 weights[structure] = 1
#             else:
#                 weights[structure] = 0
            weights[structure] = structure_sizes_percent[structure]
            sigmas[structure] = alpha * structure_covars[structure]
        except Exception as e:
            sys.stderr.write("Error loading structure-specific transform for %s: %s.\n" % (structure, str(e)))

    return rigid_parameters, anchor_points, sigmas, weights


def transform_volume_bspline(vol, buvwx, buvwy, buvwz, volume_shape, interval=None,
                             ctrl_x_intervals=None,
                             ctrl_y_intervals=None,
                             ctrl_z_intervals=None,
                             centroid_m=(0,0,0), centroid_f=(0,0,0),
                            fill_holes=True):
    """
    Transform volume by a B-spline transform.

    Args:
        vol (3d-ndarray or 2-tuple): input binary volume. If tuple, (volume in bbox, bbox).
        volume_shape (3-tuple): xdim, ydim, zdim
        interval (float): control point spacing in three directions.
    """

    if isinstance(vol, tuple):
        vol_in_bbox, (xmin, xmax, ymin, ymax, zmin, zmax) = vol
        vol_dtype = vol_in_bbox.dtype
        nzvoxels_m_temp = parallel_where_binary(vol_in_bbox > 0) + (xmin, ymin, zmin)
    else:
        nzvoxels_m_temp = parallel_where_binary(vol > 0)
        vol_dtype = vol.dtype
    # "_temp" is appended to avoid name conflict with module level variable defined in registration.py

    nzs_m_aligned_to_f = transform_points_bspline(buvwx, buvwy, buvwz, volume_shape=volume_shape,
                                                  interval=interval,
                                                  ctrl_x_intervals=ctrl_x_intervals,
                                                  ctrl_y_intervals=ctrl_y_intervals,
                                                  ctrl_z_intervals=ctrl_z_intervals,
                                                  pts=nzvoxels_m_temp,
                                                  c=centroid_m, c_prime=centroid_f).astype(np.int16)

    nzs_m_xmin_f, nzs_m_ymin_f, nzs_m_zmin_f = np.min(nzs_m_aligned_to_f, axis=0)
    nzs_m_xmax_f, nzs_m_ymax_f, nzs_m_zmax_f = np.max(nzs_m_aligned_to_f, axis=0)

    xdim_f = nzs_m_xmax_f - nzs_m_xmin_f + 1
    ydim_f = nzs_m_ymax_f - nzs_m_ymin_f + 1
    zdim_f = nzs_m_zmax_f - nzs_m_zmin_f + 1

    volume_m_aligned_to_f = np.zeros((ydim_f, xdim_f, zdim_f), vol_dtype)
    xs_f_wrt_bbox, ys_f_wrt_bbox, zs_f_wrt_bbox = (nzs_m_aligned_to_f - (nzs_m_xmin_f, nzs_m_ymin_f, nzs_m_zmin_f)).T
    xs_m, ys_m, zs_m = nzvoxels_m_temp.T

    if isinstance(vol, tuple):
        volume_m_aligned_to_f[ys_f_wrt_bbox, xs_f_wrt_bbox, zs_f_wrt_bbox] = vol_in_bbox[ys_m-ymin, xs_m-xmin, zs_m-zmin]
    else:
        volume_m_aligned_to_f[ys_f_wrt_bbox, xs_f_wrt_bbox, zs_f_wrt_bbox] = vol[ys_m, xs_m, zs_m]

    del nzs_m_aligned_to_f

    if fill_holes:
        if np.issubdtype(volume_m_aligned_to_f.dtype, np.float):
            dense_volume = fill_sparse_score_volume(volume_m_aligned_to_f)
        elif np.issubdtype(volume_m_aligned_to_f.dtype, np.integer):
            dense_volume = fill_sparse_volume(volume_m_aligned_to_f)
        else:
            raise Exception('transform_volume spline: Volume must be either float or int.')
        return dense_volume, (nzs_m_xmin_f, nzs_m_xmax_f, nzs_m_ymin_f, nzs_m_ymax_f, nzs_m_zmin_f, nzs_m_zmax_f)
    else:
        return volume_m_aligned_to_f, (nzs_m_xmin_f, nzs_m_xmax_f, nzs_m_ymin_f, nzs_m_ymax_f, nzs_m_zmin_f, nzs_m_zmax_f)

def transform_volume_v4(volume, transform=None, return_origin_instead_of_bbox=False):
    """
    One can specify initial shift and the transform separately.
    First, `centroid_m` and `centroid_f` are aligned.
    Then the tranform (R,t) parameterized by `tf_params` is applied.
    The relationship between coordinates in the fixed and moving volumes is:
    coord_f - centroid_f = np.dot(R, (coord_m - centroid_m)) + t

    One can also incorporate the initial shift into tf_params. In that case, do not specify `centroid_m` and `centroid_f`.
    coord_f = np.dot(T, coord_m)

    Args:
        volume ()
        transform ()

    Returns:
    """

    if isinstance(volume, np.ndarray):
        vol = volume
        origin = np.zeros((3,))
    elif isinstance(volume, tuple):
        if len(volume[1]) == 6: # bbox
            raise
        elif len(volume[1]) == 3: # origin
            vol = volume[0]
            origin = volume[1]
    else:
        raise


    tf_dict = convert_transform_forms(transform=transform, out_form='dict')
    tf_params = tf_dict['parameters']
    centroid_m = tf_dict['centroid_m_wrt_wholebrain']
    centroid_f = tf_dict['centroid_f_wrt_wholebrain']

    nzvoxels_m_temp = parallel_where_binary(vol > 0)
    # "_temp" is appended to avoid name conflict with module level variable defined in registration.py

    assert origin is not None or bbox is not None, 'Must provide origin or bbox.'
    if origin is None:
        if bbox is not None:
            origin = bbox[[0,2,4]]

    nzs_m_aligned_to_f = transform_points_affine(tf_params, pts=nzvoxels_m_temp + origin,
                            c=centroid_m, c_prime=centroid_f).astype(np.int16)

    nzs_m_xmin_f, nzs_m_ymin_f, nzs_m_zmin_f = np.min(nzs_m_aligned_to_f, axis=0)
    nzs_m_xmax_f, nzs_m_ymax_f, nzs_m_zmax_f = np.max(nzs_m_aligned_to_f, axis=0)

    xdim_f = nzs_m_xmax_f - nzs_m_xmin_f + 1
    ydim_f = nzs_m_ymax_f - nzs_m_ymin_f + 1
    zdim_f = nzs_m_zmax_f - nzs_m_zmin_f + 1

    volume_m_aligned_to_f = np.zeros((ydim_f, xdim_f, zdim_f), vol.dtype)
    xs_f_wrt_bbox, ys_f_wrt_bbox, zs_f_wrt_inbbox = (nzs_m_aligned_to_f - (nzs_m_xmin_f, nzs_m_ymin_f, nzs_m_zmin_f)).T
    xs_m, ys_m, zs_m = nzvoxels_m_temp.T
    volume_m_aligned_to_f[ys_f_wrt_bbox, xs_f_wrt_bbox, zs_f_wrt_inbbox] = vol[ys_m, xs_m, zs_m]

    del nzs_m_aligned_to_f

    t = time.time()

    if np.issubdtype(volume_m_aligned_to_f.dtype, np.float):
        dense_volume = fill_sparse_score_volume(volume_m_aligned_to_f)
    elif np.issubdtype(volume_m_aligned_to_f.dtype, np.integer):
        if not np.issubdtype(volume_m_aligned_to_f.dtype, np.uint8):
            dense_volume = fill_sparse_volume(volume_m_aligned_to_f)
        else:
            dense_volume = volume_m_aligned_to_f
    elif np.issubdtype(volume_m_aligned_to_f.dtype, bool):
        dense_volume = fill_sparse_score_volume(volume_m_aligned_to_f.astype(np.int)).astype(vol.dtype)
    else:
        raise Exception('transform_volume: Volume must be either float or int.')

    sys.stderr.write('Interpolating/filling sparse volume: %.2f seconds.\n' % (time.time() - t))

    if return_origin_instead_of_bbox:
        return dense_volume, np.array((nzs_m_xmin_f, nzs_m_ymin_f, nzs_m_zmin_f))
    else:
        return dense_volume, np.array((nzs_m_xmin_f, nzs_m_xmax_f, nzs_m_ymin_f, nzs_m_ymax_f, nzs_m_zmin_f, nzs_m_zmax_f))


# def transform_volume_v3(vol, tf_params=None, bbox=None, origin=None, centroid_m=(0,0,0), centroid_f=(0,0,0), return_origin_instead_of_bbox=False, transform=None):
#     """
#     One can specify initial shift and the transform separately.
#     First, `centroid_m` and `centroid_f` are aligned.
#     Then the tranform (R,t) parameterized by `tf_params` is applied.
#     The relationship between coordinates in the fixed and moving volumes is:
#     coord_f - centroid_f = np.dot(R, (coord_m - centroid_m)) + t
#
#     One can also incorporate the initial shift into tf_params. In that case, do not specify `centroid_m` and `centroid_f`.
#     coord_f = np.dot(T, coord_m)
#
#     Args:
#         vol (3D ndarray of float or int): the volume to transform. If dtype is int, treated as label volume; if is float, treated as score volume.
#         bbox (6-tuple): bounding box of the input volume.
#         origin (3-tuple)
#         tf_params ((nparam,)-ndarray): flattened vector of transform parameters. If `tf_params` already incorporates the initial shift that aligns two centroids, then there is no need to specify arguments `centroid_m` and `centroid_f`.
#         centroid_m (3-tuple): transform center in the volume to transform
#         centroid_f (3-tuple): transform center in the result volume.
#
#     Returns:
#         (3d array, 6-tuple): resulting volume, bounding box whose coordinates are relative to the input volume.
#     """
#
#     if transform_parameters is not None:
#         tf_dict = convert_transform_forms(transform_parameters=transform_parameters, out_form='dict')
#         tf_params = tf_dict['parameters']
#         centroid_m = tf_dict['centroid_m_wrt_wholebrain']
#         centroid_f = tf_dict['centroid_f_wrt_wholebrain']
#
#     nzvoxels_m_temp = parallel_where_binary(vol > 0)
#     # "_temp" is appended to avoid name conflict with module level variable defined in registration.py
#
#     assert origin is not None or bbox is not None, 'Must provide origin or bbox.'
#     if origin is None:
#         if bbox is not None:
#             origin = bbox[[0,2,4]]
#
#     nzs_m_aligned_to_f = transform_points_affine(tf_params, pts=nzvoxels_m_temp + origin,
#                             c=centroid_m, c_prime=centroid_f).astype(np.int16)
#
#     nzs_m_xmin_f, nzs_m_ymin_f, nzs_m_zmin_f = np.min(nzs_m_aligned_to_f, axis=0)
#     nzs_m_xmax_f, nzs_m_ymax_f, nzs_m_zmax_f = np.max(nzs_m_aligned_to_f, axis=0)
#
#     xdim_f = nzs_m_xmax_f - nzs_m_xmin_f + 1
#     ydim_f = nzs_m_ymax_f - nzs_m_ymin_f + 1
#     zdim_f = nzs_m_zmax_f - nzs_m_zmin_f + 1
#
#     volume_m_aligned_to_f = np.zeros((ydim_f, xdim_f, zdim_f), vol.dtype)
#     xs_f_wrt_bbox, ys_f_wrt_bbox, zs_f_wrt_inbbox = (nzs_m_aligned_to_f - (nzs_m_xmin_f, nzs_m_ymin_f, nzs_m_zmin_f)).T
#     xs_m, ys_m, zs_m = nzvoxels_m_temp.T
#     volume_m_aligned_to_f[ys_f_wrt_bbox, xs_f_wrt_bbox, zs_f_wrt_inbbox] = vol[ys_m, xs_m, zs_m]
#
#     del nzs_m_aligned_to_f
#
#     t = time.time()
#
#     if np.issubdtype(volume_m_aligned_to_f.dtype, np.float):
#         dense_volume = fill_sparse_score_volume(volume_m_aligned_to_f)
#     elif np.issubdtype(volume_m_aligned_to_f.dtype, np.integer):
#         if not np.issubdtype(volume_m_aligned_to_f.dtype, np.uint8):
#             dense_volume = fill_sparse_volume(volume_m_aligned_to_f)
#         else:
#             dense_volume = volume_m_aligned_to_f
#     elif np.issubdtype(volume_m_aligned_to_f.dtype, bool):
#         dense_volume = fill_sparse_score_volume(volume_m_aligned_to_f.astype(np.int)).astype(vol.dtype)
#     else:
#         raise Exception('transform_volume: Volume must be either float or int.')
#
#     sys.stderr.write('Interpolating/filling sparse volume: %.2f seconds.\n' % (time.time() - t))
#
#     if return_origin_instead_of_bbox:
#         return dense_volume, np.array((nzs_m_xmin_f, nzs_m_ymin_f, nzs_m_zmin_f))
#     else:
#         return dense_volume, np.array((nzs_m_xmin_f, nzs_m_xmax_f, nzs_m_ymin_f, nzs_m_ymax_f, nzs_m_zmin_f, nzs_m_zmax_f))


# def transform_volume_v2(vol, tf_params, centroid_m=(0,0,0), centroid_f=(0,0,0), fill_sparse=True):
#     """
#     One can specify initial shift and the transform separately.
#     First, `centroid_m` and `centroid_f` are aligned.
#     Then the tranform (R,t) parameterized by `tf_params` is applied.
#     The relationship between coordinates in the fixed and moving volumes is:
#     coord_f - centroid_f = np.dot(R, (coord_m - centroid_m)) + t
#
#     One can also incorporate the initial shift into tf_params. In that case, do not specify `centroid_m` and `centroid_f`.
#     coord_f = np.dot(T, coord_m)
#
#     Args:
#         vol (3D ndarray of float or int): the volume to transform. If dtype is int, treated as label volume; if is float, treated as score volume.
#         tf_params ((nparam,)-ndarray): flattened vector of transform parameters. If `tf_params` already incorporates the initial shift that aligns two centroids, then there is no need to specify arguments `centroid_m` and `centroid_f`.
#         centroid_m (3-tuple): transform center in the volume to transform
#         centroid_f (3-tuple): transform center in the result volume.
#
#     Returns:
#         (3d array, 6-tuple): resulting volume, bounding box whose coordinates are relative to the input volume.
#     """
#
#     t = time.time()
#     nzvoxels_m_temp = parallel_where_binary(vol > 0)
#     # "_temp" is appended to avoid name conflict with module level variable defined in registration.py
#     sys.stderr.write('parallel_where_binary: %.2f seconds.\n' % (time.time() - t))
#
#     t = time.time()
#     nzs_m_aligned_to_f = transform_points_affine(tf_params, pts=nzvoxels_m_temp,
#                             c=centroid_m, c_prime=centroid_f).astype(np.int16)
#     sys.stderr.write('transform_points_affine: %.2f seconds.\n' % (time.time() - t))
#
#     nzs_m_xmin_f, nzs_m_ymin_f, nzs_m_zmin_f = np.min(nzs_m_aligned_to_f, axis=0)
#     nzs_m_xmax_f, nzs_m_ymax_f, nzs_m_zmax_f = np.max(nzs_m_aligned_to_f, axis=0)
#
#     xdim_f = nzs_m_xmax_f - nzs_m_xmin_f + 1
#     ydim_f = nzs_m_ymax_f - nzs_m_ymin_f + 1
#     zdim_f = nzs_m_zmax_f - nzs_m_zmin_f + 1
#
#     volume_m_aligned_to_f = np.zeros((ydim_f, xdim_f, zdim_f), vol.dtype)
#     xs_f_wrt_bbox, ys_f_wrt_bbox, zs_f_wrt_inbbox = (nzs_m_aligned_to_f - (nzs_m_xmin_f, nzs_m_ymin_f, nzs_m_zmin_f)).T
#     xs_m, ys_m, zs_m = nzvoxels_m_temp.T
#     volume_m_aligned_to_f[ys_f_wrt_bbox, xs_f_wrt_bbox, zs_f_wrt_inbbox] = vol[ys_m, xs_m, zs_m]
#
#     del nzs_m_aligned_to_f
#
#     t = time.time()
#
#     if fill_sparse:
#         if np.issubdtype(volume_m_aligned_to_f.dtype, np.float):
#             print 'float'
#             dense_volume = fill_sparse_score_volume(volume_m_aligned_to_f)
#         elif np.issubdtype(volume_m_aligned_to_f.dtype, np.integer):
#             print 'int'
#             if np.issubdtype(volume_m_aligned_to_f.dtype, np.uint8):
#                 print 'uint8'
#                 dense_volume = volume_m_aligned_to_f
#             else:
#                 dense_volume = fill_sparse_volume(volume_m_aligned_to_f)
#         else:
#             raise Exception('transform_volume: Volume must be either float or int, not %s.' % volume_m_aligned_to_f.dtype)
#     else:
#         dense_volume = volume_m_aligned_to_f
#
#     sys.stderr.write('Interpolating/filling sparse volume: %.2f seconds.\n' % (time.time() - t))
#
#     return dense_volume, np.array((nzs_m_xmin_f, nzs_m_xmax_f, nzs_m_ymin_f, nzs_m_ymax_f, nzs_m_zmin_f, nzs_m_zmax_f))
#
#
# def transform_volume(vol, global_params, centroid_m=(0,0,0), centroid_f=(0,0,0), xdim_f=None, ydim_f=None, zdim_f=None):
#     """
#     First, centroid_m and centroid_f are aligned.
#     Then the tranform parameterized by global_params is applied.
#     The resulting volume will have dimension (xdim_f, ydim_f, zdim_f).
#
#     Args:
#         vol (3d array): the volume to transform
#         global_params (12-tuple): flattened vector of transform parameters
#         centroid_m (3-tuple): transform center in the volume to transform
#         centroid_f (3-tuple): transform center in the result volume.
#         xmin_f (int): if None, this is inferred from the
#         ydim_f (int): if None, the
#         zdim_f (int): if None, the
#     """
#
#     nzvoxels_m_temp = parallel_where_binary(vol > 0)
#     # "_temp" is appended to avoid name conflict with module level variable defined in registration.py
#
#     nzs_m_aligned_to_f = transform_points_affine(global_params, pts=nzvoxels_m_temp,
#                             c=centroid_m, c_prime=centroid_f).astype(np.int16)
#
#     volume_m_aligned_to_f = np.zeros((ydim_f, xdim_f, zdim_f), vol.dtype)
#
#     xs_f, ys_f, zs_f = nzs_m_aligned_to_f.T
#
#     valid = (xs_f >= 0) & (ys_f >= 0) & (zs_f >= 0) & \
#     (xs_f < xdim_f) & (ys_f < ydim_f) & (zs_f < zdim_f)
#
#     xs_m, ys_m, zs_m = nzvoxels_m_temp.T
#
#     volume_m_aligned_to_f[ys_f[valid], xs_f[valid], zs_f[valid]] = \
#     vol[ys_m[valid], xs_m[valid], zs_m[valid]]
#
#     del nzs_m_aligned_to_f
#
#     if np.issubdtype(volume_m_aligned_to_f.dtype, np.float):
#         dense_volume = fill_sparse_score_volume(volume_m_aligned_to_f)
#     elif np.issubdtype(volume_m_aligned_to_f.dtype, np.integer):
#         dense_volume = fill_sparse_volume(volume_m_aligned_to_f)
#         #dense_volume = volume_m_aligned_to_f
#     else:
#         raise Exception('transform_volume: Volume must be either float or int.')
#
#     return dense_volume

def transform_volume_inverse(vol, global_params, centroid_m, centroid_f, xdim_m, ydim_m, zdim_m):

    nzvoxels_m_temp = parallel_where_binary(vol > 0)
    # "_temp" is appended to avoid name conflict with module level variable defined in registration.py

    nzs_m_aligned_to_f = transform_points_inverse(global_params, pts_prime=nzvoxels_m_temp,
                            c=centroid_m, c_prime=centroid_f).astype(np.int16)

    # volume_m_aligned_to_f = np.zeros((ydim_f, xdim_f, zdim_f), vol.dtype)
    volume_m_aligned_to_f = np.zeros((ydim_m, xdim_m, zdim_m), vol.dtype) # Notice when reversing, m becomes f

    xs_f, ys_f, zs_f = nzs_m_aligned_to_f.T

    # valid = (xs_f >= 0) & (ys_f >= 0) & (zs_f >= 0) & \
    # (xs_f < xdim_f) & (ys_f < ydim_f) & (zs_f < zdim_f)
    # Notice when reversing, m becomes f
    valid = (xs_f >= 0) & (ys_f >= 0) & (zs_f >= 0) & \
    (xs_f < xdim_m) & (ys_f < ydim_m) & (zs_f < zdim_m)

    xs_m, ys_m, zs_m = nzvoxels_m_temp.T

    volume_m_aligned_to_f[ys_f[valid], xs_f[valid], zs_f[valid]] = \
    vol[ys_m[valid], xs_m[valid], zs_m[valid]]

    del nzs_m_aligned_to_f

    if np.issubdtype(volume_m_aligned_to_f.dtype, np.float):
        dense_volume = fill_sparse_score_volume(volume_m_aligned_to_f)
    elif np.issubdtype(volume_m_aligned_to_f.dtype, np.integer):
        dense_volume = fill_sparse_volume(volume_m_aligned_to_f)
    else:
        raise Exception('transform_volume: Volume must be either float or int.')

    return dense_volume

from skimage.morphology import closing, disk


def fill_sparse_score_image(img):
    """
    Densify a sparse 2D image.
    """
    dense_img = np.zeros_like(img)
    xmin, xmax, ymin, ymax = bbox_2d(img)
    roi = img[ymin:ymax+1, xmin:xmax+1]
    roi_dense_img = np.zeros_like(roi)
    roi_dense_img = closing((roi*255).astype(np.int)/255., disk(1))
    dense_img[ymin:ymax+1, xmin:xmax+1] = roi_dense_img.copy()
    return dense_img

def fill_sparse_score_volume(vol):
    """
    Densify a sparse 3D volume, by densifying every 2D slice.
    """
    dense_vol = np.zeros_like(vol)
    xmin, xmax, ymin, ymax, zmin, zmax = bbox_3d(vol)
    roi = vol[ymin:ymax+1, xmin:xmax+1, zmin:zmax+1]
    roi_dense_vol = np.zeros_like(roi)
    for z in range(roi.shape[2]):
        roi_dense_vol[..., z] = closing((roi[..., z]*255).astype(np.int)/255., disk(1))
    dense_vol[ymin:ymax+1, xmin:xmax+1, zmin:zmax+1] = roi_dense_vol.copy()
    return dense_vol


def fill_sparse_volume(volume_sparse):
    """
    Fill all holes of a integer-labeled volume. Assuming background label is 0.

    Args:
        volume_sparse (3D ndarray of int): sparse label volume.

    Returns:
        volume_filled (3D ndarray of int): filled label volume.
    """

    # Padding is necessary,
    # because if the input volume touches the border,
    # as a first step of closing, the dilation will fill the whole volume,
    # resulting in subsequent erosion not recovering the boundary.
    padding = 10
    closing_element_radius = 5
    # from skimage.morphology import binary_closing, ball
    from scipy.ndimage.morphology import binary_fill_holes, binary_closing

    volume = np.zeros_like(volume_sparse, np.int8)
    for ind in np.unique(volume_sparse):

        # Assume background label is 0.
        if ind == 0:
            continue

        vb = volume_sparse == ind
        xmin,xmax,ymin,ymax,zmin,zmax = bbox_3d(vb)
        vs = vb[ymin:ymax+1, xmin:xmax+1, zmin:zmax+1]
        vs_padded = np.pad(vs, ((padding,padding),(padding,padding),(padding,padding)),
                            mode='constant', constant_values=0)
        # t = time.time()
        # vs_padded_filled = binary_closing(vs_padded, ball(closing_element_radius))
        vs_padded_filled = binary_closing(vs_padded, structure=np.ones((closing_element_radius,closing_element_radius,closing_element_radius)))
        # print time.time() -t, 's'
        vs_filled = vs_padded_filled[padding:-padding, padding:-padding, padding:-padding]
        volume[ymin:ymax+1, xmin:xmax+1, zmin:zmax+1][vs_filled.astype(np.bool)] = ind

    return volume

# def alignment_parameters_to_transform_matrix(transform_parameters):
#     """
#     Returns:
#         (4,4) matrix that maps wholebrain domain of the moving brain to wholebrain domain of the fixed brain.
#     """
#     cf = np.array(transform_parameters['centroid_f'])
#     cm = np.array(transform_parameters['centroid_m'])
#     of = np.array(transform_parameters['domain_f_origin_wrt_wholebrain'])
#     om = np.array(transform_parameters['domain_m_origin_wrt_wholebrain'])
#     params = np.array(transform_parameters['parameters'])
#     T = consolidate(params=params, centroid_m=cm+om, centroid_f=cf+of)[:3]
#     return T

def convert_transform_forms(out_form, transform=None, aligner=None, select_best='last_value'):
    """
    Args:
        out_form: (3,4) or (4,4) or (12,) or "dict", "tuple"
    """

    if aligner is not None:
        centroid_m = aligner.centroid_m
        centroid_f = aligner.centroid_f

        if select_best == 'last_value':
            params = aligner.Ts[-1]
        elif select_best == 'max_value':
            params = aligner.Ts[np.argmax(aligner.scores)]
        else:
            raise Exception("select_best %s is not recognize." % select_best)
    else:
        if isinstance(transform, dict):
            if 'centroid_f_wrt_wholebrain' in transform:
                centroid_f = np.array(transform['centroid_f_wrt_wholebrain'])
                centroid_m = np.array(transform['centroid_m_wrt_wholebrain'])
                params = np.array(transform['parameters'])
            elif 'centroid_f' in transforms:
                centroid_f = np.array(transform['centroid_f'])
                centroid_m = np.array(transform['centroid_m'])
                params = np.array(transform['parameters'])
            else:
                raise
        elif isinstance(transform, np.ndarray):
            if transform.shape == (12,):
                params = transform
                centroid_m = np.zeros((3,))
                centroid_f = np.zeros((3,))
            elif transform.shape == (3,4):
                params = transform.flatten()
                centroid_m = np.zeros((3,))
                centroid_f = np.zeros((3,))
            elif transform.shape == (4,4):
                params = transform[:3].flatten()
                centroid_m = np.zeros((3,))
                centroid_f = np.zeros((3,))
            else:
                raise
        else:
            raise Exception("Transform type %s is not recognized" % type(transform))

    T = consolidate(params=params, centroid_m=centroid_m, centroid_f=centroid_f)

    if out_form == (3,4):
        return T[:3]
    elif out_form == (4,4):
        return T
    elif out_form == (12,):
        return T[:3].flatten()
    elif out_form == 'dict':
        return dict(centroid_f_wrt_wholebrain = np.zeros((3,)),
                    centroid_m_wrt_wholebrain = np.zeros((3,)),
                    parameters = T[:3].flatten())
    elif out_form == 'tuple':
        return T[:3].flatten(), np.zeros((3,)), np.zeros((3,))
    else:
        raise Exception("Output form of %s is not recognized." % out_form)

    return T


# def alignment_parameters_to_transform_matrix_v2(transform_parameters):
#     """
#     Returns:
#         (4,4) matrix that maps wholebrain domain of the moving brain to wholebrain domain of the fixed brain.
#     """
#
#     if isinstance(transform_parameters, dict):
#         centroid_f = np.array(transform_parameters['centroid_f_wrt_wholebrain'])
#         centroid_m = np.array(transform_parameters['centroid_m_wrt_wholebrain'])
#         params = np.array(transform_parameters['parameters'])
#     elif isinstance(transform_parameters, np.ndarray):
#         if transform_parameters.shape == (12,):
#             params = transform_parameters
#             centroid_m = np.zeros((3,))
#             centroid_f = np.zeros((3,))
#         elif transform_parameters.shape == (3,4):
#             params = transform_parameters.flatten()
#             centroid_m = np.zeros((3,))
#             centroid_f = np.zeros((3,))
#         elif transform_parameters.shape == (4,4):
#             params = transform_parameters[:3].flatten()
#             centroid_m = np.zeros((3,))
#             centroid_f = np.zeros((3,))
#         else:
#             raise
#     else:
#         raise Exception(type(transform_parameters))
#
#     T = consolidate(params=params, centroid_m=centroid_m, centroid_f=centroid_f)
#     return T

# def transform_volume_v2(volume, bbox=None, origin=None, transform=None):
#     """
#     Args:
#         vol: the volume to transform
#         bbox: wrt wholebrain
#         transform_parameters (dict): the dict that describes the transform
#
#     Returns:
#         (2-tuple): (volume, bounding box wrt wholebrain of fixed brain)
#     """
#
#     if origin is not None:
#         volume_m_warped_inbbox, volume_m_warped_origin_wrt_fixedWholebrain = \
#             transform_volume_v3(vol=volume, origin=origin, tranform=transform, return_origin_instead_of_bbox=True)
#         return volume_m_warped_inbbox, volume_m_warped_origin_wrt_fixedWholebrain
#     elif bbox is not None:
#         volume_m_warped_inbbox, volume_m_warped_bbox_wrt_fixedWholebrain = \
#             transform_volume_v3(vol=volume, bbox=bbox,  tranform=transform)
#         return volume_m_warped_inbbox, volume_m_warped_bbox_wrt_fixedWholebrain
#     else:
#         raise

# def transform_points_by_transform_parameters_v2(pts, transform_parameters):
#     """
#     Args:
#         pts ((n,3)-array): wrt wholebrain
#     """
#
#     T = alignment_parameters_to_transform_matrix_v2(transform_parameters)
#     R = T[:3,:3]
#     t = T[:3,3]
#     return np.dot(R, np.array(pts).T).T + t

# def transform_volume_by_alignment_parameters(volume, transform_parameters=None, bbox=None, origin=None):
#     """
#     Args:
#         vol: the volume to transform
#         bbox: wrt wholebrain
#         transform_parameters (dict): the dict that describes the transform
#
#     Returns:
#         (2-tuple): (volume, bounding box wrt wholebrain of fixed brain)
#     """
#
#     T = alignment_parameters_to_transform_matrix(transform_parameters)
#
#     if origin is not None:
#         volume_m_warped_inbbox, volume_m_warped_origin_wrt_fixedWholebrain = \
#             transform_volume_v3(vol=volume, origin=origin, tf_params=T.flatten(), return_origin_instead_of_bbox=True)
#         return volume_m_warped_inbbox, volume_m_warped_origin_wrt_fixedWholebrain
#     elif bbox is not None:
#         volume_m_warped_inbbox, volume_m_warped_bbox_wrt_fixedWholebrain = \
#             transform_volume_v3(vol=volume, bbox=bbox, tf_params=T.flatten())
#         return volume_m_warped_inbbox, volume_m_warped_bbox_wrt_fixedWholebrain
#     else:
#         raise

# def transform_points_by_transform_parameters(pts, transform_parameters):
#     """
#     Args:
#         pts ((n,3)-array): wrt wholebrain
#     """
#
#     T = alignment_parameters_to_transform_matrix(transform_parameters)
#     R = T[:3,:3]
#     t = T[:3,3]
#     return np.dot(R, np.array(pts).T).T + t

def compose_alignment_parameters(list_of_transform_parameters):
    """
    Args:
        list_of_transform_parameters: the transforms are applied in the order from left to right.

    Returns:
        (4,4)-array: transform matrix that maps wholebrain domain of moving brain to wholebrain domain of fixed brain.
    """

    T0 = np.eye(4)

    for transform_parameters in list_of_transform_parameters:
        
        T = convert_transform_forms(out_form=(4,4), transform=transform_parameters)

#         if isinstance(transform_parameters, dict):
#             # cf = np.array(transform_parameters['centroid_f'])
#             # cm = np.array(transform_parameters['centroid_m'])
#             # of = np.array(transform_parameters['domain_f_origin_wrt_wholebrain'])
#             # om = np.array(transform_parameters['domain_m_origin_wrt_wholebrain'])
#             # params = np.array(transform_parameters['parameters'])
#             # T = consolidate(params=params, centroid_m=cm+om, centroid_f=cf+of)

#             cf = np.array(transform_parameters['centroid_f_wrt_wholebrain'])
#             cm = np.array(transform_parameters['centroid_m_wrt_wholebrain'])
#             params = np.array(transform_parameters['parameters'])
#             T = consolidate(params=params, centroid_m=cm, centroid_f=cf)
#         elif transform_parameters.shape == (3,4):
#             T = np.vstack([transform_parameters, [0,0,0,1]])
#         elif transform_parameters.shape == (12,):
#             T = np.vstack([transform_parameters.reshape((3,4)), [0,0,0,1]])
#         elif transform_parameters.shape == (4,4):
#             T = transform_parameters
#         else:
#             print transform_parameters.shape
#             raise

        T0 = np.dot(T, T0)

    return T0


def fit_plane(X):
    """
    Fit a plane to a set of 3d points

    Parameters
    ----------
    X : n x 3 array
        points

    Returns
    ------
    normal : (3,) vector
        the normal vector of the plane
    c : (3,) vector
        a point on the plane
    """

    # http://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
    # http://math.stackexchange.com/a/3871
    X = np.array(X)
    c = X.mean(axis=0)
    Xc = X - c
    U, _, VT = np.linalg.svd(Xc.T)
    return U[:,-1], c

def R_align_two_vectors(a, b):
    """
    Find the rotation matrix that align a onto b.
    """
    # http://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/897677#897677

    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    v_skew = np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])
    R = np.eye(3) + v_skew + np.dot(v_skew, v_skew)*(1-c)/(s + 1e-5)**2
    return R


def average_location(centroid_allLandmarks=None, mean_centroid_allLandmarks=None):
    """
    Compute the standard centroid of every structure.

    This first estimates the mid-sagittal plane.
    Then find a standard centroid for each structure, that is closest to the mean and also being symmetric with respect to mid-sagittal plane.

    Args:
        centroid_allLandmarks (dict {str: (n,3)-array})
        mean_centroid_allLandmarks (dict {str: (3,)-array})

    Returns:
        standard_centroids_wrt_canonical: average locations of every structure, relative to the midplane anchor. Paired structures are symmetric relative to the mid-plane defined by centroid and normal.
        instance_centroids_wrt_canonical: the instance centroids in canonical atlas space
        midplane_anchor: a point on the mid-sagittal plane
        midplane_normal: normal vector of the mid-sagittal plane estimated from centroids in original coordinates. Note that this is NOT the mid-plane normal using canonical coordinates, which by design should be (0,0,1).
        transform_matrix_to_atlasCanonicalSpace: (4,4) matrix that maps to canonical atlas space
        """

    if mean_centroid_allLandmarks is None:
        mean_centroid_allLandmarks = {name: np.mean(centroids, axis=0)
                                  for name, centroids in centroid_allLandmarks.iteritems()}

    names = set([convert_to_original_name(name_s) for name_s in mean_centroid_allLandmarks.keys()])

    # Fit a midplane from the midpoints of symmetric landmark centroids
    midpoints = {}
    for name in names:
        lname = convert_to_left_name(name)
        rname = convert_to_right_name(name)

#         names = labelMap_unsidedToSided[name]

#         # maybe ignoring singular instances is better
#         if len(names) == 2:
        if lname in mean_centroid_allLandmarks and rname in mean_centroid_allLandmarks:
            midpoints[name] = .5 * mean_centroid_allLandmarks[lname] + .5 * mean_centroid_allLandmarks[rname]
        else:
            midpoints[name] = mean_centroid_allLandmarks[name]

    # print midpoints

    midplane_normal, midplane_anchor = fit_plane(np.c_[midpoints.values()])

    print 'Mid-sagittal plane normal vector =', midplane_normal
    print 'Mid-sagittal plane anchor =', midplane_anchor

    R_to_canonical = R_align_two_vectors(midplane_normal, (0, 0, 1))

    # points_midplane_oriented = {name: np.dot(R_to_canonical, p - midplane_anchor)
    #                             for name, p in mean_centroid_allLandmarks.iteritems()}

    transform_matrix_to_atlasCanonicalSpace = consolidate(params=np.column_stack([R_to_canonical, np.zeros((3,))]),
                centroid_m=midplane_anchor,
               centroid_f=(0,0,0))

    print 'Transform matrix to canonical atlas space ='
    print transform_matrix_to_atlasCanonicalSpace

    print 'Angular deviation around y axis (degree) =', np.rad2deg(np.arccos(midplane_normal[2]))

    points_midplane_oriented = {name: transform_points(p, transform=transform_matrix_to_atlasCanonicalSpace)
                                for name, p in mean_centroid_allLandmarks.iteritems()}

    if centroid_allLandmarks is not None:
    
        instance_centroid_rel2atlasCanonicalSpace = \
    {n: transform_points(s,transform=transform_matrix_to_atlasCanonicalSpace)
    for n, s in centroid_allLandmarks.iteritems()}
    else:
        instance_centroid_rel2atlasCanonicalSpace = None
    
    canonical_locations = {}

    for name in names:

        lname = convert_to_left_name(name)
        rname = convert_to_right_name(name)

        if lname in points_midplane_oriented and rname in points_midplane_oriented:

            x, y, mz = .5 * points_midplane_oriented[lname] + .5 * points_midplane_oriented[rname]

            canonical_locations[lname] = np.r_[x, y, points_midplane_oriented[lname][2]-mz]
            canonical_locations[rname] = np.r_[x, y, points_midplane_oriented[rname][2]-mz]
        else:
            x, y, _ = points_midplane_oriented[name]
            canonical_locations[name] = np.r_[x, y, 0]

    return canonical_locations, instance_centroid_rel2atlasCanonicalSpace, midplane_anchor, midplane_normal, transform_matrix_to_atlasCanonicalSpace
