import matplotlib

import os
import csv
import sys
from operator import itemgetter
from subprocess import check_output, call
import json
import cPickle as pickle
from datetime import datetime
import time
from operator import itemgetter
from itertools import izip
from collections import defaultdict

from multiprocess import Pool
from skimage.io import imread, imsave
from skimage.util import img_as_ubyte, img_as_float
from skimage.color import gray2rgb, rgb2gray
import numpy as np
import matplotlib.pyplot as plt
try:
    import cv2
except:
    sys.stderr.write('Cannot load cv2.\n')

import bloscpack as bp

from ipywidgets import FloatProgress
from IPython.display import display

from skimage.measure import grid_points_in_poly, subdivide_polygon, approximate_polygon
from skimage.measure import find_contours, regionprops

from skimage.filters import gaussian


def convert_defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d


def compute_gradient_v2(volume, smooth_first=False, dtype=np.float16):
    """
    Args:
        volume
        smooth_first (bool): If true, smooth each volume before computing gradients.
        This is useful if volume is binary and gradients are only nonzero at structure borders.

    Note:
        # 3.3 second - re-computing is much faster than loading
        # .astype(np.float32) is important;
        # Otherwise the score volume is type np.float16, np.gradient requires np.float32 and will have to convert which is very slow
        # 2s (float32) vs. 20s (float16)
    """

    if isinstance(volume, dict):

        # gradients = {}
        # for ind, (v, o) in volumes.iteritems():
        #     print "Computing gradient for", ind
        #     # t1 = time.time()
        #     gradients[ind] = (compute_gradient_v2((v, o), smooth_first=smooth_first), o)
        #     # sys.stderr.write("Overall: %.2f seconds.\n" % (time.time()-t1))

        gradients = {ind: compute_gradient_v2((v, o), smooth_first=smooth_first)
                     for ind, (v, o) in volume.iteritems()}

        return gradients

    else:
        v, o = convert_volume_forms(volume, out_form=("volume", "origin"))

        g = np.zeros((3,) + v.shape)

        # t = time.time()
        cropped_v, (xmin,xmax,ymin,ymax,zmin,zmax) = crop_volume_to_minimal(v, margin=5, return_origin_instead_of_bbox=False)
        # sys.stderr.write("Crop: %.2f seconds.\n" % (time.time()-t))

        if smooth_first:
            # t = time.time()
            cropped_v = gaussian(cropped_v, 3)
            # sys.stderr.write("Smooth: %.2f seconds.\n" % (time.time()-t))

        # t = time.time()
        cropped_v_gy_gx_gz = np.gradient(cropped_v.astype(np.float32), 3, 3, 3)
        # sys.stderr.write("Compute gradient: %.2f seconds.\n" % (time.time()-t))

        g[0][ymin:ymax+1, xmin:xmax+1, zmin:zmax+1] = cropped_v_gy_gx_gz[1]
        g[1][ymin:ymax+1, xmin:xmax+1, zmin:zmax+1] = cropped_v_gy_gx_gz[0]
        g[2][ymin:ymax+1, xmin:xmax+1, zmin:zmax+1] = cropped_v_gy_gx_gz[2]

        return g.astype(dtype), o


def load_data(fp, polydata_instead_of_face_vertex_list=True, download_s3=True):

    from vis3d_utilities import load_mesh_stl
    from distributed_utilities import download_from_s3

    if download_s3:
        download_from_s3(fp)

    if fp.endswith('.bp'):
        data = bp.unpack_ndarray_file(fp)
    elif fp.endswith('.json'):
        data = load_json(fp)
    elif fp.endswith('.pkl'):
        data = load_pickle(fp)
    elif fp.endswith('.stl'):
        data = load_mesh_stl(fp, return_polydata_only=polydata_instead_of_face_vertex_list)
    elif fp.endswith('.txt'):
        data = np.loadtxt(fp)
    elif fp.endswith('.png') or fp.endswith('.tif'):
        data = imread(fp)
    else:
        raise

    return data

def save_data(data, fp, upload_s3=True):

    from distributed_utilities import upload_to_s3
    from vis3d_utilities import save_mesh_stl

    create_parent_dir_if_not_exists(fp)

    if fp.endswith('.bp'):
        bp.pack_ndarray_file(np.ascontiguousarray(data), fp) 
        # ascontiguousarray is important, without which sometimes the loaded array will be different from saved.
    elif fp.endswith('.json'):
        save_json(data, fp)
    elif fp.endswith('.pkl'):
        save_pickle(data, fp)
    elif fp.endswith('.hdf'):
        save_hdf_v2(data, fp)
    elif fp.endswith('.stl'):
        save_mesh_stl(data, fp)
    elif fp.endswith('.txt'):
        if isinstance(data, np.ndarray):
            np.savetxt(fp, data)
        else:
            raise
    elif fp.endswith('.png') or fp.endswith('.tif') or fp.endswith('.jpg'):
        imsave(fp, data)
    else:
        raise

    if upload_s3:
        upload_to_s3(fp)

##################################################################

def mirror_volume_v2(volume, new_centroid, centroid_wrt_origin=None):
    """
    Use to get the mirror image of the volume.

    `Volume` argument is the volume in right hemisphere.
    Note: This assumes the mirror plane is vertical; Consider adding a mirror plane as argument

    Args:
        volume: any representation
        new_centroid: the centroid of the resulting mirrored volume.
        centroid_wrt_origin: if not specified, this uses the center of mass.

    Returns:
        (volume, origin): new origin is wrt the same coordinate frame as `new_centroid`.
    """

    vol, ori = convert_volume_forms(volume=volume, out_form=("volume", "origin"))
    ydim, xdim, zdim = vol.shape
    if centroid_wrt_origin is None:
        centroid_wrt_origin = get_centroid_3d(vol)
    centroid_x_wrt_origin, centroid_y_wrt_origin, centroid_z_wrt_origin = centroid_wrt_origin
    new_origin_wrt_centroid = (-centroid_x_wrt_origin, -centroid_y_wrt_origin, - (zdim - 1 - centroid_z_wrt_origin))

    new_origin = new_centroid + new_origin_wrt_centroid
    new_vol = vol[:,:,::-1].copy()
    return new_vol, new_origin

def convert_volume_forms(volume, out_form):
    """
    Convert a (volume, origin) tuple into a bounding box.
    """

    if isinstance(volume, np.ndarray):
        vol = volume
        ori = np.zeros((3,))
    elif isinstance(volume, tuple):
        assert len(volume) == 2
        vol = volume[0]
        if len(volume[1]) == 3:
            ori = volume[1]
        elif len(volume[1]) == 6:
            ori = volume[1][[0,2,4]]
        else:
            raise

    bbox = np.array([ori[0], ori[0] + vol.shape[1]-1, ori[1], ori[1] + vol.shape[0]-1, ori[2], ori[2] + vol.shape[2]-1])

    if out_form == ("volume", 'origin'):
        return (vol, ori)
    elif out_form == ("volume", 'bbox'):
        return (vol, bbox)
    elif out_form == "volume":
        return vol
    else:
        raise Exception("out_form %s is not recognized.")

        
def volume_origin_to_bbox(v, o):
    """
    Convert a (volume, origin) tuple into a bounding box.
    """
    return np.array([o[0], o[0] + v.shape[1]-1, o[1], o[1] + v.shape[0]-1, o[2], o[2] + v.shape[2]-1])

####################################################################

def get_structure_length_at_direction(structure_vol, d):

    xyzs = np.array(np.where(structure_vol))[[1,0,2]]
    q = np.dot(d/np.linalg.norm(d), xyzs)
    structure_length = q.max() - q.min()
    return structure_length

####################################################################

def plot_by_method_by_structure(data_all_stacks_all_structures, structures, stacks,
                                stack_to_color=None, ylabel='', title='', ylim=[0,1],
                                yspacing=.2, style='scatter',
                               figsize=(20, 6), spacing_btw_stacks=1,
                               xticks_fontsize=20):

    if stack_to_color is None:
        stack_to_color = {stack: random_colors(1)[0] for stack in data_all_stacks_all_structures.keys()}

    n_structures = len(structures)
    n_stacks = len(data_all_stacks_all_structures.keys())

    plt.figure(figsize=figsize);
    for stack_i, stack in enumerate(stacks):
        data_all_structures = data_all_stacks_all_structures[stack]
        data_mean = [np.mean(data_all_structures[s]) for s in structures]
        data_std = [np.std(data_all_structures[s]) for s in structures]
        plt.bar(stack_i + (n_stacks + spacing_btw_stacks) * np.arange(n_structures), data_mean, yerr=data_std, label=stack)

        plt.gca().yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.5)
        # Hide these grid behind plot objects
        plt.gca().set_axisbelow(True)

    for structure_i in xrange(0, n_structures):
        plt.axvline(x = n_stacks + (n_stacks + spacing_btw_stacks) * structure_i + spacing_btw_stacks / 2.,
                    color='k', linewidth=.3)

    plt.xticks(np.arange((n_stacks)/2, n_structures*(n_stacks + spacing_btw_stacks), (n_stacks + spacing_btw_stacks)),
               structures, rotation='60', fontsize=xticks_fontsize);
    plt.yticks(np.arange(ylim[0], ylim[1] + yspacing, yspacing),
               map(lambda x: '%.2f'%x, np.arange(ylim[0], ylim[1]+yspacing, yspacing)),
               fontsize=20);
    plt.xlabel('Structures', fontsize=20);
    plt.ylabel(ylabel, fontsize=20);
    plt.xlim([-1, len(structures) * (n_stacks + spacing_btw_stacks) + 1]);
    plt.ylim(ylim);
    plt.legend();
    plt.title(title, fontsize=20);


def plot_by_stack_by_structure(data_all_stacks_all_structures, structures,
                               yticks=None, yticklabel_fmt='%.2f', yticks_fontsize=15,
                               stack_to_color=None, ylabel='', title='', style='scatter',
                               figsize=(20, 6), xticks_fontsize=12, xlabel='Structures', xlim=None,
                              ):
    """
    Plot the input data, with structures as x-axis. Different stacks are represented using different colors.
    
    Args:
        style (str): scatter or boxplot.
    """

    if stack_to_color is None:
        stack_to_color = {stack: random_colors(1)[0] for stack in data_all_stacks_all_structures.keys()}

    fig, ax = plt.subplots(figsize=figsize)

    if style == 'scatter':
        for stack in sorted(data_all_stacks_all_structures.keys()):
            data_all_structures = data_all_stacks_all_structures[stack]
            vals = [data_all_structures[s] if s in data_all_structures else None
                    for i, s in enumerate(structures)]
            ax.scatter(range(len(vals)), vals, marker='o', s=100, label=stack, c=np.array(stack_to_color[stack])/255.);
    elif style == 'boxplot':
        
        D = [[data_all_stacks_all_structures[stack][struct] 
              for stack in data_all_stacks_all_structures.iterkeys() 
             if struct in data_all_stacks_all_structures[stack]]
            for struct in structures]
        
        bplot = plt.boxplot(np.array(D), positions=range(0, len(structures)), patch_artist=True);
#         for patch in bplot['boxes']:
#             patch.set_facecolor(np.array(stack_to_color[stack])/255.)

        ax.yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.5)
        # Hide these grid behind plot objects
        ax.set_axisbelow(True)
    else:
        raise Exception("%s is not recognized." % style)

    plt.xticks(range(len(structures)), structures, rotation='60', fontsize=xticks_fontsize);

    # plt.yticks(np.arange(ylim[0], ylim[1] + yspacing, yspacing),
    #            map(lambda x: '%.2f'%x, np.arange(ylim[0], ylim[1]+yspacing, yspacing)),
    #            fontsize=20);
    plt.yticks(yticks, [yticklabel_fmt % y for y in yticks], fontsize=yticks_fontsize);
    plt.xlabel(xlabel, fontsize=20);
    plt.ylabel(ylabel, fontsize=20);
    if xlim is None:
        xlim = [-1, len(structures)+1]
    ax.set_xlim(xlim);
    ax.set_ylim([yticks[0], yticks[-1]+yticks[-1]-yticks[-2]]);
    plt.legend();
    ax.set_title(title, fontsize=20);
    
    return fig, ax

#####################################################################

def identify_shape(img_fp):
    return map(int, check_output("identify -format %%Wx%%H \"%s\"" % img_fp, shell=True).split('x'))

#####################################################################

def get_timestamp_now(fmt="%m%d%Y%H%M%S"):
    from datetime import datetime
    return datetime.now().strftime(fmt)

######################################################################

def rescale_by_resampling(v, scaling=None, new_shape=None):
    """
    Args:
        new_shape: width, height
    """

    # print v.shape, scaling

    if new_shape is not None:
        return v[np.meshgrid(np.floor(np.linspace(0, v.shape[0]-1, new_shape[1])).astype(np.int),
                  np.floor(np.linspace(0, v.shape[1]-1, new_shape[0])).astype(np.int), indexing='ij')]
    else:
        if scaling == 1:
            return v

        if v.ndim == 3:
            if v.shape[-1] == 3: # RGB image
                return v[np.meshgrid(np.floor(np.arange(0, v.shape[0], 1./scaling)).astype(np.int),
                  np.floor(np.arange(0, v.shape[1], 1./scaling)).astype(np.int), indexing='ij')]
            else: # 3-d volume
                return v[np.meshgrid(np.floor(np.arange(0, v.shape[0], 1./scaling)).astype(np.int),
                  np.floor(np.arange(0, v.shape[1], 1./scaling)).astype(np.int),
                  np.floor(np.arange(0, v.shape[2], 1./scaling)).astype(np.int), indexing='ij')]
        elif v.ndim == 2:
            return v[np.meshgrid(np.floor(np.arange(0, v.shape[0], 1./scaling)).astype(np.int),
                  np.floor(np.arange(0, v.shape[1], 1./scaling)).astype(np.int), indexing='ij')]
        else:
            raise


# def rescale_volume_by_resampling(vol_m, scaling, dtype=None, return_mapping_only=False):
#     """
#     Args:
#         T ((12,)-array): affine transform matrix from fixed vol coords to moving vol coords.
#         scaling (float): the scaling factor.
#     Returns:
#         vol_m_warped_to_f
#     """
#
#     if dtype is None:
#         dtype = vol_m.dtype
#
#     ydim_m, xdim_m, zdim_m = vol_m.shape
#     xdim_f, ydim_f, zdim_f = (int(np.round(xdim_m * scaling)),
#                               int(np.round(ydim_m * scaling)),
#                             int(np.round(zdim_m * scaling)))
#
#     xyzs_f = np.array(np.meshgrid(range(xdim_f), range(ydim_f), range(zdim_f), indexing='xy'))
#     xyzs_f = np.rollaxis(xyzs_f, axis=0, start=4)
#     xyzs_f = xyzs_f.reshape((-1,3))
#
#     xyzs_m = np.round(xyzs_f/scaling).astype(np.int)
#
#     if return_mapping_only:
#         return xyzs_m, xyzs_f
#
#     vol_m_warped_to_f = np.zeros((ydim_f, xdim_f, zdim_f), dtype=dtype)
#
#     valid_mask = (xyzs_m[:,2] >= 0) & (xyzs_m[:,2] < vol_m.shape[2]) \
#                 & (xyzs_m[:,0] >= 0) & (xyzs_m[:,0] < vol_m.shape[1]) \
#                 & (xyzs_m[:,1] >= 0) & (xyzs_m[:,1] < vol_m.shape[0])
#
#     vol_m_warped_to_f[xyzs_f[valid_mask, 1], xyzs_f[valid_mask, 0], xyzs_f[valid_mask, 2]] = \
#     vol_m[xyzs_m[valid_mask, 1], xyzs_m[valid_mask, 0], xyzs_m[valid_mask, 2]]
#
#     return vol_m_warped_to_f


###################################################################

def get_structure_centroids(vol_bbox_dict=None, vol_origin_dict=None, vol_dict=None):
    """
    Compute structure centroids.
    """

    structure_centroids = {}
    if vol_bbox_dict is not None:
        for label, (v, bb) in vol_bbox_dict.iteritems():
            xmin, _, ymin, _, zmin, _ = bb
            ym, xm, zm = np.mean(np.nonzero(v), axis=1)
            structure_centroids[label] = (xm+xmin, ym+ymin, zm+zmin)
    elif vol_origin_dict is not None:
        for label, (v, o) in vol_origin_dict.iteritems():
            xmin, ymin, zmin = o
            ym, xm, zm = np.mean(np.nonzero(v), axis=1)
            structure_centroids[label] = (xm+xmin, ym+ymin, zm+zmin)
    elif vol_dict is not None:
        for label, v in vol_dict.iteritems():
            print np.where(v)
            ym, xm, zm = np.mean(np.nonzero(v), axis=1)
            structure_centroids[label] = (xm, ym, zm)
    return structure_centroids


def get_centroid_3d(v):
    """
    Compute the centroids of volumes.

    Args:
        v: volumes as 3d array, or dict of volumes, or dict of (volume, origin))
    """

    if isinstance(v, dict):
        centroids = {}
        for n, s in v.iteritems():
            if isinstance(s, tuple): # volume, origin_or_bbox
                vol, origin_or_bbox = s
                if len(origin_or_bbox) == 3:
                    origin = origin_or_bbox
                elif len(origin_or_bbox) == 6:
                    bbox = origin_or_bbox
                    origin = bbox[[0,2,4]]
                else:
                    raise
                centroids[n] = np.mean(np.where(vol), axis=1)[[1,0,2]] + origin
            else: # volume
                centroids[n] = np.mean(np.where(s), axis=1)[[1,0,2]]
        return centroids
    else:
        return np.mean(np.where(v), axis=1)[[1,0,2]]

def compute_midpoints(structure_centroids):
    """
    Compute the mid-points of each structure.

    Args:
        structure_centroids (dict of dict): {sided name: (centroid x,y,z)}

    Returns:
        dict of (3,)-array: {unsided name: mid-point}
    """

    from metadata import all_known_structures, singular_structures, convert_to_left_name, convert_to_right_name

    midpoints = {}
    for s in all_known_structures:
        if s in singular_structures:
            if s not in structure_centroids:
                continue
            c = np.array(structure_centroids[s])
        else:
            sl = convert_to_left_name(s)
            sr = convert_to_right_name(s)
            if sl not in structure_centroids or sr not in structure_centroids:
                continue
            c = (np.array(structure_centroids[sl]) + np.array(structure_centroids[sr]))/2
        midpoints[s] = c
    return midpoints

def eulerAnglesToRotationMatrix(theta):
    """
    Calculates Rotation Matrix given euler angles.
    """

    R_x = np.array([[1,         0,                  0                   ],
                    [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
                    [0,         np.sin(theta[0]), np.cos(theta[0])  ]
                    ])
    R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
                    ])
    R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                    [np.sin(theta[2]),    np.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R


def rotationMatrixToEulerAngles(R) :
    """
    Calculates rotation matrix to euler angles
    The result is the same as MATLAB except the order
    of the euler angles ( x and z are swapped ).

    Ref: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    """
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if  not singular :
        x = np.arctan2(R[2,1] , R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else :
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def plot_centroid_means_and_covars_3d(instance_centroids,
                                        nominal_locations,
                                        canonical_centroid=None,
                                        canonical_normal=None,
                                      cov_mat_allStructures=None,
                                      radii_allStructures=None,
                                      ellipsoid_matrix_allStructures=None,
                                     colors=None,
                                     show_canonical_centroid=True,
                                     xlim=(0,400),
                                     ylim=(0,400),
                                     zlim=(0,400),
                                     xlabel='x',
                                     ylabel='y',
                                     zlabel='z',
                                     title='Centroid means and covariances'):
    """
    Plot the means and covariance matrices in 3D.
    All coordinates are relative to cropped MD589.

    Args:
        instance_centroids (dict {str: list of (3,)-arrays}): centroid coordinate of each instance relative to the canonical centroid
        nominal_locations (dict {str: (3,)-arrays}): the average centroid for all instance centroid of every structure relative to canonical centroid
        canonical_centroid ((3,)-arrays): coordinate of the origin of canonical frame, defined relative to atlas
        canonical_normal ((3,)-arrays): normal vector of the mid-sagittal plane. The mid-sagittal plane is supppose to pass the `canonical_centroid`.
        cov_mat_allStructures (dict {str: (3,3)-ndarray}): covariance_matrices
        radii_allStructures (dict {str: (3,)-ndarray}): radius of each axis
        ellipsoid_matrix_allStructures (dict {str: (3,3)-ndarray}): Of each matrix, each row is a eigenvector of the corresponding covariance matrix
        colors (dict {str: 3-tuple}): for example: {'7N': (1,0,0)}.
    """

    # Load ellipsoid: three radius and axes.
    if radii_allStructures is not None and ellipsoid_matrix_allStructures is not None:
        pass
    elif cov_mat_allStructures is not None:
        radii_allStructures, ellipsoid_matrix_allStructures = compute_ellipsoid_from_covar(cov_mat_allStructures)
    else:
        _, radii_allStructures, ellipsoid_matrix_allStructures = compute_covar_from_instance_centroids(instance_centroids)

    # Plot in 3D.

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from metadata import name_unsided_to_color, convert_to_original_name

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')

    if colors is None:
        colors = {name_s: (0,0,1) for name_s in instance_centroids}

    for name_s, centroids in instance_centroids.iteritems():
    #     if name_s == '7N_L' or name_s == '7N_R':

        if canonical_centroid is None:
            centroids2 = np.array(centroids)
        else:
            centroids2 = np.array(centroids) + canonical_centroid

        ax.scatter(centroids2[:,0], centroids2[:,1], centroids2[:,2],
                   marker='o', s=100, alpha=.1, color=colors[name_s])

        if canonical_centroid is None:
            c = nominal_locations[name_s]
        else:
            c = nominal_locations[name_s] + canonical_centroid

        ax.scatter(c[0], c[1], c[2],
                   color=colors[name_s], marker='*', s=100)

        # Plot uncerntainty ellipsoids
        u = np.linspace(0.0, 2.0 * np.pi, 100)
        v = np.linspace(0.0, np.pi, 100)
        x = radii_allStructures[name_s][0] * np.outer(np.cos(u), np.sin(v))
        y = radii_allStructures[name_s][1] * np.outer(np.sin(u), np.sin(v))
        z = radii_allStructures[name_s][2] * np.outer(np.ones_like(u), np.cos(v))
        for i in range(len(u)):
            for j in range(len(v)):
                [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], ellipsoid_matrix_allStructures[name_s]) + c

    #     ax.plot_surface(x, y, z, color='b')
        ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color='b', alpha=0.2)

    if canonical_centroid is not None:
        if show_canonical_centroid:
            ax.scatter(canonical_centroid[0], canonical_centroid[1], canonical_centroid[2],
               color=(0,0,0), marker='^', s=200)

        # Plot mid-sagittal plane
        if canonical_normal is not None:
            canonical_midplane_xx, canonical_midplane_yy = np.meshgrid(range(xlim[0], xlim[1], 100), range(ylim[0], ylim[1], 100), indexing='xy')
            canonical_midplane_z = -(canonical_normal[0]*(canonical_midplane_xx-canonical_centroid[0]) + \
            canonical_normal[1]*(canonical_midplane_yy-canonical_centroid[1]) + \
            canonical_normal[2]*(-canonical_centroid[2]))/canonical_normal[2]
            ax.plot_surface(canonical_midplane_xx, canonical_midplane_yy, canonical_midplane_z, alpha=.1)
    else:
        sys.stderr.write("canonical_centroid not provided. Skip plotting cenonical centroid and mid-sagittal plane.\n")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    # ax.set_axis_off()
    ax.set_xlim3d([xlim[0], xlim[1]]);
    ax.set_ylim3d([ylim[0], ylim[1]]);
    ax.set_zlim3d([zlim[0], zlim[1]]);
    # ax.view_init(azim = 90 + 20,elev = 0 - 20)
    ax.view_init(azim = 270, elev = 0)

    # Hide y-axis (https://stackoverflow.com/questions/12391271/matplotlib-turn-off-z-axis-only-in-3-d-plot)
    ax.w_yaxis.line.set_lw(0.)
    ax.set_yticks([])

    ax.set_aspect(1.0)
    ax.set_title(title)
    plt.legend()
    plt.show()

def compute_ellipsoid_from_covar(covar_mat):
    """
    Compute the ellipsoid (three radii and three axes) of each structure from covariance matrices.
    Radii are the square root of the singular values (or 1 sigma).

    Returns:
        dict {str: (3,)-ndarray}: radius of each axis
        dict {str: (3,3)-ndarray}: Of each matrix, each row is a eigenvector of the corresponding covariance matrix
    """

    radii_allStructures = {}
    ellipsoid_matrix_allStructures = {}
    for name_s, cov_mat in sorted(covar_mat.items()):
        u, s, vt = np.linalg.svd(cov_mat)
    #     print name_s, u[:,0], u[:,1], u[:,2],
        radii_allStructures[name_s] = np.sqrt(s)
        ellipsoid_matrix_allStructures[name_s] = vt
    return radii_allStructures, ellipsoid_matrix_allStructures

def compute_covar_from_instance_centroids(instance_centroids):
    """
    Compute the covariance matrices based on instance centroids.

    Args:
        instance_centroids: dict {str: list of (3,)-arrays}

    Returns:
        dict {str: (3,3)-ndarray}: covariance_matrices
        dict {str: (3,)-ndarray}: radius of each axis
        dict {str: (3,3)-ndarray}: Of each matrix, each row is a eigenvector of the corresponding covariance matrix
    """

    cov_mat_allStructures = {}
    radii_allStructures = {}
    ellipsoid_matrix_allStructures = {}
    for name_s, centroids in sorted(instance_centroids.items()):
        centroids2 = np.array(centroids)
        cov_mat = np.cov(centroids2.T)
        cov_mat_allStructures[name_s] = cov_mat
        u, s, vt = np.linalg.svd(cov_mat)
    #     print name_s, u[:,0], u[:,1], u[:,2],
        radii_allStructures[name_s] = np.sqrt(s)
        ellipsoid_matrix_allStructures[name_s] = vt

    return cov_mat_allStructures, radii_allStructures, ellipsoid_matrix_allStructures


def find_contour_points_3d(labeled_volume, along_direction, positions=None, sample_every=10):
    """
    Find the cross-section contours given a (binary?) volume.

    Args:
        labeled_volume (3D ndarray of int): integer-labeled volume.
        along_direction (str): x/coronal, y/horizontal or z/sagittal.
        positions (None or list of int): if None, find contours at all positions of input volume, from 0 to the depth of volume.

    Returns:
        dict {int: (n,2)-ndarray}: contours. {voxel position: contour vertices (second dim, first dim)}.
        For example, If `along_direction=y`, returns (z,x); if direction=x, returns (z,y).
    """

    import multiprocessing
    # nproc = multiprocessing.cpu_count()
    nproc = 1

    if along_direction == 'z' or along_direction == 'sagittal':
        if positions is None:
            positions = range(0, labeled_volume.shape[2])
    elif along_direction == 'x' or along_direction == 'coronal':
        if positions is None:
            positions = range(0, labeled_volume.shape[1])
    elif along_direction == 'y' or along_direction == 'horizontal':
        if positions is None:
            positions = range(0, labeled_volume.shape[0])

    def find_contour_points_slice(p):
        """
        Args:
            p (int): position
        """
        if along_direction == 'x':
            if p < 0 or p >= labeled_volume.shape[1]:
                return
            vol_slice = labeled_volume[:, p, :]
        elif along_direction == 'coronal':
            if p < 0 or p >= labeled_volume.shape[1]:
                return
            vol_slice = labeled_volume[:, p, ::-1]
        elif along_direction == 'y':
            if p < 0 or p >= labeled_volume.shape[0]:
                return
            vol_slice = labeled_volume[p, :, :]
        elif along_direction == 'horizontal':
            if p < 0 or p >= labeled_volume.shape[0]:
                return
            vol_slice = labeled_volume[p, :, ::-1].T
        elif along_direction == 'z' or along_direction == 'sagittal':
            if p < 0 or p >= labeled_volume.shape[2]:
                return
            vol_slice = labeled_volume[:, :, p]
        else:
            raise

        cnts = find_contour_points(vol_slice.astype(np.uint8), sample_every=sample_every)
        if len(cnts) == 0 or 1 not in cnts:
            # sys.stderr.write('No contour of reconstructed volume is found at position %d.\n' % p)
            return
        else:
            if len(cnts[1]) > 1:
                sys.stderr.write('%s contours of reconstructed volume is found at position %d (%s). Use the longest one.\n' % (len(cnts[1]), p, map(len, cnts[1])))
                cnt = np.array(cnts[1][np.argmax(map(len, cnts[1]))])
            else:
                cnt = np.array(cnts[1][0])
            if len(cnt) <= 2:
                sys.stderr.write('contour has less than three vertices. Ignore.\n')
                return
            else:
                return cnt

    pool = Pool(nproc)
    contours = dict(zip(positions, pool.map(find_contour_points_slice, positions)))
    pool.close()
    pool.join()

    contours = {p: cnt for p, cnt in contours.iteritems() if cnt is not None}

    return contours

def find_contour_points(labelmap, sample_every=10, min_length=0):
    """
    Find contour coordinates.

    Args:
        labelmap (2d array of int): integer-labeled 2D image
        sample_every (int): can be interpreted as distance between points.
        min_length (int): contours with fewer vertices are discarded.
        This argument is being deprecated because the vertex number does not
        reliably measure actual contour length in microns.
        It is better to leave this decision to calling routines.
    Returns:
        a dict of lists: {label: list of contours each consisting of a list of (x,y) coordinates}
    """

    padding = 5

    if np.count_nonzero(labelmap) == 0:
        # sys.stderr.write('No contour can be found because the image is blank.\n')
        return {}

    regions = regionprops(labelmap.astype(np.int))
    contour_points = {}

    for r in regions:

        (min_row, min_col, max_row, max_col) = r.bbox

        padded = np.pad(r.filled_image, ((padding,padding),(padding,padding)),
                        mode='constant', constant_values=0)

        contours = find_contours(padded, level=.5, fully_connected='high')
        contours = [cnt.astype(np.int) for cnt in contours if len(cnt) > min_length]
        if len(contours) > 0:
#             if len(contours) > 1:
#                 sys.stderr.write('%d: region has more than one part\n' % r.label)

            contours = sorted(contours, key=lambda c: len(c), reverse=True)
            contours_list = [c-(padding, padding) for c in contours]
            contour_points[r.label] = sorted([c[np.arange(0, c.shape[0], sample_every)][:, ::-1] + (min_col, min_row)
                                for c in contours_list], key=lambda c: len(c), reverse=True)

        elif len(contours) == 0:
#             sys.stderr.write('no contour is found\n')
            continue

    #         viz = np.zeros_like(r.filled_image)
    #         viz[pts_sampled[:,0], pts_sampled[:,1]] = 1
    #         plt.imshow(viz, cmap=plt.cm.gray);
    #         plt.show();

    return contour_points

def draw_alignment(warped_atlas, fixed_volumes, level_spacing=10, zs=None, ncols=5, structures=None, colors=None,
                  markers=None):
    """
    Args:
        structures (list of str or int): structures to show; the type matches the keys of `warped_atlas` and `fixed_volumes`.
        colors (dict ): {structure id: (3,)-array}
        markers ((n,3)-array): coordinates of markers
    """

    ydim_f, xdim_f, zdim_f = fixed_volumes.values()[0].shape

    aspect_ratio = float(xdim_f)/ydim_f # width / height

    if zs is None:
        zs = np.arange(0, zdim, level_spacing)
    n = len(zs)

    nrows = int(np.ceil(len(zs) / float(ncols)))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, squeeze=True,
                             figsize=(ncols*5*aspect_ratio, nrows*5))
    axes = axes.flatten()

    for zi in range(len(axes)):
        if zi >= n:
            axes[zi].axis('off');
        else:
            z = zs[zi]

            viz = np.zeros((ydim_f, xdim_f, 3), np.uint8)

            for l in fixed_volumes.keys():
                if l not in structures:
                    continue
                zslice = fixed_volumes[l][..., z]
                contours = find_contour_points(zslice)
                if len(contours) == 0:
                    continue

                for cnt in contours[1]:
                    cv2.polylines(viz, pts=[cnt.astype(np.int)], isClosed=True,
                              thickness=1,
                              color=colors[l])

            if markers is not None:
                markers_on_slice = markers[(markers[...,2] > z-1) & (markers[...,2] < z+1)]
                for m in markers_on_slice:
                    cv2.circle(viz, center=tuple(m[:2].astype(np.int)), radius=2, color=(255,255,255), thickness=-1)

            ####################

            cutoff_level = 0.5

            for l in warped_atlas.keys():
                if l not in structures:
                    continue
                zslice = warped_atlas[l][..., z]
                contours = find_contour_points(zslice > cutoff_level)
                if len(contours) == 0:
                    continue

                for cnt in contours[1]:
                    cv2.polylines(viz, pts=[cnt.astype(np.int)], isClosed=True,
                              thickness=1,
                             color=colors[l])

            ######################

            axes[zi].imshow(viz)
            axes[zi].set_title("z=%d" % z)
            axes[zi].set_xticks([]);
            axes[zi].set_yticks([]);

    plt.show()


def get_grid_mesh_coordinates(bbox, spacings=(1,1,1), dot_spacing=1, include_borderline=True):
    """
    Get the coordinates of grid lines.

    Args:
        spacings (3-tuple): spacing between grid lines in x,y and z.
        dot_spacing (int): the spacing between dots if broken lines are desired.

    Returns:
        (n,3)-array
    """

    xmin,xmax,ymin,ymax,zmin,zmax = bbox

    xdim, ydim, zdim = (xmax+1-xmin, ymax+1-ymin, zmax+1-zmin)

    xs = np.arange(0, xdim, spacings[0])
    ys = np.arange(0, ydim, spacings[1])
    zs = np.arange(0, zdim, spacings[2])

    vol = np.zeros((ydim, xdim, zdim), np.bool)
    xs = xs.astype(np.int)
    ys = ys.astype(np.int)
    zs = zs.astype(np.int)
    xs = xs[(xs >= 0) & (xs < xdim)]
    ys = ys[(ys >= 0) & (ys < ydim)]
    zs = zs[(zs >= 0) & (zs < zdim)]
    if include_borderline:
        if 0 not in xs:
            xs = np.r_[0, xs, xdim-1]
        else:
            xs = np.r_[xs, xdim-1]
        if 0 not in ys:
            ys = np.r_[0, ys, ydim-1]
        else:
            ys = np.r_[ys, ydim-1]
        if 0 not in zs:
            zs = np.r_[0, zs, zdim-1]
        else:
            zs = np.r_[zs, zdim-1]
    for y in ys:
        vol[y, xs, ::dot_spacing] = 1
        vol[y, ::dot_spacing, zs] = 1
    for x in xs:
        vol[ys, x, ::dot_spacing] = 1
        vol[::dot_spacing, x, zs] = 1
    for z in zs:
        vol[ys, ::dot_spacing, z] = 1
        vol[::dot_spacing, xs, z] = 1

    ys, xs, zs = np.nonzero(vol)

    return np.c_[xs, ys, zs] + (xmin,ymin,zmin)


def get_grid_mesh_volume(xs, ys, zs, vol_shape, s=1, include_borderline=True):
    """
    Get a boolean volume with grid lines set to True.

    Args:
        s (int): the spacing between dots if broken lines are desired.

    Returns:
        3D array of boolean
    """

    xs, ys, zs = get_grid_mesh_coordinates(**locals())

    xdim, ydim, zdim = vol_shape
    vol = np.zeros((ydim, xdim, zdim), np.bool)
    xs = xs.astype(np.int)
    ys = ys.astype(np.int)
    zs = zs.astype(np.int)
    xs = xs[(xs >= 0) & (xs < xdim)]
    ys = ys[(ys >= 0) & (ys < ydim)]
    zs = zs[(zs >= 0) & (zs < zdim)]
    if include_borderline:
        if 0 not in xs:
            xs = np.r_[0, xs, xdim-1]
        else:
            xs = np.r_[xs, xdim-1]
        if 0 not in ys:
            ys = np.r_[0, ys, ydim-1]
        else:
            ys = np.r_[ys, ydim-1]
        if 0 not in zs:
            zs = np.r_[0, zs, zdim-1]
        else:
            zs = np.r_[zs, zdim-1]
    for y in ys:
        vol[y, xs, ::s] = 1
        vol[y, ::s, zs] = 1
    for x in xs:
        vol[ys, x, ::s] = 1
        vol[::s, x, zs] = 1
    for z in zs:
        vol[ys, ::s, z] = 1
        vol[::s, xs, z] = 1

    return vol

def get_grid_point_volume(xs, ys, zs, vol_shape, return_nz=False):
    """
    Get a boolean volume with grid point set to True.
    """

    xdim, ydim, zdim = vol_shape
    vol = np.zeros((ydim, xdim, zdim), np.bool)
    xs = xs.astype(np.int)
    ys = ys.astype(np.int)
    zs = zs.astype(np.int)
    xs = xs[(xs >= 0) & (xs < xdim)]
    ys = ys[(ys >= 0) & (ys < ydim)]
    zs = zs[(zs >= 0) & (zs < zdim)]
    gp_xs, gp_ys, gp_zs = np.meshgrid(xs, ys, zs, indexing='ij')
    gp_xyzs = np.c_[gp_xs.flatten(), gp_ys.flatten(), gp_zs.flatten()]
    vol[gp_xyzs[:,1], gp_xyzs[:,0], gp_xyzs[:,2]] = 1

    if return_nz:
        return vol, gp_xyzs
    else:
        return vol

def return_gridline_points_v2(xdim, ydim, spacing, z):
    xs = np.arange(0, xdim, spacing)
    ys = np.arange(0, ydim, spacing)
    return return_gridline_points(xs, ys, z, xdim, ydim)

def return_gridline_points(xs, ys, z, w, h):
    grid_points = np.array([(x,y,z) for x in range(w) for y in ys] + [(x,y,z) for x in xs for y in range(h)])
    return grid_points

def consolidate(params, centroid_m=(0,0,0), centroid_f=(0,0,0)):
    """
    Convert the set (parameter, centroid m, centroid f) to a single matrix.

    Args:
        params ((12,)-array):
        centroid_m ((3,)-array):
        centroid_f ((3,)-array):

    Returns:
        ((4,4)-array)
    """
    G = params.reshape((3,4))
    R = G[:3,:3]
    t = - np.dot(R, centroid_m) + G[:3,3] + centroid_f
    return np.vstack([np.c_[R,t], [0,0,0,1]])

def jaccard_masks(m1, m2, wrt_min=False):
    """
    Args:
        m1 (ndarray of boolean):
        m2 (ndarray of boolean):
        wrt_min (bool): If true, the denominator is the minimum between two masks.
    """
    if wrt_min:
        return np.count_nonzero(m1 & m2) / float(min(np.count_nonzero(m1), np.count_nonzero(m2)))
    else:
        return np.count_nonzero(m1 & m2) / float(np.count_nonzero(m1 | m2))

def dice(hm, hf):
    """
    Compute the Dice similarity index between two boolean images. The value ranges between 0 and 1.
    """
    return 2 * np.count_nonzero(hm & hf) / float(np.count_nonzero(hm) + np.count_nonzero(hf))

########################################################################################

def crop_volume_to_minimal(vol, origin=(0,0,0), margin=0, return_origin_instead_of_bbox=True):
    """
    Returns:
        (nonzero part of volume, origin of cropped volume)
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bbox_3d(vol)
    xmin = max(0, xmin - margin)
    ymin = max(0, ymin - margin)
    zmin = max(0, zmin - margin)
    xmax = min(vol.shape[1]-1, xmax + margin)
    ymax = min(vol.shape[0]-1, ymax + margin)
    zmax = min(vol.shape[2]-1, zmax + margin)

    if return_origin_instead_of_bbox:
        return vol[ymin:ymax+1, xmin:xmax+1, zmin:zmax+1], np.array(origin) + (xmin,ymin,zmin)
    else:
        return vol[ymin:ymax+1, xmin:xmax+1, zmin:zmax+1], np.array(origin)[[0,0,1,1,2,2]] + (xmin,xmax,ymin,ymax,zmin,zmax)

def get_overall_bbox(vol_bbox_tuples=None, bboxes=None):
    if bboxes is None:
        bboxes = np.array([b for v, b in vol_bbox_tuples])
    xmin, ymin, zmin = np.min(bboxes[:, [0,2,4]], axis=0)
    xmax, ymax, zmax = np.max(bboxes[:, [1,3,5]], axis=0)
    bbox = xmin, xmax, ymin, ymax, zmin, zmax
    return bbox

def crop_and_pad_volumes(out_bbox=None, vol_bbox_dict=None, vol_bbox_tuples=None, vol_bbox=None):
    """
    Args:
        out_bbox ((6,)-array): the output bounding box, must use the same reference system as the vol_bbox input.
        vol_bbox_dict (dict {key: (vol, bbox)})
        vol_bbox_tuples (list of (vol, bbox) tuples)

    Returns:
        list of 3d arrays or dict {structure name: 3d array}
    """

    if vol_bbox is not None:
        if isinstance(vol_bbox, dict):
            vols = {l: crop_and_pad_volume(v, in_bbox=b, out_bbox=out_bbox) for l, (v, b) in volumes.iteritems()}
        elif isinstance(vol_bbox, list):
            vols = [crop_and_pad_volume(v, in_bbox=b, out_bbox=out_bbox) for (v, b) in volumes]
        else:
            raise
    else:
        if vol_bbox_tuples is not None:
            vols = [crop_and_pad_volume(v, in_bbox=b, out_bbox=out_bbox) for (v, b) in vol_bbox_tuples]
        elif vol_bbox_dict is not None:
            vols = {l: crop_and_pad_volume(v, in_bbox=b, out_bbox=out_bbox) for l, (v, b) in vol_bbox_dict.iteritems()}
        else:
            raise

    return vols

def convert_vol_bbox_dict_to_overall_vol(vol_bbox_dict=None, vol_bbox_tuples=None, vol_origin_dict=None):
    """
    Must provide exactly one of the three choices of arguments.
    `bbox` or `origin` can be provided as float, but will be casted as integer before cropping and padding.

    Args:
        vol_bbox_dict (dict {key: 3d-array of float32, 6-tuple of float}): represents {name_s: (vol, bbox)}
        vol_origin_dict (dict {key: 3d-array of float32, 3-tuple of float}): represents {name_s: (vol, origin)}

    Returns:
        (list or dict of 3d arrays, (6,)-ndarray of int): (volumes in overall coordinate system, the common overall bounding box)
    """

    if vol_origin_dict is not None:
        vol_bbox_dict = {k: (v, (o[0], o[0]+v.shape[1]-1, o[1], o[1]+v.shape[0]-1, o[2], o[2]+v.shape[2]-1)) for k,(v,o) in vol_origin_dict.iteritems()}

    if vol_bbox_dict is not None:
        volume_bbox = np.round(get_overall_bbox(vol_bbox_tuples=vol_bbox_dict.values())).astype(np.int)
        volumes = crop_and_pad_volumes(out_bbox=volume_bbox, vol_bbox_dict=vol_bbox_dict)
    else:
        volume_bbox = np.round(get_overall_bbox(vol_bbox_tuples=vol_bbox_tuples)).astype(np.int)
        volumes = crop_and_pad_volumes(out_bbox=volume_bbox, vol_bbox_tuples=vol_bbox_tuples)
    return volumes, np.array(volume_bbox)


def crop_and_pad_volume(in_vol, in_bbox=None, in_origin=(0,0,0), out_bbox=None):
    """
    Crop and pad an volume.

    in_vol and in_bbox together define the input volume in a underlying space.
    out_bbox then defines how to crop the underlying space, which generates the output volume.

    Args:
        in_bbox ((6,) array): the bounding box that the input volume is defined on. If None, assume origin is at (0,0,0) of the input volume.
        in_origin ((3,) array): the input volume origin coordinate in the space. Used only if in_bbox is not specified. Default is (0,0,0), meaning the input volume is located at the origin of the underlying space.
        out_bbox ((6,) array): the bounding box that the output volume is defined on. If not given, each dimension is from 0 to the max reach of any structure.

    Returns:
        3d-array: cropped/padded volume
    """

    if in_bbox is None:
        assert in_origin is not None
        in_xmin, in_ymin, in_zmin = in_origin
        in_xmax = in_xmin + in_vol.shape[1] - 1
        in_ymax = in_ymin + in_vol.shape[0] - 1
        in_zmax = in_zmin + in_vol.shape[2] - 1
    else:
        in_bbox = np.array(in_bbox).astype(np.int)
        in_xmin, in_xmax, in_ymin, in_ymax, in_zmin, in_zmax = in_bbox
    in_xdim = in_xmax - in_xmin + 1
    in_ydim = in_ymax - in_ymin + 1
    in_zdim = in_zmax - in_zmin + 1
        # print 'in', in_xdim, in_ydim, in_zdim

    if out_bbox is None:
        out_xmin = 0
        out_ymin = 0
        out_zmin = 0
        out_xmax = in_xmax
        out_ymax = in_ymax
        out_zmax = in_zmax
    elif isinstance(out_bbox, np.ndarray) and out_bbox.ndim == 3:
        out_xmin, out_xmax, out_ymin, out_ymax, out_zmin, out_zmax = (0, out_bbox.shape[1]-1, 0, out_bbox.shape[0]-1, 0, out_bbox.shape[2]-1)
    else:
        out_bbox = np.array(out_bbox).astype(np.int)
        out_xmin, out_xmax, out_ymin, out_ymax, out_zmin, out_zmax = out_bbox
    out_xdim = out_xmax - out_xmin + 1
    out_ydim = out_ymax - out_ymin + 1
    out_zdim = out_zmax - out_zmin + 1

    # print out_xmin, out_xmax, out_ymin, out_ymax, out_zmin, out_zmax

    if out_xmin > in_xmax or out_xmax < in_xmin or out_ymin > in_ymax or out_ymax < in_ymin or out_zmin > in_zmax or out_zmax < in_zmin:
        return np.zeros((out_ydim, out_xdim, out_zdim), np.int)

    if out_xmax > in_xmax:
        in_vol = np.pad(in_vol, pad_width=[(0,0),(0, out_xmax-in_xmax),(0,0)], mode='constant', constant_values=0)
        # print 'pad x'
    if out_ymax > in_ymax:
        in_vol = np.pad(in_vol, pad_width=[(0, out_ymax-in_ymax),(0,0),(0,0)], mode='constant', constant_values=0)
        # print 'pad y'
    if out_zmax > in_zmax:
        in_vol = np.pad(in_vol, pad_width=[(0,0),(0,0),(0, out_zmax-in_zmax)], mode='constant', constant_values=0)
        # print 'pad z'

    out_vol = np.zeros((out_ydim, out_xdim, out_zdim), in_vol.dtype)
    ymin = max(in_ymin, out_ymin)
    xmin = max(in_xmin, out_xmin)
    zmin = max(in_zmin, out_zmin)
    ymax = out_ymax
    xmax = out_xmax
    zmax = out_zmax
    # print 'in_vol', np.array(in_vol.shape)[[1,0,2]]
    # print xmin, xmax, ymin, ymax, zmin, zmax
    # print xmin-in_xmin, xmax+1-in_xmin
    # assert ymin >= 0 and xmin >= 0 and zmin >= 0
    out_vol[ymin-out_ymin:ymax+1-out_ymin,
            xmin-out_xmin:xmax+1-out_xmin,
            zmin-out_zmin:zmax+1-out_zmin] = in_vol[ymin-in_ymin:ymax+1-in_ymin, xmin-in_xmin:xmax+1-in_xmin, zmin-in_zmin:zmax+1-in_zmin]

    assert out_vol.shape[1] == out_xdim
    assert out_vol.shape[0] == out_ydim
    assert out_vol.shape[2] == out_zdim

    return out_vol

########################################################################################

def crop_large_image(fp, bbox):
    """
    Args:
        fp (str): image file path
        bbox (4-tuple of int): xmin,xmax,ymin,ymax

    Returns:
        region_img (2d-array)
    """

    xmin,xmax,ymin,ymax = bbox
    h = ymax+1-ymin
    w = xmax+1-xmin

    execute_command( """convert \"%(im_fp)s\" -crop %(w)dx%(h)d+%(x)d+%(y)d /tmp/tmp.tif""" % \
               {'im_fp': fp, 'w':w, 'h':h, 'x':xmin, 'y':ymin})
    return imread('/tmp/tmp.tif', -1)

def rescale_intensity_v2(im, low, high):
    """
    Linearly map `low` to 0 and `high` to 255.

    Args:
        im (2d array of float): input image.
    """

    from skimage.exposure import rescale_intensity, adjust_gamma
    if low > high:
        im_out = rescale_intensity(low-im.astype(np.float), (0, low-high), (0, 255)).astype(np.uint8)
    else:
        im_out = rescale_intensity(im.astype(np.float), (low, high), (0, 255)).astype(np.uint8)
    return im_out


def visualize_blob_contour(binary_img, bg_img):
    """
    Args:
        binary_img: the binary image
        rgb_img: the background image

    Returns:
        Contoured image.
    """
    from registration_utilities import find_contour_points

    viz = gray2rgb(bg_img)
    for cnt in find_contour_points(binary_img)[1]:
        cv2.polylines(viz, [cnt.astype(np.int)], isClosed=True, color=(255,0,0), thickness=2)
    return viz


def shell_escape(s):
    """
    Escape a string (treat it as a single complete string) in shell commands.
    """
    from tempfile import mkstemp
    fd, path = mkstemp()
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(s)
        cmd = r"""cat %s | sed -e "s/'/'\\\\''/g; 1s/^/'/; \$s/\$/'/" """ % path
        escaped_str = check_output(cmd, shell=True)
    finally:
        os.remove(path)

    return escaped_str

def plot_histograms(hists, bins, titles=None, ncols=4, xlabel='', ylabel='', suptitle='', normalize=False, cellsize=(2, 1.5), **kwargs):
    """
    cellsize: (w,h) for each cell
    """

    if isinstance(hists, dict):
        titles = hists.keys()
        hists = hists.values()

    if normalize:
        hists = hists/np.sum(hists, axis=1).astype(np.float)[:,None]

    if titles is None:
        titles = ['' for _ in range(len(hists))]

    n = len(hists)
    nrows = int(np.ceil(n/float(ncols)))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, \
                            figsize=(ncols*cellsize[0], nrows*cellsize[1]), **kwargs)
    axes = axes.flatten()

    width = np.abs(np.mean(np.diff(bins)))*.8

    c = 0
    for name_u, h in zip(titles, hists):

        axes[c].bar(bins, h/float(h.sum()), width=width);
        axes[c].set_xlabel(xlabel);
        axes[c].set_ylabel(ylabel);
        axes[c].set_title(name_u);
        c += 1

    for i in range(c, len(axes)):
        axes[i].axis('off')

    plt.suptitle(suptitle);
    plt.tight_layout();
    plt.show()

def save_pickle(obj, fp):
    with open(fp, 'w') as f:
        pickle.dump(obj, f)

def save_json(obj, fp):
    with open(fp, 'w') as f:
        # numpy array is not JSON serializable; have to convert them to list.
        if isinstance(obj, dict):
            obj = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in obj.iteritems()}
        json.dump(obj, f)

def load_json(fp):
    with open(fp, 'r') as f:
        return json.load(f)

def load_pickle(fp):
    with open(fp, 'r') as f:
        obj = pickle.load(f)

    return obj

def one_liner_to_arr(line, func):
    return np.array(map(func, line.strip().split()))

def array_to_one_liner(arr):
    return ' '.join(map(str, arr)) + '\n'

def show_progress_bar(min, max):
    bar = FloatProgress(min=min, max=max)
    display(bar)
    return bar

from enum import Enum

class PolygonType(Enum):
    CLOSED = 'closed'
    OPEN = 'open'
    TEXTURE = 'textured'
    TEXTURE_WITH_CONTOUR = 'texture with contour'
    DIRECTION = 'directionality'

def create_parent_dir_if_not_exists(fp):
    create_if_not_exists(os.path.dirname(fp))

def create_if_not_exists(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            sys.stderr.write('%s\n' % e);

    return path

def execute_command(cmd, stdout=None, stderr=None):
    sys.stderr.write(cmd + '\n')

    # try:
#     from errand_boy.transports.unixsocket import UNIXSocketTransport
#     errand_boy_transport = UNIXSocketTransport()
#     stdout, stderr, retcode = errand_boy_transport.run_cmd(cmd)

#     print stdout
#     print stderr

    # import os
    # retcode = os.system(cmd)
    retcode = call(cmd, shell=True, stdout=stdout, stderr=stderr)
    sys.stderr.write('return code: %d\n' % retcode)

    # if retcode < 0:
    #     print >>sys.stderr, "Child was terminated by signal", -retcode
    # else:
    #     print >>sys.stderr, "Child returned", retcode
    # except OSError as e:
    #     print >>sys.stderr, "Execution failed:", e
    #     raise e

def draw_arrow(image, p, q, color, arrow_magnitude=9, thickness=5, line_type=8, shift=0):
    # adapted from http://mlikihazar.blogspot.com.au/2013/02/draw-arrow-opencv.html

    import cv2

    # draw arrow tail
    cv2.line(image, p, q, color, thickness, line_type, shift)
    # calc angle of the arrow
    angle = np.arctan2(p[1]-q[1], p[0]-q[0])
    # starting point of first line of arrow head
    p = (int(q[0] + arrow_magnitude * np.cos(angle + np.pi/4)),
    int(q[1] + arrow_magnitude * np.sin(angle + np.pi/4)))
    # draw first half of arrow head
    cv2.line(image, p, q, color, thickness, line_type, shift)
    # starting point of second line of arrow head
    p = (int(q[0] + arrow_magnitude * np.cos(angle - np.pi/4)),
    int(q[1] + arrow_magnitude * np.sin(angle - np.pi/4)))
    # draw second half of arrow head
    cv2.line(image, p, q, color, thickness, line_type, shift)


def save_hdf_v2(data, fn, key='data', mode='w'):
    """
    Save data as a hdf file.
    If data is dict of dict, convert to DataFrame before saving as hdf.
    If data is dict of elementary items, convert to pandas.Series before saving as hdf.

    Args:
        data (pandas.DataFrame, dict or dict of dict)
        mode (str): if 'w', overwrite original content. If 'a', append.
    """

    import pandas
    create_parent_dir_if_not_exists(fn)
    if isinstance(data, pandas.DataFrame):
        data.to_hdf(fn, key=key, mode=mode) # important to set mode='w', default is 'a' (append)
    elif isinstance(data, dict):
        if isinstance(data.values()[0], dict): # dict of dict
            pandas.DataFrame(data).T.to_hdf(fn, key=key, mode='w')
        else:
            pandas.Series(data=data).to_hdf(fn, key, mode='w')

def load_hdf_v2(fn, key='data'):
    import pandas
    return pandas.read_hdf(fn, key)

def save_hdf(data, fn, complevel=9, key='data'):
    filters = Filters(complevel=complevel, complib='blosc')
    with open_file(fn, mode="w") as f:
        _ = f.create_carray('/', key, Atom.from_dtype(data.dtype), filters=filters, obj=data)

def load_hdf(fn, key='data'):
    """
    Used by loading features.
    """
    with open_file(fn, mode="r") as f:
        data = f.get_node('/'+key).read()
    return data


def unique_rows(a, return_index=True):
    # http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(b, return_index=True)
    unique_a = a[idx]
    if return_index:
        return unique_a, idx
    else:
        return unique_a

def unique_rows2(a):
    ind = np.lexsort(a.T)
    return a[np.concatenate(([True],np.any(a[ind[1:]]!=a[ind[:-1]],axis=1)))]


def order_nodes(sps, neighbor_graph, verbose=False):

    from networkx.algorithms import dfs_successors, dfs_postorder_nodes


    subg = neighbor_graph.subgraph(sps)
    d_suc = dfs_successors(subg)

    x = [(a,b) for a,b in d_suc.iteritems() if len(b) == 2]

    if verbose:
        print 'root, two_leaves', x

    if len(x) == 0:
        trav = list(dfs_postorder_nodes(subg))
    else:
        if verbose:
            print 'd_succ'
            for it in d_suc.iteritems():
                print it

        root, two_leaves = x[0]

        left_branch = []
        right_branch = []

        c = two_leaves[0]
        left_branch.append(c)
        while c in d_suc:
            c = d_suc[c][0]
            left_branch.append(c)

        c = two_leaves[1]
        right_branch.append(c)
        while c in d_suc:
            c = d_suc[c][0]
            right_branch.append(c)

        trav = left_branch[::-1] + [root] + right_branch

        if verbose:
            print 'left_branch', left_branch
            print 'right_branch', right_branch

    return trav

def find_score_peaks(scores, min_size = 4, min_distance=10, threshold_rel=.3, threshold_abs=0, peakedness_lim=0,
                    peakedness_radius=1, verbose=False):

    from skimage.feature import peak_local_max

    scores2 = scores.copy()
    scores2[np.isnan(scores)] = np.nanmin(scores)
    scores = scores2

    if len(scores) > min_size:

        scores_shifted = scores[min_size-1:]
        scores_shifted_positive = scores_shifted - scores_shifted.min()

        peaks_shifted = np.atleast_1d(np.squeeze(peak_local_max(scores_shifted_positive,
                                    min_distance=min_distance, threshold_abs=threshold_abs-scores_shifted.min(), exclude_border=False)))

        # print peaks_shifted

        if len(peaks_shifted) == 0:
            high_peaks_sorted = np.array([np.argmax(scores)], np.int)
            high_peaks_peakedness = np.inf

        else:
            peaks_shifted = peaks_shifted[scores_shifted[peaks_shifted] >= np.max(scores_shifted) - threshold_rel]
            peaks_shifted = np.unique(np.r_[peaks_shifted, np.argmax(scores_shifted)])

            if verbose:
                print 'raw peaks', np.atleast_1d(np.squeeze(min_size - 1 + peaks_shifted))

            if len(peaks_shifted) > 0:
                peaks = min_size - 1 + peaks_shifted
            else:
                peaks = np.array([np.argmax(scores[min_size-1:]) + min_size-1], np.int)

            peakedness = np.zeros((len(peaks),))
            for i, p in enumerate(peaks):
                nbrs = np.r_[scores[max(min_size-1, p-peakedness_radius):p], scores[p+1:min(len(scores), p+1+peakedness_radius)]]
                assert len(nbrs) > 0
                peakedness[i] = scores[p]-np.mean(nbrs)

            if verbose:
                print 'peakedness', peakedness
                print 'filtered peaks', np.atleast_1d(np.squeeze(peaks))

            high_peaks = peaks[peakedness > peakedness_lim]
            high_peaks = np.unique(np.r_[high_peaks, min_size - 1 + np.argmax(scores_shifted)])

            high_peaks_order = scores[high_peaks].argsort()[::-1]
            high_peaks_sorted = high_peaks[high_peaks_order]

            high_peaks_peakedness = np.zeros((len(high_peaks),))
            for i, p in enumerate(high_peaks):
                nbrs = np.r_[scores[max(min_size-1, p-peakedness_radius):p], scores[p+1:min(len(scores), p+1+peakedness_radius)]]
                assert len(nbrs) > 0
                high_peaks_peakedness[i] = scores[p]-np.mean(nbrs)

    else:
        high_peaks_sorted = np.array([np.argmax(scores)], np.int)
        high_peaks_peakedness = np.inf

    return high_peaks_sorted, high_peaks_peakedness


# def find_z_section_map(stack, volume_zmin, downsample_factor = 16):
#
#     section_thickness = 20 # in um
#     xy_pixel_distance_lossless = 0.46
#     xy_pixel_distance_tb = xy_pixel_distance_lossless * 32 # in um, thumbnail
#     # factor = section_thickness/xy_pixel_distance_lossless
#
#     xy_pixel_distance_downsampled = xy_pixel_distance_lossless * downsample_factor
#     z_xy_ratio_downsampled = section_thickness / xy_pixel_distance_downsampled
#
#     # build annotation volume
#     section_bs_begin, section_bs_end = section_range_lookup[stack]
#     print section_bs_begin, section_bs_end
#
#     map_z_to_section = {}
#     for s in range(section_bs_begin, section_bs_end+1):
#         for z in range(int(z_xy_ratio_downsampled*s) - volume_zmin,
#                        int(z_xy_ratio_downsampled*(s+1)) - volume_zmin + 1):
#             map_z_to_section[z] = s
#
#     return map_z_to_section

def fit_ellipse_to_points(pts):

    pts = np.array(list(pts) if isinstance(pts, set) else pts)

    c0 = pts.mean(axis=0)

    coords0 = pts - c0

    U,S,V = np.linalg.svd(np.dot(coords0.T, coords0)/coords0.shape[0])
    v1 = U[:,0]
    v2 = U[:,1]
    s1 = np.sqrt(S[0])
    s2 = np.sqrt(S[1])

    return v1, v2, s1, s2, c0


def scores_to_vote(scores):
    vals = np.unique(scores)
    d = dict(zip(vals, np.linspace(0, 1, len(vals))))
    votes = np.array([d[s] for s in scores])
    votes = votes/votes.sum()
    return votes


def display_image(vis, filename='tmp.jpg'):

    if vis.dtype != np.uint8:
        imsave(filename, img_as_ubyte(vis))
    else:
        imsave(filename, vis)

    from IPython.display import FileLink
    return FileLink(filename)


def pad_patches_to_same_size(vizs, pad_value=0, keep_center=False, common_shape=None):
    """
    If patch size is larger than common shape, crop to common shape.
    """

    # If common_shape is not given, use the largest of all data
    if common_shape is None:
        common_shape = np.max([p.shape[:2] for p in vizs], axis=0)

    dt = vizs[0].dtype
    ndim = vizs[0].ndim

    if ndim == 2:
        common_box = (pad_value*np.ones((common_shape[0], common_shape[1]))).astype(dt)
    elif ndim == 3:
        common_box = (pad_value*np.ones((common_shape[0], common_shape[1], p.shape[2]))).astype(dt)

    patches_padded = []
    for p in vizs:
        patch_padded = common_box.copy()

        if keep_center:

            top_margin = (common_shape[0] - p.shape[0])/2
            if top_margin < 0:
                ymin = 0
                ymax = common_shape[0]-1
                ymin2 = -top_margin
                ymax2 = -top_margin+common_shape[0]-1
            else:
                ymin = top_margin
                ymax = top_margin + p.shape[0] - 1
                ymin2 = 0
                ymax2 = p.shape[0]-1

            left_margin = (common_shape[1] - p.shape[1])/2
            if left_margin < 0:
                xmin = 0
                xmax = common_shape[1]-1
                xmin2 = -left_margin
                xmax2 = -left_margin+common_shape[1]-1
            else:
                xmin = left_margin
                xmax = left_margin + p.shape[1] - 1
                xmin2 = 0
                xmax2 = p.shape[1]-1

            patch_padded[ymin:ymax+1, xmin:xmax+1] = p[ymin2:ymax2+1, xmin2:xmax2+1]
#             patch_padded[top_margin:top_margin+p.shape[0], left_margin:left_margin+p.shape[1]] = p
        else:
            # assert p.shape[0] < common_shape[0] and p.shape[1] < common_shape[1]
            patch_padded[:p.shape[0], :p.shape[1]] = p

        patches_padded.append(patch_padded)

    return patches_padded

# def pad_patches_to_same_size(vizs, pad_value=0, keep_center=False, common_shape=None):
#
#     # If common_shape is not given, use the largest of all data
#     if common_shape is None:
#         common_shape = np.max([p.shape[:2] for p in vizs], axis=0)
#
#     dt = vizs[0].dtype
#     ndim = vizs[0].ndim
#
#     if ndim == 2:
#         common_box = (pad_value*np.ones((common_shape[0], common_shape[1]))).astype(dt)
#     elif ndim == 3:
#         common_box = (pad_value*np.ones((common_shape[0], common_shape[1], 3))).astype(dt)
#
#     patches_padded = []
#     for p in vizs:
#         patch_padded = common_box.copy()
#
#         if keep_center:
#             top_margin = (common_shape[0] - p.shape[0])/2
#             left_margin = (common_shape[1] - p.shape[1])/2
#             patch_padded[top_margin:top_margin+p.shape[0], left_margin:left_margin+p.shape[1]] = p
#         else:
#             patch_padded[:p.shape[0], :p.shape[1]] = p
#         patches_padded.append(patch_padded)
#
#     return patches_padded

def display_volume_sections_checkerboard(vol_f, vol_mTof, every=5, ncols=5, direction='z', start_level=None,
                                         grid_size = 60, **kwargs):
    """
    Args:
        direction (str): x,y or z
    """

    assert vol_f.shape == vol_mTof.shape

    vol_mTof_colored = np.zeros(vol_mTof.shape + (3,))
    vol_mTof_colored[..., 0] = vol_mTof # use red
    vol_f_colored = np.zeros(vol_f.shape + (3,))
    vol_f_colored[..., 1] = vol_f # use green

    vol_checkerboard_colored = vol_mTof_colored.copy()
    if direction == 'x':
        for zi, z in enumerate(range(0, vol_mTof.shape[2], grid_size)):
            for yi, y in enumerate(range(0, vol_mTof.shape[0], grid_size)):
                if (zi + yi) % 2 == 1:
                    vol_checkerboard_colored[y:y+grid_size, :, z:z+grid_size] = vol_f_colored[y:y+grid_size, :, z:z+grid_size].copy()
    elif direction == 'z':
        for xi, x in enumerate(range(0, vol_mTof.shape[1], grid_size)):
            for yi, y in enumerate(range(0, vol_mTof.shape[0], grid_size)):
                if (xi + yi) % 2 == 1:
                    vol_checkerboard_colored[y:y+grid_size, x:x+grid_size, :] = vol_f_colored[y:y+grid_size, x:x+grid_size, :].copy()

    ############################

    display_volume_sections(vol_checkerboard_colored, every=every,
                            direction=direction, start_level=start_level, ncols=ncols, **kwargs)


def display_volume_sections(vol, every=5, ncols=5, direction='z', start_level=None, **kwargs):
    """
    Show the sections of a volume in a grid display.

    Args:
        direction (str): x,y or z
    """

    if direction == 'z':
        zmin, zmax = bbox_3d(vol)[4:]
        if start_level is None:
            zs = range(zmin, zmax+1, every)
        else:
            zs = range(start_level, zmax+1, every)
        vizs = [vol[:, :, z] for z in zs]
        titles = ['z=%d' % z  for z in zs]
    elif direction == 'x':
        xmin, xmax = bbox_3d(vol)[:2]
        if start_level is None:
            xs = range(xmin, xmax+1, every)
        else:
            xs = range(start_level, xmax+1, every)
        vizs = [vol[:, x, :] for x in xs]
        titles = ['x=%d' % x for x in xs]
    elif direction == 'y':
        ymin, ymax = bbox_3d(vol)[2:4]
        if start_level is None:
            ys = range(ymin, ymax+1, every)
        else:
            ys = range(start_level, ymax+1, every)
        vizs = [vol[y, :, :] for y in ys]
        titles = ['y=%d' % y for y in ys]

    display_images_in_grids(vizs, nc=ncols, titles=titles, **kwargs)


def display_images_in_grids(vizs, nc, titles=None, export_fn=None, maintain_shape=True, pad_color='white',
                            title_fontsize=10, **kwargs):
    """
    Display a list of images in a grid.

    Args:
        vizs (list of images):
        nc (int): number of images in each row
        maintain_shape (bool): pad patches to same size.
        pad_color (str): black or white
    """

    if maintain_shape:
        if pad_color == 'white':
            pad_value = 255
        elif pad_color == 'black':
            pad_value = 0
        vizs = pad_patches_to_same_size(vizs, pad_value=pad_value)

    n = len(vizs)
    nr = int(np.ceil(n/float(nc)))
    aspect_ratio = vizs[0].shape[1]/float(vizs[0].shape[0]) # width / height

    fig, axes = plt.subplots(nr, nc, figsize=(nc*5*aspect_ratio, nr*5))
    axes = axes.flatten()

    for i in range(len(axes)):
        if i >= n:
            axes[i].axis('off');
        else:
            if vizs[i].dtype == np.float16:
                vizs[i] = vizs[i].astype(np.float32)
            axes[i].imshow(vizs[i], **kwargs);
            if titles is not None:
                axes[i].set_title(titles[i], fontsize=title_fontsize);
            axes[i].set_xticks([]);
            axes[i].set_yticks([]);

    fig.tight_layout();

    if export_fn is not None:
        create_if_not_exists(os.path.dirname(export_fn))
        plt.savefig(export_fn);
        plt.close(fig)
    else:
        plt.show();

# <codecell>

# import numpy as np
# from scipy.ndimage.filters import maximum_filter
# from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

# def detect_peaks(image):
#     """
#     Takes an image and detect the peaks usingthe local maximum filter.
#     Returns a boolean mask of the peaks (i.e. 1 when
#     the pixel's value is the neighborhood maximum, 0 otherwise)
#     """

#     # define an 8-connected neighborhood
#     neighborhood = generate_binary_structure(2,2)

#     #apply the local maximum filter; all pixel of maximal value
#     #in their neighborhood are set to 1
#     local_max = maximum_filter(image, footprint=neighborhood)==image
#     #local_max is a mask that contains the peaks we are
#     #looking for, but also the background.
#     #In order to isolate the peaks we must remove the background from the mask.

#     #we create the mask of the background
#     background = (image==0)

#     #a little technicality: we must erode the background in order to
#     #successfully subtract it form local_max, otherwise a line will
#     #appear along the background border (artifact of the local maximum filter)
#     eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

#     #we obtain the final mask, containing only peaks,
#     #by removing the background from the local_max mask
#     detected_peaks = local_max - eroded_background

#     return detected_peaks

# <codecell>

# def visualize_cluster(scores, cluster='all', title='', filename=None):
#     '''
#     Generate black and white image with the cluster of superpixels highlighted
#     '''

#     vis = scores[segmentation]
#     if cluster != 'all':
#         cluster_selection = np.equal.outer(segmentation, cluster).any(axis=2)
#         vis[~cluster_selection] = 0

#     plt.matshow(vis, cmap=plt.cm.Greys_r);
#     plt.axis('off');
#     plt.title(title)
#     if filename is not None:
#         plt.savefig(os.path.join(result_dir, 'stages', filename + '.png'), bbox_inches='tight')
# #     plt.show()
#     plt.close();


def paint_superpixels_on_image(superpixels, segmentation, img):
    '''
    Highlight a cluster of superpixels on the real image
    '''

    cluster_map = -1*np.ones_like(segmentation)
    for s in superpixels:
        cluster_map[segmentation==s] = 1
    vis = label2rgb(cluster_map, image=img)
    return vis

def paint_superpixel_groups_on_image(sp_groups, segmentation, img, colors):
    '''
    Highlight multiple superpixel groups with different colors on the real image
    '''

    cluster_map = -1*np.ones_like(segmentation)
    for i, sp_group in enumerate(sp_groups):
        for j in sp_group:
            cluster_map[segmentation==j] = i
    vis = label2rgb(cluster_map, image=img, colors=colors)
    return vis

# <codecell>

def kl(a,b):
    m = (a!=0) & (b!=0)
    return np.sum(a[m]*np.log(a[m]/b[m]))

def js(u,v):
    m = .5 * (u + v)
    r = .5 * (kl(u,m) + kl(v,m))
    return r

# <codecell>

def chi2(u,v):
    """
    Compute Chi^2 distance between two distributions.

    Empty bins are ignored.

    """

    u[u==0] = 1e-6
    v[v==0] = 1e-6
    r = np.sum(((u-v)**2).astype(np.float)/(u+v))

    # m = (u != 0) & (v != 0)
    # r = np.sum(((u[m]-v[m])**2).astype(np.float)/(u[m]+v[m]))

    # r = np.nansum(((u-v)**2).astype(np.float)/(u+v))
    return r


def chi2s(h1s, h2s):
    """
    h1s is n x n_texton
    MUST be float type
    """
    return np.sum((h1s-h2s)**2/(h1s+h2s+1e-10), axis=1)

# def chi2s(h1s, h2s):
#     '''
#     h1s is n x n_texton
#     '''
#     s = (h1s+h2s).astype(np.float)
#     with np.errstate(divide='ignore', invalid='ignore'):
#         ss = (h1s-h2s)**2/s
#     ss[s==0] = 0
#     return np.sum(ss, axis=1)


def alpha_blending(src_rgb, dst_rgb, src_alpha, dst_alpha):


    if src_rgb.dtype == np.uint8:
        src_rgb = img_as_float(src_rgb)

    if dst_rgb.dtype == np.uint8:
        dst_rgb = img_as_float(dst_rgb)

    if src_rgb.ndim == 2:
        src_rgb = gray2rgb(src_rgb)

    if dst_rgb.ndim == 2:
        dst_rgb = gray2rgb(dst_rgb)

    if isinstance(src_alpha, float) or  isinstance(src_alpha, int):
        src_alpha = src_alpha * np.ones((src_rgb.shape[0], src_rgb.shape[1]))

    if isinstance(dst_alpha, float) or  isinstance(dst_alpha, int):
        dst_alpha = dst_alpha * np.ones((dst_rgb.shape[0], dst_rgb.shape[1]))

    out_alpha = src_alpha + dst_alpha * (1. - src_alpha)
    out_rgb = (src_rgb * src_alpha[..., None] +
               dst_rgb * dst_alpha[..., None] * (1. - src_alpha[..., None])) / out_alpha[..., None]

    out = np.zeros((src_rgb.shape[0], src_rgb.shape[1], 4))

    out[..., :3] = out_rgb
    out[..., 3] = out_alpha

    return out


def bbox_2d(img):
    """
    Returns:
        (xmin, xmax, ymin, ymax)
    """

    if np.count_nonzero(img) == 0:
        raise Exception('bbox2d: Image is empty.')
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return cmin, cmax, rmin, rmax

def bbox_3d(img):

    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    try:
        rmin, rmax = np.where(r)[0][[0, -1]]
        cmin, cmax = np.where(c)[0][[0, -1]]
        zmin, zmax = np.where(z)[0][[0, -1]]
    except:
        raise Exception('Input is empty.\n')

    return cmin, cmax, rmin, rmax, zmin, zmax

def compute_centroid_3d(vol):
    """
    Args:
        vol (3-d array): if float, return weighted centroid
    """
    # from scipy.ndimage.measurements import center_of_mass
    # return center_of_mass(vol)
    return np.mean(np.where(vol), axis=1)[[1,0,2]]

# def sample(points, num_samples):
#     n = len(points)
#     sampled_indices = np.random.choice(range(n), min(num_samples, n), replace=False)
#     return points[sampled_indices]

def apply_function_to_nested_list(func, l):
    """
    Func applies to the list consisting of all elements of l, and return a list.
    l: a list of list
    """
    from itertools import chain
    result = func(list(chain(*l)))
    csum = np.cumsum(map(len, l))
    new_l = [result[(0 if i == 0 else csum[i-1]):csum[i]] for i in range(len(l))]
    return new_l


def apply_function_to_dict(func, d):
    """
    Args:
        func:
            a function that takes as input the list consisting of a flatten list of values of `d`, and return a list.
        d (dict {key: list}):
    """
    from itertools import chain
    result = func(list(chain(*d.values())))
    csum = np.cumsum(map(len, d.values()))
    new_d = {k: result[(0 if i == 0 else csum[i-1]):csum[i]] for i, k in enumerate(d.keys())}
    return new_d

def smart_map(data, keyfunc, func):
    """
    Args:
        data (list): data
        keyfunc (func): a lambda function for data key.
        func (func): a function that takes f(key, group). `group` is a sublist of `data`.
                    func does a global operation using key.
                    then does a same operation on each entry of group.
                    return a list.
    """

    from itertools import groupby
    from multiprocess import Pool

    keyfunc_with_enum = lambda (i, x): keyfunc(x)

    grouping = groupby(sorted(enumerate(data), key=keyfunc_with_enum), keyfunc_with_enum)
    grouping_noidx = {}
    grouping_idx = {}
    for k, group in grouping:
        grouping_idx[k], grouping_noidx[k] = zip(*group)

    pool = Pool(15)
    results_by_key = pool.map(lambda (k, group): func(k, group), grouping_noidx.iteritems())
    pool.close()
    pool.join()

    # results_by_key = [func(k, group) for k, group in grouping_noidx.iteritems()]

    keys = grouping_noidx.keys()
    results_inOrigOrder = {i: res for k, results in zip(keys, results_by_key)
                           for i, res in zip(grouping_idx[k], results)}

    return results_inOrigOrder.values()

boynton_colors = dict(blue=(0,0,255),
    red=(255,0,0),
    green=(0,255,0),
    yellow=(255,255,0),
    magenta=(255,0,255),
    pink=(255,128,128),
    gray=(128,128,128),
    brown=(128,0,0),
    orange=(255,128,0))

kelly_colors = dict(vivid_yellow=(255, 179, 0),
                    strong_purple=(128, 62, 117),
                    vivid_orange=(255, 104, 0),
                    very_light_blue=(166, 189, 215),
                    vivid_red=(193, 0, 32),
                    grayish_yellow=(206, 162, 98),
                    medium_gray=(129, 112, 102),

                    # these aren't good for people with defective color vision:
                    vivid_green=(0, 125, 52),
                    strong_purplish_pink=(246, 118, 142),
                    strong_blue=(0, 83, 138),
                    strong_yellowish_pink=(255, 122, 92),
                    strong_violet=(83, 55, 122),
                    vivid_orange_yellow=(255, 142, 0),
                    strong_purplish_red=(179, 40, 81),
                    vivid_greenish_yellow=(244, 200, 0),
                    strong_reddish_brown=(127, 24, 13),
                    vivid_yellowish_green=(147, 170, 0),
                    deep_yellowish_brown=(89, 51, 21),
                    vivid_reddish_orange=(241, 58, 19),
                    dark_olive_green=(35, 44, 22))

high_contrast_colors = boynton_colors.values() + kelly_colors.values()

import randomcolor

def random_colors(count):
    rand_color = randomcolor.RandomColor()
    random_colors = [map(int, rgb_str[4:-1].replace(',', ' ').split())
                     for rgb_str in rand_color.generate(luminosity="bright", count=count, format_='rgb')]
    return random_colors

def read_dict_from_txt(fn, converter=np.float, key_converter=np.int):
    d = {}
    with open(fn, 'r') as f:
        for line in f.readlines():
            items = line.split()
            if len(items) == 2:
                d[key_converter(items[0])] = converter(items[1])
            else:
                d[key_converter(items[0])] = np.array(map(converter, items[1:]))
    return d

def write_dict_to_txt(d, fn, fmt='%f'):
    with open(fn, 'w') as f:
        for k, vals in d.iteritems():
            f.write(k + ' ' +  (' '.join([fmt]*len(vals))) % tuple(vals) + '\n')
