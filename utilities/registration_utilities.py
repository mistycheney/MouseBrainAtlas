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
    assert np.count_nonzero([theta_xy, theta_yz, theta_xz]) <= 1, \
    "Current implementation is sound only if only one rotation is given."

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


def N(i,t):
    """
    Cubic B-spline base functions.
    
    Args:
        i (int): control point index. Can be negative (?)
        t (float): position.
    """
    
    if i <= t and t < i+1:
        return (t-i)**3/6.
    elif i+1 <= t and t < i+2:
        return (t-i)**2*(i+2-t)/6. + (t-i)*(i+3-t)*(t-i-1)/6. + (i+4-t)*(t-i-1)**2/6.
    elif i+2 <= t and t < i+3:
        return (t-i)*(i+3-t)**2/6. + (i+4-t)*(i+3-t)*(t-i-1)/6. + (i+4-t)**2*(t-i-2)/6.
    elif i+3 <= t and t < i+4:
        return (i+4-t)**3/6.
    else:
        return 0
    
volume_f = None
volume_m = None
nzvoxels_m = None
nzvoxels_centered_m = None
grad_f = None
grad_m = None

#########################################################################

from skimage.filters import gaussian

class Aligner4(object):
    def __init__(self, volume_f_, volume_m_=None, nzvoxels_m_=None, centroid_f=None, centroid_m=None, \
                labelIndexMap_m2f=None, label_weights=None, reg_weights=None, zrange=None, nz_thresh=0):
        """
        Variant that takes in two probabilistic volumes.

        Args:
            volume_f_ (dict): the fixed probabilistic volume. dict of {numeric label: 3d array}
            volume_m_ (dict): the moving probabilistic volume. dict of {numeric label: 3d array}
            labelIndexMap_m2f (dict): mapping between moving volume labels and fixed volume labels. dict of {moving label: fixed label}
            label_weights (dict): {numeric label: weight}
            reg_weights (3-array): regularization weights for (tx,ty,tz) respectively.
            zrange (2-tuple): If given, only use the portion of both volumes that is between zmin and zmax (inclusive).
            nz_thresh (float): a number above which the score value is used for registration.
        """

        self.labelIndexMap_m2f = labelIndexMap_m2f

        if isinstance(volume_m_, dict): # probabilistic volume
            labels_in_volume_m = set(np.unique(volume_m_.keys()))
        else: # annotation volume
            labels_in_volume_m = set(np.unique(volume_m_))

        if isinstance(volume_f_, dict): # probabilistic volume
            labels_in_volume_f = set(np.unique(volume_f_.keys()))
        else: # annotation volume
            labels_in_volume_f = set(np.unique(volume_f_))

        self.all_indices_f = set([])
        self.all_indices_m = set([])
        for idx_m in set(self.labelIndexMap_m2f.keys()) & labels_in_volume_m:
            idx_f = self.labelIndexMap_m2f[idx_m]
            if idx_f in labels_in_volume_f:
                self.all_indices_f.add(idx_f)
                self.all_indices_m.add(idx_m)
        # self.all_indices_m = list(self.all_indices_m)

        # self.all_indices_m = list(set(self.labelIndexMap_m2f.keys()) & labels_in_volume_m)
        # self.all_indices_f = set([self.labelIndexMap_m2f[ind_m] for ind_m in self.all_indices_m])

        global volume_f

        if isinstance(volume_f_, dict): # probabilistic volume
            # volume_f = volume_f_
            volume_f = {i: volume_f_[i] for i in self.all_indices_f}

        else: # annotation volume
            volume_f = {i: np.zeros_like(volume_f_, dtype=np.float16) for i in self.all_indices_f}
            for i in self.all_indices_f:
                mask = volume_f_ == i
                volume_f[i][mask] = 1.
                del mask

        # for i, v in volume_f.iteritems():
        #     volume_f[i] = gaussian(v, 3)

        global volume_m

        if isinstance(volume_m_, dict): # probabilistic volume
            # volume_m = volume_m_
            volume_m = {i: volume_m_[i] for i in self.all_indices_m}
        else: # annotation volume; also convert to dict, treated the same as prob. volume
            volume_m = {i: np.zeros_like(volume_m_, dtype=np.float16) for i in self.all_indices_m}
            for i in self.all_indices_m:
                mask = volume_m_ == i
                volume_m[i][mask] = 1.
                del mask

        # Slice volumes if arange is specified
        if zrange is not None:
            self.zl, self.zh = zrange
            volume_f = {l: v[..., self.zl:self.zh+1] for l, v in volume_f.iteritems()}
            volume_m = {l: v[..., self.zl:self.zh+1] for l, v in volume_m.iteritems()}

            volume_f_sliced = {}
            for l, v in volume_f.iteritems():
                if np.count_nonzero(v) == 0:
                    sys.stderr.write('Fixed volume %(label)s at z=(%(zl)d,%(zh)d) is empty.\n' % dict(label=l, zl=self.zl, zh=self.zh))
                else:
                    volume_f_sliced[l] = v

            volume_m_sliced = {}
            for l, v in volume_m.iteritems():
                if np.count_nonzero(v) == 0:
                    sys.stderr.write('Moving volume %(label)s at z=(%(zl)d,%(zh)d) is empty.\n' % dict(label=l, zl=self.zl, zh=self.zh))
                else:
                    volume_m_sliced[l] = v

            if len(volume_f_sliced) == 0:
                raise Exception('All fixed volumes at z=(%(zl)d,%(zh)d) is empty.\n' % dict(zl=self.zl, zh=self.zh))

            if len(volume_m_sliced) == 0:
                raise Exception('All moving volumes at z=(%(zl)d,%(zh)d) is empty.\n' % dict(zl=self.zl, zh=self.zh))

            volume_f = volume_f_sliced
            volume_m = volume_m_sliced

        # for i, v in volume_m.iteritems():
        #     volume_m[i] = gaussian(v, 3)

        assert volume_f is not None, 'Template volume is not specified.'
        assert volume_m is not None, 'Moving volume is not specified.'

        self.ydim_m, self.xdim_m, self.zdim_m = volume_m.values()[0].shape
        self.ydim_f, self.xdim_f, self.zdim_f = volume_f.values()[0].shape

        global nzvoxels_m
        if nzvoxels_m_ is None:

            # pool = Pool(16)
            # # nzvoxels_m_ = pool.map(lambda i: parallel_where_binary(volume_m[i] > 0, num_samples=int(1e6)),
            # #                         self.all_indices_m)
            # nzvoxels_m_ = pool.map(lambda i: parallel_where_binary(volume_m[i] > 0),
            #                         self.all_indices_m)
            # pool.terminate()
            # pool.join()

            nzvoxels_m_ = [parallel_where_binary(volume_m[i] > nz_thresh) for i in list(self.all_indices_m)]
            nzvoxels_m = dict(zip(list(self.all_indices_m), nzvoxels_m_))
        else:
            nzvoxels_m = nzvoxels_m_

        if label_weights is None:
            if not hasattr(self, 'label_weights'):
                sys.stderr.write('Label weights not set, default to 1 for all structures.\n')
                self.label_weights = {ind_m: 1 for ind_m in self.all_indices_m}
        else:
            self.label_weights = label_weights

        if reg_weights is None:
            if not hasattr(self, 'reg_weights'):
                sys.stderr.write('Regularization weights not set, default to 0.\n')
                self.reg_weights = np.array([0,0,0])
        else:
            self.reg_weights = reg_weights

    def set_label_weights(self, label_weights):
        self.label_weights = label_weights

    def set_regularization_weights(self, reg_weights):
        self.reg_weights = reg_weights
    
    def set_bspline_grid_size(self, interval):
        """
        Args:
            interval (float): x,y,z interval in voxels.
        """
                
        ctrl_x_intervals = np.arange(0, self.xdim_m, interval)
        ctrl_y_intervals = np.arange(0, self.ydim_m, interval)
        ctrl_z_intervals = np.arange(0, self.zdim_m, interval)
        
        ctrl_x_intervals_centered = ctrl_x_intervals - self.centroid_m[0]
        ctrl_y_intervals_centered = ctrl_y_intervals - self.centroid_m[1]
        ctrl_z_intervals_centered = ctrl_z_intervals - self.centroid_m[2]

        self.n_ctrl = len(ctrl_x_intervals) * len(ctrl_y_intervals) * len(ctrl_z_intervals)

        self.NuNvNw_allTestPts = {}
        
        for ind_m, test_pts in nzvoxels_centered_m.iteritems():
            
            t = time.time()
        
            NuPx_allTestPts = np.array([[N(ctrl_x/float(interval), x/float(interval)) 
                                         for testPt_i, (x, y, z) in enumerate(test_pts)]
                                        for ctrlXInterval_i, ctrl_x in enumerate(ctrl_x_intervals_centered)])
            # (n_ctrlx, n_all_nz_m)
            NvPy_allTestPts = np.array([[N(ctrl_y/float(interval), y/float(interval)) 
                                         for testPt_i, (x, y, z) in enumerate(test_pts)]
                                        for ctrlYInterval_i, ctrl_y in enumerate(ctrl_y_intervals_centered)])
            # (n_ctrly, n_all_nz_m)
            NwPz_allTestPts = np.array([[N(ctrl_z/float(interval), z/float(interval)) 
                                         for testPt_i, (x, y, z) in enumerate(test_pts)]
                                        for ctrlZInterval_i, ctrl_z in enumerate(ctrl_z_intervals_centered)])
            # (n_ctrlz, n_all_nz_m)
            # print 'NwPz_allTestPts', NwPz_allTestPts.shape
            
            sys.stderr.write("Compute NuPx/NvPy/NwPz: %.2f seconds.\n" % (time.time()-t) )

            t = time.time()

            self.NuNvNw_allTestPts[ind_m] = np.array([np.ravel(np.tensordot(np.tensordot(NuPx_allTestPts[:,testPt_i], 
                                                                                  NvPy_allTestPts[:,testPt_i], 0), 
                                                                     NwPz_allTestPts[:,testPt_i], 0))
                                  for testPt_i in range(len(test_pts))])
            # print 'self.NuNvNw_allTestPts', self.NuNvNw_allTestPts[ind_m].shape
            # (n_all_nz_m, n_ctrl)

            sys.stderr.write("Compute every control point's contribution to every nonzero test point: %.2f seconds.\n" % (time.time()-t) )
            
    def set_centroid(self, centroid_m=None, centroid_f=None, indices_m=None):
        """
        Args:
            centroid_m (str or (3,)-ndarray): Coordinates or one of structure_centroid, volume_centroid, origin
            centroid_f (str or (3,)-ndarray): Coordinates or one of centroid_m, structure_centroid, volume_centroid, origin
        """

        if indices_m is None:
            indices_m = self.all_indices_m

        if isinstance(centroid_m, basestring):
            if centroid_m == 'structure_centroid':
                self.centroid_m = np.concatenate([nzvoxels_m[i] for i in indices_m]).mean(axis=0)
            elif centroid_m == 'volume_centroid':
                self.centroid_m = np.r_[.5*self.xdim_m, .5*self.ydim_m, .5*self.zdim_m]
            elif centroid_m == 'origin':
                self.centroid_m = np.zeros((3,))
            else:
                raise Exception('centroid_m not recognized.')

        if isinstance(centroid_f, basestring):
            if centroid_f == 'centroid_m':
                self.centroid_f = self.centroid_m
            elif centroid_m == 'structure_centroid':
                self.centroid_f = np.hstack([np.nonzero(volume_f[self.labelIndexMap_m2f[i]]) for i in indices_m]).mean(axis=1)[[1,0,2]]
            elif centroid_f == 'volume_centroid':
                self.centroid_f = np.r_[.5*self.xdim_f, .5*self.ydim_f, .5*self.zdim_f]
            elif centroid_f == 'origin':
                self.centroid_f = np.zeros((3,))
            else:
                raise Exception('centroid_f not recognized.')

        sys.stderr.write("centroid_m: %s, centroid_f: %s\n" % (self.centroid_m, self.centroid_f))

        global nzvoxels_centered_m
        nzvoxels_centered_m = {ind_m: nzvs - self.centroid_m for ind_m, nzvs in nzvoxels_m.iteritems()}

    # def compute_gradient(self, indices_f=None):
    # # For some unknown reasons, directly computing gradients in Aligner class is quick (~2s) for only the first 4-5 volumes;
    # # extra volumes take exponentially long time. However, loading from disk is quick for all 28 volumes.

    #     if indices_f is None:
    #         indices_f = set([self.labelIndexMap_m2f[ind_m] for ind_m in self.all_indices_m])
    #         print indices_f
    #
    #     global grad_f
    #     grad_f = {ind_f: np.empty((3, self.ydim_f, self.xdim_f, self.zdim_f), dtype=np.float16) for ind_f in indices_f}
    #     # conversion between float16 and float32 takes time.
    #
    #     t1 = time.time()
    #     for ind_f in indices_f:
    #         t = time.time()
    #         # np.gradient needs float32 - will convert if input is not.
    #         gy_gx_gz = np.gradient(volume_f[ind_f].astype(np.float32), 3, 3, 3)
    #         if hasattr(self, 'zl'):
    #             grad_f[ind_f][0] = gy_gx_gz[1][..., self.zl:self.zh+1]
    #             grad_f[ind_f][1] = gy_gx_gz[0][..., self.zl:self.zh+1]
    #             grad_f[ind_f][2] = gy_gx_gz[2][..., self.zl:self.zh+1]
    #         else:
    #             grad_f[ind_f][0] = gy_gx_gz[1]
    #             grad_f[ind_f][1] = gy_gx_gz[0]
    #             grad_f[ind_f][2] = gy_gx_gz[2]
    #         del gy_gx_gz
    #         sys.stderr.write('load gradient %s: %f seconds\n' % (ind_f, time.time() - t)) #4s
    #
    #     sys.stderr.write('overall: %f seconds\n' % (time.time() - t1)) # ~100s

    def load_gradient(self, gradient_filepath_map_f=None, indices_f=None, gradients=None):
        """Load gradients.

        Need to pass gradient_filepath_map_f in from outside because Aligner class should be agnostic about structure names.

        Args:
            gradient_filepath_map_f (dict of str): path string that contains formatting parts and (suffix).
            graidents (dict of (3,dimx,dimy,dimz) arrays): 

        """

        if indices_f is None:
            indices_f = set([self.labelIndexMap_m2f[ind_m] for ind_m in self.all_indices_m])
            sys.stderr.write('indices_f: %s\n' % indices_f) 

        global grad_f
        
        if gradients is not None:
            grad_f = gradients
        else:
            assert gradient_filepath_map_f is not None, 'gradient_filepath_map_f not specified.'
            grad_f = {ind_f: np.zeros((3, self.ydim_f, self.xdim_f, self.zdim_f), dtype=np.float16) for ind_f in indices_f}

            t1 = time.time()

            for ind_f in indices_f:

                t = time.time()

                download_from_s3(gradient_filepath_map_f[ind_f] % {'suffix': 'gx'}, is_dir=False)
                download_from_s3(gradient_filepath_map_f[ind_f] % {'suffix': 'gy'}, is_dir=False)
                download_from_s3(gradient_filepath_map_f[ind_f] % {'suffix': 'gz'}, is_dir=False)

                if hasattr(self, 'zl'):
                    grad_f[ind_f][0] = bp.unpack_ndarray_file(gradient_filepath_map_f[ind_f] % {'suffix': 'gx'})[..., self.zl:self.zh+1]
                    grad_f[ind_f][1] = bp.unpack_ndarray_file(gradient_filepath_map_f[ind_f] % {'suffix': 'gy'})[..., self.zl:self.zh+1]
                    grad_f[ind_f][2] = bp.unpack_ndarray_file(gradient_filepath_map_f[ind_f] % {'suffix': 'gz'})[..., self.zl:self.zh+1]
                else:
                    grad_f[ind_f][0] = bp.unpack_ndarray_file(gradient_filepath_map_f[ind_f] % {'suffix': 'gx'})
                    grad_f[ind_f][1] = bp.unpack_ndarray_file(gradient_filepath_map_f[ind_f] % {'suffix': 'gy'})
                    grad_f[ind_f][2] = bp.unpack_ndarray_file(gradient_filepath_map_f[ind_f] % {'suffix': 'gz'})

                sys.stderr.write('load gradient %s: %f seconds\n' % (ind_f, time.time() - t)) # ~6s

            sys.stderr.write('overall: %f seconds\n' % (time.time() - t1)) # ~100s

            
    def get_valid_voxels_after_transform(self, T, tf_type, ind_m, return_valid):
        """
        Args:
            T (ndarray): transform parameter vector
            tf_type (str): rigid, affine or bspline.
            return_valid (bool): whether to return a boolean list indicating which nonzero moving voxels are valid.
        """
        
        if tf_type == 'affine' or tf_type == 'rigid':
            # t = time.time()
            pts_prime = transform_points_affine(np.array(T), 
                                                pts_centered=nzvoxels_centered_m[ind_m],
                                                c_prime=self.centroid_f).astype(np.int16)
            # sys.stderr.write("transform all points: %.2f s\n" % (time.time() - t))
                        
        elif tf_type == 'bspline':
            n_params = len(T)
            buvwx = T[:n_params/3]
            buvwy = T[n_params/3:n_params/3*2]
            buvwz = T[n_params/3*2:]
            pts_prime = transform_points_bspline(buvwx, buvwy, buvwz, 
                                                 pts_centered=nzvoxels_centered_m[ind_m], c_prime=self.centroid_f,
                                                NuNvNw_allTestPts=self.NuNvNw_allTestPts[ind_m]).astype(np.int16)

        # print 'before'
        # print nzvoxels_centered_m[ind_m]
        # print 'after'
        # print pts_prime

        # t = time.time()
        xs_prime, ys_prime, zs_prime = pts_prime.T
        valid_moving_voxel_indicator = (xs_prime >= 0) & (ys_prime >= 0) & (zs_prime >= 0) & \
                (xs_prime < self.xdim_f) & (ys_prime < self.ydim_f) & (zs_prime < self.zdim_f)
        # sys.stderr.write("find valid: %.2f s\n" % (time.time() - t))
        
        # print pts_prime.max(axis=0), pts_prime.min(axis=0)
        # print self.xdim_f, self.ydim_f, self.zdim_f

        # sys.stderr.write("%d total moving, %d valid\n" % (len(nzvoxels_centered_m[ind_m]), np.count_nonzero(valid_moving_voxel_indicator)))
        # print len(pts_prime), np.count_nonzero(valid)

        xs_prime_valid = xs_prime[valid_moving_voxel_indicator]
        ys_prime_valid = ys_prime[valid_moving_voxel_indicator]
        zs_prime_valid = zs_prime[valid_moving_voxel_indicator]

        if return_valid:
            return xs_prime_valid, ys_prime_valid, zs_prime_valid, valid_moving_voxel_indicator
        else:
            return xs_prime_valid, ys_prime_valid, zs_prime_valid


    def compute_score_and_gradient_one(self, T, tf_type, num_samples=None, ind_m=None):
        """
        Compute score and gradient of one structure.
         
        Args:
            T ((nparam,)-array): flattened array of transform parameters.
            tf_type (str): rigid, affine or bspline.
            ind_m (int): index of a structure.
        """
        
        # t = time.time()            

        score, xs_prime_valid, ys_prime_valid, zs_prime_valid, valid_moving_voxel_indicators = self.compute_score_one(T, tf_type=tf_type, ind_m=ind_m, return_valid=True)
        
        # sys.stderr.write("compute_score_one: %.2f s\n" % (time.time() - t)) 
        
        # Moving volume's valid voxel coordinates (not centralized).
        # t = time.time()
        xyzs_valid = nzvoxels_m[ind_m][valid_moving_voxel_indicators]
        # sys.stderr.write("fancy indexing into moving volume nz voxels: %.2f s\n" % (time.time() - t)) 
        # Moving volume's value at valid voxels. (n_valid_nz_m, )
        # t = time.time()
        S_m_valid_scores = volume_m[ind_m][xyzs_valid[:,1], xyzs_valid[:,0], xyzs_valid[:,2]]
        # sys.stderr.write("fancy indexing into moving volume: %.2f s\n" % (time.time() - t)) 
        
        # Moving volume's valid voxel coordinates (centralized).
        # t = time.time()
        dxs, dys, dzs = nzvoxels_centered_m[ind_m][valid_moving_voxel_indicators].T
        # sys.stderr.write("fancy indexing into centralized moving volume nzvoxels: %.2f s\n" % (time.time() - t)) 
                
        if tf_type == 'bspline':
            NuNvNw_allTestPts = self.NuNvNw_allTestPts[ind_m][valid_moving_voxel_indicators].copy()
        
        ind_f = self.labelIndexMap_m2f[ind_m]
        
        # Fixed volume's gradients at valid voxels.

        # t = time.time()
        Sx = grad_f[ind_f][0, ys_prime_valid, xs_prime_valid, zs_prime_valid]
        Sy = grad_f[ind_f][1, ys_prime_valid, xs_prime_valid, zs_prime_valid]
        Sz = grad_f[ind_f][2, ys_prime_valid, xs_prime_valid, zs_prime_valid]
        if np.all(Sx == 0) and np.all(Sy == 0) and np.all(Sz == 0):
            raise Exception("Image gradient at all valid voxel is zero.")
        # sys.stderr.write("fancy indexing into fixed volume gradient: %.2f s\n" % (time.time() - t)) 
        
        xs_prime_valid = xs_prime_valid.astype(np.float)
        ys_prime_valid = ys_prime_valid.astype(np.float)
        zs_prime_valid = zs_prime_valid.astype(np.float)
        
        t = time.time()                    
        
        # Sample within valid voxels.
        # Note that sampling takes time. Maybe it is better not sampling.
        if num_samples is not None:
            
            # t = time.time()
            n_valid = np.count_nonzero(valid_moving_voxel_indicators)
            # Typical n ranges from 63984 to 451341
            # sys.stderr.write("count_nonzero: %.2f s\n" % (time.time() - t))
            n_sample = min(num_samples, n_valid)
            sys.stderr.write('%d: use %d samples out of %d valid\n' % (ind_m, n_sample, n_valid))
            import random
            ii = sorted(random.sample(range(n_valid), n_sample))

            S_m_valid_scores = S_m_valid_scores[ii]
            dxs = dxs[ii]
            dys = dys[ii]
            dzs = dzs[ii]
            Sx = Sx[ii]
            Sy = Sy[ii]
            Sz = Sz[ii]
            xs_prime_valid = xs_prime_valid[ii]
            ys_prime_valid = ys_prime_valid[ii]
            zs_prime_valid = zs_prime_valid[ii]
            
            if tf_type == 'bspline':
                NuNvNw_allTestPts = NuNvNw_allTestPts[ii]
        # else:
        #     n_sample = n_valid
        
        sys.stderr.write("sample: %.2f s\n" % (time.time() - t)) 
        
        if tf_type == 'rigid' or tf_type == 'affine':
            
            # t = time.time()
            # q is dF/dp for a single voxel, where p is a transform parameter.
            if tf_type == 'rigid':
                q = np.c_[Sx, Sy, Sz,
                        -Sy*zs_prime_valid + Sz*ys_prime_valid,
                        Sx*zs_prime_valid - Sz*xs_prime_valid,
                        -Sx*ys_prime_valid + Sy*xs_prime_valid]
            elif tf_type == 'affine':
                q = np.c_[Sx*dxs, Sx*dys, Sx*dzs, Sx, Sy*dxs, Sy*dys, Sy*dzs, Sy, Sz*dxs, Sz*dys, Sz*dzs, Sz]
            # sys.stderr.write("compute gradient, all voxels: %.2f s\n" % (time.time() - t)) 
            
            # Whether to scale gradient to match the scores' scale depends on whether AdaGrad is used;
            # if used, then the scale will be automatically adapted so the scaling does not matter
            # t = time.time()
            # grad = q.sum(axis=0) / 1e6
            grad = (S_m_valid_scores[:,None] * q).sum(axis=0)
            if np.all(grad == 0):
                raise Exception("Gradient is zero.")
            # sys.stderr.write("compute gradient, sum: %.2f s\n" % (time.time() - t)) 

        elif tf_type == 'bspline':
            
            dqxdbuvwx_allTestPts = NuNvNw_allTestPts
            # (n_valid_nz_m, n_ctrl)
            dqydbuvwy_allTestPts = NuNvNw_allTestPts
            dqzdbuvwz_allTestPts = NuNvNw_allTestPts

            dSdbuvwx_allTestPts = Sx[:,None] * dqxdbuvwx_allTestPts
            dSdbuvwy_allTestPts = Sy[:,None] * dqydbuvwy_allTestPts
            dSdbuvwz_allTestPts = Sz[:,None] * dqzdbuvwz_allTestPts
            # print 'dSdbuvwz_allTestPts', dSdbuvwz_allTestPts.shape
            # (n_valid_nz_m, n_ctrl)

            dFdbuvwx = np.dot(S_m_valid_scores, dSdbuvwx_allTestPts) # (n_ctrl, )
            dFdbuvwy = np.dot(S_m_valid_scores, dSdbuvwy_allTestPts) # (n_ctrl, )
            dFdbuvwz = np.dot(S_m_valid_scores, dSdbuvwz_allTestPts) # (n_ctrl, )
            # print 'dFdbuvwz', dFdbuvwz.shape
            
            grad = np.concatenate([dFdbuvwx, dFdbuvwy, dFdbuvwz]) # (n_ctrl*3, )
            # print 'grad', grad.shape
            sys.stderr.write('grad_min: %.2f, grad_max: %.2f\n' % (grad.min(), grad.max())) 
            
            q = None

        
        # t = time.time()
        
        # regularized version
        if tf_type == 'rigid' or tf_type == 'affine':
            tx = T[3]
            ty = T[7]
            tz = T[11]

            if tf_type == 'rigid':
                grad[0] = grad[0] - 2*self.reg_weights[0] * tx
                # print grad[0], 2*self.reg_weights[0] * tx
                grad[1] = grad[1] - 2*self.reg_weights[1] * ty
                grad[2] = grad[2] - 2*self.reg_weights[2] * tz
            elif tf_type == 'affine':
                grad[3] = grad[3] - 2*self.reg_weights[0] * tx
                # print grad[3], 2*self.reg_weights[0] * tx
                grad[7] = grad[7] - 2*self.reg_weights[1] * ty
                grad[11] = grad[11] - 2*self.reg_weights[2] * tz
        elif tf_type == 'bspline':
            pass
                
        # sys.stderr.write("3: %.2f s\n" % (time.time() - t)) 

        # del q, Sx, Sy, Sz, dxs, dys, dzs, xs_prime_valid, ys_prime_valid, zs_prime_valid
        del q, Sx, Sy, Sz, dxs, dys, dzs, xs_prime_valid, ys_prime_valid, zs_prime_valid, S_m_valid_scores
        # del xs_valid, ys_valid, zs_valid
            
        return score, grad

    def compute_score_and_gradient(self, T, tf_type, num_samples=None, indices_m=None):
        """
        Compute score and gradient.
        v is update on the Lie space.

        Args:
            T ((nparam,)-ndarray): flattened array of transform parameters
            num_samples (int): Number of sample points to compute gradient.
            tf_type (str): if 'rigid', compute gradient with respect to (tx,ty,tz,w1,w2,w3);
                            if 'affine', compute gradient with respect to 12 parameters;
                            if 'bspline', compute gradient wrt given number of parameters.
            indices_m (integer list):

        Returns:
            (tuple): tuple containing:
            - score (int): score
            - grad (float): gradient
        """

        score = 0

        if tf_type == 'rigid':
            grad = np.zeros((6,))
        elif tf_type == 'affine':
            grad = np.zeros((12,))
        elif tf_type == 'bspline':
            grad = np.zeros((self.n_ctrl*3,))
            
        if indices_m is None:
            indices_m = self.all_indices_m

        # serial
        for ind_m in indices_m:
            # t = time.time()
            try:
                score_one, grad_one = self.compute_score_and_gradient_one(T, tf_type=tf_type, num_samples=num_samples, ind_m=ind_m)
                # sys.stderr.write("compute_score_and_gradient_one: %.2f s\n" % (time.time()-t))
                # sys.stderr.write("%d, %f\n" % (ind_m, score_one))
                # print "grad_one", grad_one.shape
                # grad += grad_one
                # score += score_one
                grad += self.label_weights[ind_m] * grad_one
                score += self.label_weights[ind_m] * score_one

            except Exception as e:
                raise e
                sys.stderr.write('Error computing score/gradient for %d: %s\n' % (ind_m, e))

        # # parallel
        ## Parallel does not save time, maybe because the computation for each subprocess is too short.
        # pool = Pool(12)
        # score_grad_tuples = pool.map(lambda ind_m: self.compute_score_and_gradient_one(T, num_samples, wrt_v, ind_m), indices_m)
        # pool.close()
        # pool.join()
        # for s, g in score_grad_tuples:
        #     score += s
        #     grad += g

        return score, grad


    def compute_score_one(self, T, tf_type, ind_m, return_valid=False):
        """
        Compute score for one label.
        Notice that raw overlap score is divided by 1e6 before returned.

        Args:
            T ((nparam,)-ndarray): flattened array of transform parameters
            tf_type (str): rigid, affine or bspline.
            ind_m (int): label on the moving volume
            return_valid (bool): whether to return valid voxels

        Returns:
            (float or tuple): if `return_valid` is true, return a tuple containing:
            - score (int): score
            - xs_prime_valid (array):
            - ys_prime_valid (array):
            - zs_prime_valid (array):
            - valid (boolean array):
        """
        
        # t = time.time()
        xs_prime_valid, ys_prime_valid, zs_prime_valid, valid_moving_voxel_indicator = self.get_valid_voxels_after_transform(T, tf_type=tf_type, ind_m=ind_m, return_valid=True)
        # sys.stderr.write("Timing 2: get_valid_voxels_after_transform: %.2f seconds.\n" % (time.time()-t))
        
        n_total = len(nzvoxels_centered_m[ind_m])
        n_valid = np.count_nonzero(valid_moving_voxel_indicator)
        n_invalid = n_total - n_valid
        # sys.stderr.write('%d: %d valid, %d out-of-bound voxels after transform.\n' % (ind_m, n_valid, n_invalid))
        if n_valid == 0:
            raise Exception('%d: No valid voxels after transform.' % ind_m)

        ind_f = self.labelIndexMap_m2f[ind_m]

        # Reducing the scale of voxel value is important for keeping the sum (i.e. the following line) in the represeantable range of the chosen data type.
        # t = time.time()
        voxel_probs_valid = volume_f[ind_f][ys_prime_valid, xs_prime_valid, zs_prime_valid] / 1e6
        # sys.stderr.write("Timing 2: fancy indexing valid voxels into fixed volume: %.2f seconds.\n" % (time.time()-t))
        
        # Penalize out-of-bound voxels, minus 1 for each such voxel
        s = voxel_probs_valid.sum() - np.sign(self.label_weights[ind_m]) * n_invalid / 1e6

        # Regularize
        if tf_type == 'affine' or tf_type == 'rigid':
            tx = T[3]
            ty = T[7]
            tz = T[11]
            s_reg = self.reg_weights[0]*tx**2 + self.reg_weights[1]*ty**2 + self.reg_weights[2]*tz**2
        else:
            s_reg = 0
        
        s = s - s_reg
        
        if return_valid:
            return s, xs_prime_valid, ys_prime_valid, zs_prime_valid, valid_moving_voxel_indicator
        else:
            return s

    def compute_score(self, T, tf_type='affine', indices_m=None, return_individual_score=False):
        """Compute score.

        Returns:
        """

        if indices_m is None:
            indices_m = self.all_indices_m

        score_all_landmarks = {}
        for ind_m in indices_m:
            try:
                score_all_landmarks[ind_m] = self.compute_score_one(T, tf_type=tf_type, ind_m=ind_m, return_valid=False)
            except Exception as e:
                sys.stderr.write('Error computing score for %d: %s\n' % (ind_m, e))
                score_all_landmarks[ind_m] = 0

        # score = np.sum(score_all_landmarks.values())

        score = 0
        for ind_m, score_one in score_all_landmarks.iteritems():
            score += self.label_weights[ind_m] * score_one

        if return_individual_score:
            return score, score_all_landmarks
        else:
            return score

    def compute_scores_neighborhood_samples(self, params, dxs, dys, dzs, indices_m=None):
        pool = Pool(processes=12)
        scores = pool.map(lambda (dx, dy, dz): self.compute_score(params + (0.,0.,0., dx, 0.,0.,0., dy, 0.,0.,0., dz), indices_m=indices_m),
                        zip(dxs, dys, dzs))
        pool.close()
        pool.join()
        return scores

    
    def compute_scores_neighborhood_samples_rotation(self, params, dtheta_xys=None, dtheta_yzs=None, dtheta_xzs=None, indices_m=None):
        pool = Pool(processes=12)
        
        if dtheta_xys is not None:
            n = len(dtheta_xys)
        elif dtheta_yzs is not None:
            n = len(dtheta_yzs)
        elif dtheta_xzs is not None:
            n = len(dtheta_xzs)
            
        if dtheta_xys is None:
            dtheta_xys = np.zeros((n,))
        if dtheta_yzs is None:
            dtheta_yzs = np.zeros((n,))
        if dtheta_xzs is None:
            dtheta_xzs = np.zeros((n,))
        
        scores = pool.map(lambda (dtheta_xy, dtheta_yz, dtheta_xz): self.compute_score(rotate_transform_vector(params, theta_xy=dtheta_xy, theta_yz=dtheta_yz, theta_xz=dtheta_xz), indices_m=indices_m),
                        zip(dtheta_xys, dtheta_yzs, dtheta_xzs))
        pool.close()
        pool.join()
        return scores    

    def compute_scores_neighborhood_grid(self, params, dxs, dys, dzs, dtheta_xys=None, indices_m=None):
        """
        Args:
            params ((12,)-array): the parameter vector around which the neighborhood is taken.
        """

        from itertools import product

        # scores = np.reshape([self.compute_score(params + (0.,0.,0., dx, 0.,0.,0., dy, 0.,0.,0., dz), indices_m=indices_m)
        #                     for dx, dy, dz in product(dxs, dys, dzs)],
        #                     (dxs.size, dys.size, dzs.size))

        #parallel
        pool = Pool(processes=12)
        if dtheta_xys is None:
            scores = pool.map(lambda (dx, dy, dz): self.compute_score(params + (0.,0.,0., dx, 0.,0.,0., dy, 0.,0.,0., dz), indices_m=indices_m),
                        product(dxs, dys, dzs))
        else:
            scores = pool.map(lambda (tx, ty, tz, theta_xy): self.compute_score(affine_components_to_vector(tx,ty,tz,theta_xy), indices_m=indices_m), product(dxs, dys, dzs, dtheta_xys))
        pool.close()
        pool.join()

        # scores = np.reshape(Parallel(n_jobs=12)(delayed(compute_score)(params + (0.,0.,0., dx, 0.,0.,0., dy, 0.,0.,0., dz))
        #                                         for dx, dy, dz in product(dxs, dys, dzs)),
        #                     (dxs.size, dys.size, dzs.size))

        return scores

    def compute_scores_neighborhood_random_rotation(self, params, n, std_theta_xy=0, std_theta_xz=0, std_theta_yz=0, indices_m=None):
        
        random_theta_xys = np.random.uniform(-1., 1., (n,)) * std_theta_xy
        random_theta_yzs = np.random.uniform(-1., 1., (n,)) * std_theta_yz
        random_theta_xzs = np.random.uniform(-1., 1., (n,)) * std_theta_xz
        
        # scores = [self.compute_score(params + dp, indices_m=indices_m) for dp in dparams]
        
        random_params = [rotate_transform_vector(params, theta_xy=theta_xy, theta_yz=theta_yz, theta_xz=theta_xz) 
                        for theta_xy, theta_yz, theta_xz in zip(random_theta_xys, random_theta_yzs, random_theta_xzs)]

        #parallel
        pool = Pool(processes=NUM_CORES/2)
        scores = pool.map(lambda p: self.compute_score(p, indices_m=indices_m), random_params)
        pool.close()
        pool.join()
        
        return scores
    
    def compute_scores_neighborhood_random(self, params, n, stds, indices_m=None):

        dparams = np.random.uniform(-1., 1., (n, len(stds))) * stds
        # scores = [self.compute_score(params + dp, indices_m=indices_m) for dp in dparams]

        #parallel
        pool = Pool(processes=12)
        scores = pool.map(lambda dp: self.compute_score(params + dp, indices_m=indices_m), dparams)
        pool.close()
        pool.join()

        # parallelism not working yet, unless put large instance members in global variable
    #     scores = Parallel(n_jobs=12)(delayed(aligner.compute_score)(params + dp) for dp in dparams)

        return scores

    def compute_hessian(self, T, indices_m=None, step=None):
        """Compute Hessian."""

        if indices_m is None:
            indices_m = self.all_indices_m

        import numdifftools as nd

        if step is None:
            step = np.r_[1e-1, 1e-1, 1e-1, 10, 1e-1, 1e-1, 1e-1, 10, 1e-1, 1e-1, 1e-1, 10]

        h = nd.Hessian(self.compute_score, step=step)
        H = h(T.flatten())
        return H

    
    def grid_search(self, grid_search_iteration_number, indices_m=None, init_n=1000, parallel=True,
                    std_tx=100, std_ty=100, std_tz=30, std_theta_xy=np.deg2rad(60),
                    return_best_score=True,
                    eta=3., stop_radius_voxel=10):
        """Grid search.

         Args:
             grid_search_iteration_number (int): number of iteration
             eta: sample number and sigma = initial value * np.exp(-iter/eta), default = 3.
         Returns:
             params_best_upToNow ((12,) float array): found parameters
        """
    
        params_best_upToNow = (0, 0, 0, 0)
        score_best_upToNow = -np.inf

        if indices_m is None:
            indices_m = self.all_indices_m

        for iteration in range(grid_search_iteration_number):

            # self.logger.info('grid search iteration %d', iteration)

            init_tx, init_ty, init_tz, init_theta_xy = params_best_upToNow

            # n = int(init_n*np.exp(-iteration/eta))
            n = init_n

            sigma_tx = std_tx*np.exp(-iteration/eta)
            sigma_ty = std_ty*np.exp(-iteration/eta)
            sigma_tz = std_tz*np.exp(-iteration/eta)
            sigma_theta_xy = std_theta_xy*np.exp(-iteration/eta)

            sys.stderr.write('sigma_tx: %.2f (voxel), sigma_ty: %.2f, sigma_tz: %.2f, sigma_theta_xy: %.2f (deg), n:%d\n' % \
            (sigma_tx, sigma_ty, sigma_tz, np.rad2deg(sigma_theta_xy), n))
            
            tx_grid = init_tx + sigma_tx * np.linspace(-1,1,n)
            ty_grid = init_ty + sigma_ty * np.linspace(-1,1,n)
            tz_grid = init_tz + sigma_tz * np.linspace(-1,1,n)
            # theta_xy_grid = init_theta_xy + sigma_theta_xy * np.linspace(-1,1,n)
            theta_xy_grid = [0]

            # samples = np.c_[tx_grid, ty_grid, tz_grid, theta_xy_grid]
            
            #############
            
            t = time.time()
            
            scores = self.compute_scores_neighborhood_grid(np.array([1,0,0,0,0,1,0,0,0,0,1,0]), 
                                                   dxs=tx_grid, dys=ty_grid, dzs=tz_grid, dtheta_xys=theta_xy_grid,
                                                   indices_m=indices_m)    
            i_best = np.argmax(scores)
            score_best = scores[i_best]
            i_tx, i_ty, i_tz, i_thetaxy = np.unravel_index(i_best, (len(tx_grid), len(ty_grid), len(tz_grid), len(theta_xy_grid)))
            tx_best = tx_grid[i_tx]
            ty_best = ty_grid[i_ty]
            tz_best = tz_grid[i_tz]
            theta_xy_best = theta_xy_grid[i_thetaxy]
            
            # # empirical speedup 7x
            # # parallel
            # if parallel:
            #     pool = Pool(processes=8)
            #     scores = pool.map(lambda (tx, ty, tz, theta_xy): self.compute_score(affine_components_to_vector(tx,ty,tz,theta_xy),
            #                                             indices_m=indices_m, tf_type='affine'), samples)
            #     pool.close()
            #     pool.join()
            # else:
            # # serial
            #     scores = [self.compute_score(affine_components_to_vector(tx,ty,tz,theta_xy), indices_m=indices_m, tf_type='affine')
            #                 for tx, ty, tz, theta_xy in samples]

            sys.stderr.write('grid search: %f seconds\n' % (time.time() - t)) # ~23s
            
            sys.stderr.write('tx_best: %.2f (voxel), ty_best: %.2f, tz_best: %.2f, theta_xy_best: %.2f (deg), score=%f\n' % \
            (tx_best, ty_best, tz_best, np.rad2deg(theta_xy_best), score_best))

            if score_best > score_best_upToNow:
                # self.logger.info('%f %f', score_best_upToNow, score_best)
                sys.stderr.write('New best: %f %f\n' % (score_best_upToNow, score_best))

                score_best_upToNow = score_best
                params_best_upToNow = tx_best, ty_best, tz_best, theta_xy_best

            if sigma_tx < stop_radius_voxel:
                # if sigma is reduced to smaller than 10 voxels, abort
                break

                # self.logger.info('%f %f %f', tx_best, ty_best, tz_best)
        sys.stderr.write('params_best_upToNow: %f %f %f %f\n' % (tx_best, ty_best, tz_best, theta_xy_best))

        if return_best_score:
            return params_best_upToNow, score_best_upToNow
        else:
            return params_best_upToNow


#     def grid_search(self, grid_search_iteration_number, indices_m=None, init_n=1000, parallel=True,
#                     std_tx=100, std_ty=100, std_tz=30, std_theta_xy=np.deg2rad(60),
#                     return_best_score=True,
#                     eta=3.):
#         """Grid search.

#         Args:
#             grid_search_iteration_number (int): number of iteration
#             eta: sample number and sigma = initial value * np.exp(-iter/eta), default = 3.

#         Returns:
#             params_best_upToNow ((12,) float array): found parameters

#         """
#         params_best_upToNow = (0, 0, 0, 0)
#         params_secondbest_upToNow = (0,0,0,0)
#         score_best_upToNow = -np.inf

#         if indices_m is None:
#             indices_m = self.all_indices_m

#         for iteration in range(grid_search_iteration_number):

#             # self.logger.info('grid search iteration %d', iteration)

#             init_tx, init_ty, init_tz, init_theta_xy = params_best_upToNow

#             n = int(init_n*np.exp(-iteration/eta)) / 2

#             sigma_tx = std_tx*np.exp(-iteration/eta)
#             sigma_ty = std_ty*np.exp(-iteration/eta)
#             sigma_tz = std_tz*np.exp(-iteration/eta)
#             sigma_theta_xy = std_theta_xy*np.exp(-iteration/eta)

#             tx_grid = init_tx + sigma_tx * np.r_[0, (2 * np.random.random(n) - 1)]
#             ty_grid = init_ty + sigma_ty * np.r_[0, (2 * np.random.random(n) - 1)]
#             tz_grid = init_tz + sigma_tz * np.r_[0, (2 * np.random.random(n) - 1)]
#             theta_xy_grid = init_theta_xy + sigma_theta_xy * np.r_[0, (2 * np.random.random(n) - 1)]

#             samples = np.c_[tx_grid, ty_grid, tz_grid, theta_xy_grid]
            
#             #######################
#             init_tx, init_ty, init_tz, init_theta_xy = params_secondbest_upToNow

#             n = int(init_n*np.exp(-iteration/eta)) / 2

#             sigma_tx = std_tx*np.exp(-iteration/eta)
#             sigma_ty = std_ty*np.exp(-iteration/eta)
#             sigma_tz = std_tz*np.exp(-iteration/eta)
#             sigma_theta_xy = std_theta_xy*np.exp(-iteration/eta)

#             tx_grid = init_tx + sigma_tx * np.r_[0, (2 * np.random.random(n) - 1)]
#             ty_grid = init_ty + sigma_ty * np.r_[0, (2 * np.random.random(n) - 1)]
#             tz_grid = init_tz + sigma_tz * np.r_[0, (2 * np.random.random(n) - 1)]
#             theta_xy_grid = init_theta_xy + sigma_theta_xy * np.r_[0, (2 * np.random.random(n) - 1)]

#             samples = np.concatenate([samples, np.c_[tx_grid, ty_grid, tz_grid, theta_xy_grid]])

#             ###############################
            
#             t = time.time()

#             # empirical speedup 7x
#             # parallel
#             if parallel:
#                 pool = Pool(processes=8)
#                 scores = pool.map(lambda (tx, ty, tz, theta_xy): self.compute_score(affine_components_to_vector(tx,ty,tz,theta_xy),
#                                                         indices_m=indices_m, tf_type='affine'), samples)
#                 pool.close()
#                 pool.join()
#             else:
#             # serial
#                 scores = [self.compute_score(affine_components_to_vector(tx,ty,tz,theta_xy), indices_m=indices_m, tf_type='affine')
#                             for tx, ty, tz, theta_xy in samples]

#             sys.stderr.write('grid search: %f seconds\n' % (time.time() - t)) # ~23s

#             sample_indices_sorted = np.argsort(scores)[::-1]
            
#             sample_index_best = sample_indices_sorted[0]
#             tx_best, ty_best, tz_best, theta_xy_best = samples[sample_index_best]
#             score_best = scores[sample_index_best]
            
#             sys.stderr.write('tx_best: %.2f (voxel), ty_best: %.2f, tz_best: %.2f, theta_xy_best: %.2f (deg), score=%.2f\n' % \
#             (tx_best, ty_best, tz_best, np.rad2deg(theta_xy_best), score_best))
#             sys.stderr.write('sigma_tx: %.2f (voxel), sigma_ty: %.2f, sigma_tz: %.2f, sigma_theta_xy: %.2f (deg)\n' % \
#             (sigma_tx, sigma_ty, sigma_tz, np.rad2deg(sigma_theta_xy)))

#             if score_best > score_best_upToNow:
#                 # self.logger.info('%f %f', score_best_upToNow, score_best)
#                 sys.stderr.write('New best: %f %f\n' % (score_best_upToNow, score_best))

#                 score_best_upToNow = score_best
#                 params_best_upToNow = tx_best, ty_best, tz_best, theta_xy_best
                
#                 for i in sample_indices_sorted[1:]:
#                     tx, ty, tz, theta_xy = samples[i]
#                     # s = scores[i]
#                     if np.linalg.norm((tx-tx_best, ty-ty_best, tz-tz_best)) > 30:
#                         tx_secondbest, ty_secondbest, tz_secondbest, theta_xy_secondbest = (tx, ty, tz, theta_xy)
                        
#                         score_secondbest_upToNow = score_secondbest
#                         params_secondbest_upToNow = tx_secondbest, ty_secondbest, tz_secondbest, theta_xy_secondbest
                        
#                         sys.stderr.write('New second_best: tx_secondbest: %.2f (voxel), ty_secondbest: %.2f, tz_secondbest: %.2f, theta_xy_secondbest: %.2f (deg), score=%.2f\n' % \
#             (tx_secondbest, ty_secondbest, tz_secondbest, np.rad2deg(theta_xy_secondbest), score_secondbest_upToNow))
                        
#                         break
#                 sys.stderr.write("i=%d\n" % i)

#             if sigma_tx < 10:
#                 break

#                 # self.logger.info('%f %f %f', tx_best, ty_best, tz_best)
#         sys.stderr.write('params_best_upToNow: %f %f %f %f\n' % (tx_best, ty_best, tz_best, theta_xy_best))

#         if return_best_score:
#             return params_best_upToNow, score_best_upToNow
#         else:
#             return params_best_upToNow


    def do_grid_search(self, grid_search_iteration_number=10, grid_search_sample_number=1000,
                      std_tx=100, std_ty=100, std_tz=30, std_theta_xy=np.deg2rad(30),
                       grid_search_eta=3., stop_radius_voxel=10,
                      indices_m=None):
        
        if indices_m is None:
            indices_m = self.all_indices_m
        
        (tx_best, ty_best, tz_best, theta_xy_best), grid_search_score = self.grid_search(grid_search_iteration_number, indices_m=indices_m,
                                                    init_n=grid_search_sample_number,
                                                    std_tx=std_tx, std_ty=std_ty, std_tz=std_tz, std_theta_xy=std_theta_xy,
                                                                                         eta=grid_search_eta, stop_radius_voxel=stop_radius_voxel,
                                                    return_best_score=True)
        # T = np.r_[1,0,0, tx_best, 0,1,0, ty_best, 0,0,1, tz_best]
        T = affine_components_to_vector(tx_best, ty_best, tz_best, theta_xy_best)
        return T, grid_search_score

    def optimize(self, tf_type, init_T=None, label_weights=None, \
                # grid_search_iteration_number=0, grid_search_sample_number=1000,
                grad_computation_sample_number=None,
                max_iter_num=1000, history_len=200, terminate_thresh_rot=.005, \
                 terminate_thresh_trans=.4, \
                indices_m=None, lr1=None, lr2=None, full_lr=None,
                # std_tx=100, std_ty=100, std_tz=30, std_theta_xy=np.deg2rad(30),
                 # grid_search_eta=3.,
                reg_weights=None,
                epsilon=1e-8,
                affine_scaling_limits=None):
        """Optimize.
        Objective = texture score - reg_weights[0] * tx**2 - reg_weights[1] * ty**2 - reg_weights[2] * tz**2

        Args:
            reg_weights: for (tx,ty,tz)
        """

        if indices_m is None:
            indices_m = self.all_indices_m

        if label_weights is not None:
            self.set_label_weights(label_weights)

        if reg_weights is not None:
            self.set_regularization_weights(reg_weights)

        if tf_type == 'rigid':
            grad_historical = np.zeros((6,))
            sq_updates_historical = np.zeros((6,))
            if lr1 is None:
                lr1 = 10.
            if lr2 is None:
                lr2 = 1e-1 # for Lie optimization, lr2 cannot be zero, otherwise causes error in computing scores.
        elif tf_type == 'affine':
            grad_historical = np.zeros((12,))
            sq_updates_historical = np.zeros((12,))
            if lr1 is None:
                lr1 = 10
            if lr2 is None:
                lr2 = 1e-1
        elif tf_type == 'bspline':
            grad_historical = np.zeros((self.n_ctrl*3,))
            sq_updates_historical = np.zeros((self.n_ctrl*3,))
            if lr1 is None:
                lr1 = 10
        else:
            raise Exception('Type must be either rigid or affine.')
            

        if init_T is None:
            if tf_type == 'affine' or tf_type == 'rigid':
                T = np.r_[1,0,0,0,0,1,0,0,0,0,1,0]
            elif tf_type == 'bspline':
                T = np.zeros((self.n_ctrl*3,))
        else:
            T = init_T
            
        score_best = -np.inf
        scores = []
        self.Ts = [T]

        for iteration in range(max_iter_num):

            # t = time.time()

            sys.stderr.write('\niteration %d\n' % iteration)

            t = time.time()

            if tf_type == 'rigid':
                # lr1, lr2 = (.1, 1e-2) # lr2 cannot be zero, otherwise causes error in computing scores.

                if full_lr is not None:
                    lr = full_lr
                else:
                    lr = np.r_[lr1,lr1,lr1,lr2,lr2,lr2]

                new_T, s, grad_historical, sq_updates_historical = self.step_lie(T, lr=lr,
                    grad_historical=grad_historical, sq_updates_historical=sq_updates_historical,
                    verbose=False, num_samples=grad_computation_sample_number,
                    indices_m=indices_m,
                    epsilon=epsilon)
                
            elif tf_type == 'affine':

                if full_lr is not None:
                    lr = full_lr
                else:
                    lr = np.r_[lr2, lr2, lr2, lr1, lr2, lr2, lr2, lr1, lr2, lr2, lr2, lr1]

                new_T, s, grad_historical, sq_updates_historical = self.step_gd(T, lr=lr, \
                                grad_historical=grad_historical, sq_updates_historical=sq_updates_historical,
                                indices_m=indices_m, tf_type='affine',
                                                                           num_samples=grad_computation_sample_number,
                                                                               scaling_limits=affine_scaling_limits)
                
            elif tf_type == 'bspline':
                
                new_T, s, grad_historical, sq_updates_historical = self.step_gd(T, lr=lr1, \
                                grad_historical=grad_historical, sq_updates_historical=sq_updates_historical,
                                indices_m=indices_m, tf_type='bspline',
                                                                           num_samples=grad_computation_sample_number)

            else:
                raise Exception('Type must be either rigid or affine.')                
                
            sys.stderr.write('step: %.2f seconds\n' % (time.time() - t))
            sys.stderr.write('current score: %f\n' % s)
                
            if tf_type == 'rigid' or tf_type == 'affine':
                sys.stderr.write('new_T: %s\n' % new_T[[3,7,11]])
                sys.stderr.write('det: %.2f\n' % np.linalg.det(new_T.reshape((3,4))[:3, :3]))
            elif tf_type == 'bspline':
                sys.stderr.write('min: %.2f, max: %.2f\n' % (new_T.min(), new_T.max()))
                
            scores.append(s)
            
            if np.isnan(s):
                break

            self.Ts.append(new_T)

            # sys.stderr.write('%f seconds\n' % (time.time()-t)) # 1.77s/iteration
            
            Ts = np.array(self.Ts)
            
            if tf_type == 'affine' or tf_type == 'rigid':
                if iteration > history_len:
                    if np.all([np.std(Ts[iteration-history_len:iteration, [3,7,11]], axis=0) < terminate_thresh_trans]) & \
                    np.all([np.std(Ts[iteration-history_len:iteration, [0,1,2,4,5,6,8,9,10]], axis=0) < terminate_thresh_rot]):
                        break

            if s > score_best:
                best_gradient_descent_params = T
                score_best = s
                
            T = new_T

        # if grid_search_iteration_number > 0:
        #     if scores[-1] <= grid_search_score:
        #         sys.stderr.write('Gradient descent does not converge to higher than grid search score. Likely stuck at local minima.\n')

        return best_gradient_descent_params, scores

    def step_lie(self, T, lr, grad_historical, sq_updates_historical, verbose=False, num_samples=1000, indices_m=None,
                epsilon=1e-8):
        """
        One optimization step over Lie group SE(3).

        Args:
            T ((12,) vector): flattened vector of 3x4 transform matrix
            lr ((12,) vector): learning rate
            grad_historical ((12,) vector): accumulated gradiant magnitude, for Adagrad or AdaDelta
            sq_updates_historical: accumulated squared update magnitude, for AdaDelta

        Returns:
            (tuple): tuple containing:

                new_T ((12,) vector): the new parameters
                score (float): current score
                grad_historical ((12,) vector): new accumulated gradient magnitude, used for Adagrad
        """

        if indices_m is None:
            indices_m = self.all_indices_m

        # print 'T:', np.ravel(T)
        # t = time.time()
        score, grad = self.compute_score_and_gradient(T, tf_type='rigid', num_samples=num_samples, indices_m=indices_m)
        # sys.stderr.write("compute_score_and_gradient: %.2f s\n" % (time.time() - t))
        # grad is (6,)-array
        
        # print 'score:', score
        # print 'grad:', grad

        # # AdaGrad Rule
        grad_historical += grad**2
        grad_adjusted = grad / np.sqrt(grad_historical + epsilon)
        # Note: It is wrong to do: grad_adjusted = grad /  (np.sqrt(grad_historical) + epsilon)
        v_opt = lr * grad_adjusted # no minus sign because maximizing

        # AdaDelta Rule
        # Does not work, very unstable!

        # gamma = .9
        # epsilon = 1e-8
        # grad_historical = gamma * grad_historical + (1-gamma) * grad**2
        # v_opt = np.sqrt(sq_updates_historical + epsilon)/np.sqrt(grad_historical + epsilon)*grad
        # sq_updates_historical = gamma * sq_updates_historical + (1-gamma) * v_opt**2

        # print 'grad = %s' % grad
        # print 'grad_historical = %s' % grad_historical
        # print 'sq_updates_historical = %s' % sq_updates_historical

        theta = np.sqrt(np.sum(v_opt[3:]**2))
        assert theta < np.pi

        exp_w, Vt = matrix_exp_v(v_opt)
        # print 'Vt', Vt

        Tm = np.reshape(T, (3,4))
        t = Tm[:, 3]
        # print 't', t
        R = Tm[:, :3]

        R_new = np.dot(exp_w, R)
        # t_new = np.dot(exp_w, t) + Vt
        t_new = t + Vt
        # print 't_new', t_new

        return np.column_stack([R_new, t_new]).flatten(), score, grad_historical, sq_updates_historical
    
    
    def step_gd(self, T, lr, grad_historical, sq_updates_historical, tf_type, surround=False, surround_weight=2., num_samples=None, indices_m=None, scaling_limits=None):
        """
        One optimization step using gradient descent with Adagrad.

        Args:
            T ((12,) vector): flattened vector of 3x4 transform matrix.
            lr ((12,) vector): learning rate
            dMdA_historical ((12,) vector): accumulated gradiant magnitude, for Adagrad

        Returns:
            (tuple): tuple containing:

                new_T ((12,) vector): the new parameters
                score (float): current score
                dMdv_historical ((12,) vector): new accumulated gradient magnitude, used for Adagrad
        """

        if indices_m is None:
            indices_m = self.all_indices_m

        score, grad = self.compute_score_and_gradient(T, tf_type=tf_type, num_samples=num_samples, indices_m=indices_m)
        
        # if surround:
        #     s_surr, dMdA_surr = compute_score_and_gradient(T, name, surround=True, num_samples=num_samples)
        #     dMdA -= surround_weight * dMdA_surr
        #     score -= surround_weight * s_surr

        # AdaGrad Rule
        grad_historical += grad**2
        grad_adjusted = grad / np.sqrt(grad_historical + 1e-10)
        new_T = T + lr*grad_adjusted
        
        # Constrain the transform
        if tf_type == 'bspline':
            # Limit the deformation at all control points to be less than 100.
            new_T = np.sign(new_T) * np.minimum(np.abs(new_T), 100)
        elif tf_type == 'affine':
            # pass
            if scaling_limits is not None:                
                new_T[0] = np.sign(new_T[0]) * np.minimum(np.maximum(np.abs(new_T[0]), scaling_limits[0]), scaling_limits[1])
                new_T[5] = np.sign(new_T[5]) * np.minimum(np.maximum(np.abs(new_T[5]), scaling_limits[0]), scaling_limits[1])
                new_T[10] = np.sign(new_T[10]) * np.minimum(np.maximum(np.abs(new_T[10]), scaling_limits[0]), scaling_limits[1])
            else:
                new_T[0] = np.sign(new_T[0]) * np.abs(new_T[0])
                new_T[5] = np.sign(new_T[5]) * np.abs(new_T[5])
                new_T[10] = np.sign(new_T[10]) * np.abs(new_T[10])
            

        # AdaDelta Rule
        # gamma = .9
        # epsilon = 1e-10
        # grad_historical = gamma * grad_historical + (1-gamma) * grad**2
        # update = np.sqrt(sq_updates_historical + epsilon)/np.sqrt(grad_historical + epsilon)*grad
        # new_T = T + update
        # sq_updates_historical = gamma * sq_updates_historical + (1-gamma) * update**2

        sys.stderr.write("in T: %.2f %.2f %.2f, out T: %.2f %.2f %.2f\n" % (T[3], T[7], T[11], new_T[3], new_T[7], new_T[11]))
        return new_T, score, grad_historical, sq_updates_historical


from scipy.optimize import approx_fprime

def hessian ( x0, f, epsilon=1.e-5, linear_approx=False, *args ):
    """
    A numerical approximation to the Hessian matrix of cost function at
    location x0 (hopefully, the minimum)

    Parameters
    ----------
    x0 :
        point
    f :
        cost function

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


def find_contour_points_3d(labeled_volume, along_direction, positions=None, sample_every=10):
    """
    This function uses multiple processes.

    Args:
        labeled_volume (3D ndarray of int): integer-labeled volume.
        along_direction (str): 'x', 'y' or 'z'.
        positions (None or list of int): if None, use all positions of input volume, from 0 to the depth of volume.

    Returns:
        contours (dict {int: (n,2)-ndarray}): {voxel position: contour vertices (second dim, first dim)}.
    """

    nproc = NUM_CORES

    if along_direction == 'z':
        if positions is None:
            positions = range(0, labeled_volume.shape[2])
    elif along_direction == 'x':
        if positions is None:
            positions = range(0, labeled_volume.shape[1])
    elif along_direction == 'y':
        if positions is None:
            positions = range(0, labeled_volume.shape[0])

    def find_contour_points_slice(p):
        if along_direction == 'x':
            vol_slice = labeled_volume[:, p, :]
        if along_direction == 'y':
            vol_slice = labeled_volume[p, :, :]
        if along_direction == 'z':
            vol_slice = labeled_volume[:, :, p]

        cnts = find_contour_points(vol_slice.astype(np.uint8), sample_every=sample_every)
        if len(cnts) == 0 or 1 not in cnts:
            sys.stderr.write('No contour of reconstructed volume is found at position %d.\n' % p)
            return
        else:
            if len(cnts[1]) > 1:
                sys.stderr.write('%s contours of reconstructed volume is found at position %d (%s). Use the longest one.\n' % (len(cnts[1]), p, map(len, cnts[1])))
                cnt = np.array(cnts[1][np.argmax(map(len, cnts[1]))])
            else:
                cnt = np.array(cnts[1][0])
            return cnt

    pool = Pool(nproc)
    contours = dict(zip(positions, pool.map(find_contour_points_slice, positions)))
    pool.close()
    pool.join()

    contours = {p: cnt for p, cnt in contours.iteritems() if cnt is not None}

    return contours

def find_contour_points(labelmap, sample_every=10, min_length=0):
    """
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

def get_structure_contours_from_aligned_atlas(volumes, volume_origin, sections, downsample_factor=32, level=.5, sample_every=1):
    """
    Re-section atlas volumes and obtain structure contours on each section.

    Args:
        volumes (dict of 3D ndarrays of float): {structure: volume}. volume is a 3d array of probability values.
        downsample_factor (int): the downscale factor of input volumes. Output contours are in original resolution.
        volume_origin (tuple): (xmin_vol_f, ymin_vol_f, zmin_vol_f) relative to cropped image volume.
        level (float):

    Returns:
        dict: {section: {name_s: (n,2)-ndarray}}. The vertex coordinates are relative to cropped image volume and in lossless resolution.
    """

    from metadata import XY_PIXEL_DISTANCE_LOSSLESS, SECTION_THICKNESS
    from collections import defaultdict

    # estimate mapping between z and section
    xy_pixel_distance_downsampled = XY_PIXEL_DISTANCE_LOSSLESS * downsample_factor
    voxel_z_size = SECTION_THICKNESS / xy_pixel_distance_downsampled

    xmin_vol_f, ymin_vol_f, zmin_vol_f = volume_origin

    structure_contours = defaultdict(dict)

    # Multiprocess is not advisable here because volumes must be duplicated across processes which is very RAM heavy.

#     def compute_contours_one_section(sec):
#         sys.stderr.write('Computing structure contours for section %d...\n' % sec)
#         z = int(np.round(voxel_z_size * (sec - 1) - zmin_vol_f))
#         contours_one_section = {}
#         # Find moving volume annotation contours
#         for name_s, vol in volumes.iteritems():
#             cnts = find_contours(vol[..., z], level=level) # rows, cols
#             for cnt in cnts:
#                 # r,c to x,y
#                 contours_one_section[name_s] = cnt[:,::-1] + (xmin_vol_f, ymin_vol_f)
#         return contours_one_section

#     pool = Pool(NUM_CORES/2)
#     structuer_contours = dict(zip(sections, pool.map(compute_contours_one_section, sections)))
#     pool.close()
#     pool.join()

    for sec in sections:
        sys.stderr.write('Computing structure contours for section %d...\n' % sec)
        z = int(np.round(voxel_z_size * (sec - 1) - zmin_vol_f))
        for name_s, vol in volumes.iteritems():
            if np.count_nonzero(vol[..., z]) == 0:
                continue

            cnts_rowcol = find_contours(vol[..., z], level=level)

            if len(cnts_rowcol) == 0:
                sys.stderr.write('Some probability mass of %s are on section %d but no contour is extracted at level=%.2f.\n' % (name_s, sec, level))
            else:
                if len(cnts_rowcol) > 1:
                    sys.stderr.write('%d contours (%s) of %s is extracted at level=%.2f on section %d. Keep only the longest.\n' % (len(cnts_rowcol), map(len, cnts_rowcol), name_s, level, sec))
                best_cnt = cnts_rowcol[np.argmax(map(len, cnts_rowcol))]
                contours_on_cropped_tb = best_cnt[:, ::-1][::sample_every] + (xmin_vol_f, ymin_vol_f)
                structure_contours[sec][name_s] = contours_on_cropped_tb * downsample_factor

    return structure_contours


# def extract_contours_from_labeled_volume(stack, volume,
#                             section_z_map=None,
#                             downsample_factor=None,
#                             volume_limits=None,
#                             labels=None, extrapolate_num_section=0,
#                             force=True, filepath=None):
#     """
#     Extract labeled contours from a labeled volume.
#     """

#     if volume == 'localAdjusted':
#         volume = bp.unpack_ndarray_file(volume_dir + '/%(stack)s/%(stack)s_localAdjustedVolume.bp'%{'stack':stack})
#     elif volume == 'globalAligned':
#         volume = bp.unpack_ndarray_file(volume_dir + '/%(stack)s/%(stack)s_atlasProjectedVolume.bp'%{'stack':stack})
#     else:
#         raise 'Volume unknown.'

#     if filepath is None:
#         filepath = volume_dir + '/initCntsAllSecs_%s.pkl' % stack

#     if os.path.exists(filepath) and not force:
#         init_cnts_allSecs = pickle.load(open(filepath, 'r'))
#     else:
#         if volume_limits is None:
#             volume_xmin, volume_xmax, volume_ymin, volume_ymax, volume_zmin, volume_zmax = \
#             np.loadtxt(os.path.join(volume_dir, '%(stack)s/%(stack)s_scoreVolume_limits.txt' % {'stack': stack}), dtype=np.int)

#         if section_z_map is None:
#             assert downsample_factor is not None, 'Because section_z_map is not given, must specify downsample_factor.'
#             z_section_map = find_z_section_map(stack, volume_zmin, downsample_factor=downsample_factor)
#             section_z_map = {sec: z for z, sec in z_section_map.iteritems()}

#         init_cnts_allSecs = {}

#         first_detect_sec, last_detect_sec = detect_bbox_range_lookup[stack]

#         for sec in range(first_detect_sec, last_detect_sec+1):

#             z = section_z_map[sec]
#             projected_annotation_labelmap = volume[..., z]

#             init_cnts = find_contour_points(projected_annotation_labelmap) # downsampled 16
#             init_cnts = dict([(labels[label_ind], (cnts[0]+(volume_xmin, volume_ymin))*2)
#                               for label_ind, cnts in init_cnts.iteritems()])

#             # extend contour to copy annotations of undetected classes from neighbors
#             if extrapolate_num_section > 0:

#                 sss = np.empty((2*extrapolate_num_section,), np.int)
#                 sss[1::2] = -np.arange(1, extrapolate_num_section+1)
#                 sss[::2] = np.arange(1, extrapolate_num_section+1)

#                 Ls = []
#                 for ss in sss:
#                     sec2 = sec + ss
#                     z2 = section_z_map[sec2]
#                     if z2 >= volume.shape[2] or z2 < 0:
#                         continue

#                     init_cnts2 = find_contour_points(volume[..., z2]) # downsampled 16
#                     init_cnts2 = dict([(labels[label_ind], (cnts[0]+(volume_xmin, volume_ymin))*2)
#                                       for label_ind, cnts in init_cnts2.iteritems()])
#                     Ls.append(init_cnts2)

#                 for ll in Ls:
#                     for l, c in ll.iteritems():
#                         if l not in init_cnts:
#                             init_cnts[l] = c

#             init_cnts_allSecs[sec] = init_cnts

#         pickle.dump(init_cnts_allSecs, open(filepath, 'w'))

#     return init_cnts_allSecs


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

        NuPx_allTestPts = np.array([[N(ctrl_x/float(interval), x/float(interval)) for testPt_i, (x, y, z) in enumerate(pts_centered)]
                                    for ctrlXInterval_i, ctrl_x in enumerate(ctrl_x_intervals_centered)])
        
        NvPy_allTestPts = np.array([[N(ctrl_y/float(interval), y/float(interval)) for testPt_i, (x, y, z) in enumerate(pts_centered)]
                                    for ctrlYInterval_i, ctrl_y in enumerate(ctrl_y_intervals_centered)])
        
        NwPz_allTestPts = np.array([[N(ctrl_z/float(interval), z/float(interval)) for testPt_i, (x, y, z) in enumerate(pts_centered)]
                                    for ctrlZInterval_i, ctrl_z in enumerate(ctrl_z_intervals_centered)])

        sys.stderr.write("Compute NuPx/NvPy/NwPz: %.2f seconds.\n" % (time.time() - t))

        # print NuPx_allTestPts.shape, NvPy_allTestPts.shape, NwPz_allTestPts.shape
        # (9, 157030) (14, 157030) (8, 157030)
        # (n_ctrlx, n_test)

        t = time.time()
        NuNvNw_allTestPts = np.array([np.ravel(np.tensordot(np.tensordot(NuPx_allTestPts[:,testPt_i], 
                                                                         NvPy_allTestPts[:,testPt_i], 0), 
                                                            NwPz_allTestPts[:,testPt_i], 0))
                                  for testPt_i in range(len(pts_centered))])
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


def transform_points_polyrigid_inverse(pts_prime, rigid_param_list, anchor_points, sigmas, weights):
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
        nzvoxels_weights = np.array([w*np.exp(-np.sum((pts_prime - ap)**2, axis=1)/sigma**2) \
                            for ap, sigma, w in zip(anchor_points_prime, sigmas, weights)])
    nzvoxels_weights = nzvoxels_weights/nzvoxels_weights.sum(axis=0)
    # nzvoxels_weights[nzvoxels_weights < 1e-1] = 0
    # nzvoxels_weights = nzvoxels_weights/nzvoxels_weights.sum(axis=0)
    # n_components x n_voxels
    
    # print nzvoxels_weights
        
    nzs_m_aligned_to_f = np.array([np.sum([w * (np.dot(Rinv, p) + tinv) for w, Rinv,tinv in zip(ws, Rs_inverse,ts_inverse)], axis=0) 
                                   for p, ws in zip(pts_prime, nzvoxels_weights.T)]).astype(np.int16)
    return nzs_m_aligned_to_f


def transform_points_polyrigid(pts, rigid_param_list, anchor_points, sigmas, weights):
    """
    Transform points by a weighted-average transform.
    
    Args:
        rigid_param_list (list of (12,)-ndarrays): list of rigid transforms
        weights (list of float): weights of component transforms. Only the relative magnitude matters.
    
    Returns:
        ((n,3)-ndarray): transformed points.
    """

    if sigmas[0].ndim == 2: # sigma is covariance matrix
        nzvoxels_weights = np.array([w*np.exp(-mahalanobis_distance_sq(pts, ap, sigma))
                            for ap, sigma, w in zip(anchor_points, sigmas, weights)]) + 1e-6
    elif sigmas[0].ndim == 1: # sigma is a single scalar
        nzvoxels_weights = np.array([w*np.exp(-np.sum((pts - ap)**2, axis=1)/sigma**2) \
                            for ap, sigma, w in zip(anchor_points, sigmas, weights)]) + 1e-6
    # add a small constant to prevent from being rounded to 0.
        
    nzvoxels_weights = nzvoxels_weights/nzvoxels_weights.sum(axis=0)
    # nzvoxels_weights[nzvoxels_weights < 1e-1] = 0
    # nzvoxels_weights = nzvoxels_weights/nzvoxels_weights.sum(axis=0)
    # n_components x n_voxels

    nzs_m_aligned_to_f = np.zeros((len(pts), 3), dtype=np.float16)

    for i, rigid_params in enumerate(rigid_param_list):
        nzs_m_aligned_to_f += nzvoxels_weights[i][:,None] * transform_points_affine(rigid_params, pts=pts).astype(np.float16)

    nzs_m_aligned_to_f = nzs_m_aligned_to_f.astype(np.int16)    
    return nzs_m_aligned_to_f
    
def transform_volume_polyrigid(vol, rigid_param_list, anchor_points, sigmas, weights, out_bbox, fill_holes=True):
    """
    Transform volume by a weighted-average transform.
    
    Args:
        rigid_param_list (list of (12,)-ndarrays): list of rigid transforms

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


def transform_volume_bspline(vol, buvwx, buvwy, buvwz, volume_shape, interval=None, 
                             ctrl_x_intervals=None,
                             ctrl_y_intervals=None,
                             ctrl_z_intervals=None,
                             centroid_m=(0,0,0), centroid_f=(0,0,0),
                            fill_holes=True):
    """
    Transform volume by a B-spline transform.
    
    Args:
        vol (3D-ndarray or 2-tuple): input volume. If tuple, (volume in bbox, bbox).
        volume_shape (3-tuple); xdim,ydim,zdim
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
            raise Exception('transform_volume: Volume must be either float or int.')
        return dense_volume, (nzs_m_xmin_f, nzs_m_xmax_f, nzs_m_ymin_f, nzs_m_ymax_f, nzs_m_zmin_f, nzs_m_zmax_f)
    else:
        return volume_m_aligned_to_f, (nzs_m_xmin_f, nzs_m_xmax_f, nzs_m_ymin_f, nzs_m_ymax_f, nzs_m_zmin_f, nzs_m_zmax_f)

def transform_volume_v2(vol, tf_params, centroid_m=(0,0,0), centroid_f=(0,0,0)):
    """
    First, centroid_m and centroid_f are aligned.
    Then the tranform parameterized by tf_params is applied.
    The resulting volume will have dimension (xdim_f, ydim_f, zdim_f).
    
    coord_f - centroid_f = np.dot(R, (coord_m - centroid_m)) + t

    Args:
        vol (3D ndarray of float or int): the volume to transform. If dtype is int, treated as label volume; if is float, treated as score volume.
        tf_params ((nparam,)-ndarray): flattened vector of transform parameters
        centroid_m (3-tuple): transform center in the volume to transform
        centroid_f (3-tuple): transform center in the result volume.

    Returns:
        (3d array, 6-tuple): resulting volume, bounding box whose coordinates are relative to the input volume.
    """

    nzvoxels_m_temp = parallel_where_binary(vol > 0)
    # "_temp" is appended to avoid name conflict with module level variable defined in registration.py

    nzs_m_aligned_to_f = transform_points_affine(tf_params, pts=nzvoxels_m_temp,
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

    if np.issubdtype(volume_m_aligned_to_f.dtype, np.float):
        dense_volume = fill_sparse_score_volume(volume_m_aligned_to_f)
    elif np.issubdtype(volume_m_aligned_to_f.dtype, np.integer):
        dense_volume = fill_sparse_volume(volume_m_aligned_to_f)
        #dense_volume = volume_m_aligned_to_f
    else:
        raise Exception('transform_volume: Volume must be either float or int.')

    return dense_volume, (nzs_m_xmin_f, nzs_m_xmax_f, nzs_m_ymin_f, nzs_m_ymax_f, nzs_m_zmin_f, nzs_m_zmax_f)


def transform_volume(vol, global_params, centroid_m=(0,0,0), centroid_f=(0,0,0), xdim_f=None, ydim_f=None, zdim_f=None):
    """
    First, centroid_m and centroid_f are aligned.
    Then the tranform parameterized by global_params is applied.
    The resulting volume will have dimension (xdim_f, ydim_f, zdim_f).

    Args:
        vol (3d array): the volume to transform
        global_params (12-tuple): flattened vector of transform parameters
        centroid_m (3-tuple): transform center in the volume to transform
        centroid_f (3-tuple): transform center in the result volume.
        xmin_f (int): if None, this is inferred from the
        ydim_f (int): if None, the
        zdim_f (int): if None, the
    """

    nzvoxels_m_temp = parallel_where_binary(vol > 0)
    # "_temp" is appended to avoid name conflict with module level variable defined in registration.py

    nzs_m_aligned_to_f = transform_points_affine(global_params, pts=nzvoxels_m_temp,
                            c=centroid_m, c_prime=centroid_f).astype(np.int16)

    volume_m_aligned_to_f = np.zeros((ydim_f, xdim_f, zdim_f), vol.dtype)

    xs_f, ys_f, zs_f = nzs_m_aligned_to_f.T

    valid = (xs_f >= 0) & (ys_f >= 0) & (zs_f >= 0) & \
    (xs_f < xdim_f) & (ys_f < ydim_f) & (zs_f < zdim_f)

    xs_m, ys_m, zs_m = nzvoxels_m_temp.T

    volume_m_aligned_to_f[ys_f[valid], xs_f[valid], zs_f[valid]] = \
    vol[ys_m[valid], xs_m[valid], zs_m[valid]]

    del nzs_m_aligned_to_f

    if np.issubdtype(volume_m_aligned_to_f.dtype, np.float):
        dense_volume = fill_sparse_score_volume(volume_m_aligned_to_f)
    elif np.issubdtype(volume_m_aligned_to_f.dtype, np.integer):
        dense_volume = fill_sparse_volume(volume_m_aligned_to_f)
        #dense_volume = volume_m_aligned_to_f
    else:
        raise Exception('transform_volume: Volume must be either float or int.')

    return dense_volume

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
    dense_img = np.zeros_like(img)
    xmin, xmax, ymin, ymax = bbox_2d(img)
    roi = img[ymin:ymax+1, xmin:xmax+1]
    roi_dense_img = np.zeros_like(roi)
    roi_dense_img = closing((roi*255).astype(np.int)/255., disk(1))
    dense_img[ymin:ymax+1, xmin:xmax+1] = roi_dense_img.copy()
    return dense_img

def fill_sparse_score_volume(vol):
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
    from skimage.morphology import binary_closing, ball

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
        vs_padded_filled = binary_closing(vs_padded, ball(closing_element_radius))
        vs_filled = vs_padded_filled[padding:-padding, padding:-padding, padding:-padding]
        volume[ymin:ymax+1, xmin:xmax+1, zmin:zmax+1][vs_filled.astype(np.bool)] = ind

    return volume

def get_grid_mesh_volume(xs, ys, zs, vol_shape, s=1, include_borderline=True):
    """
    Get a boolean volume with grid lines set to True.
    
    Args:
        s (int): the spacing between dots if broken lines are desired.
        
    Returns:
        3D array of boolean
    """
    
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

