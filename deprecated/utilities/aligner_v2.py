import numpy as np
import sys
import os
import time

from skimage.morphology import binary_closing, disk, binary_dilation, binary_erosion
from skimage.measure import grid_points_in_poly, subdivide_polygon, approximate_polygon
from skimage.measure import find_contours, regionprops
from skimage.filters import gaussian

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
from registration_utilities import *

volume_f = None
volume_m = None
volume_f_origin = None
volume_m_origin = None
nzvoxels_m = None
nzvoxels_centered_m = None
grad_f = None
grad_f_origin = None

class Aligner(object):
    def __init__(self, volume_f_, volume_m_=None, nzvoxels_m_=None, centroid_f=None, centroid_m=None, \
                labelIndexMap_m2f=None, label_weights=None, reg_weights=None, zrange=None, nz_thresh=0):
        """
        Variant that takes in two probabilistic volumes.

        Args:
            volume_f_ (dict of (vol, origin)): the fixed probabilistic volume and origin tuples. The frame of origin must be consistent with that of moving volume.
            volume_m_ (dict of (vol, origin)): the moving probabilistic volume and origin tuples. The frame of origin must be consistent with that of fixed volume.
            labelIndexMap_m2f (dict): mapping between moving volume labels and fixed volume labels. dict of {moving label: fixed label}
            label_weights (dict): {numeric label: weight}
            reg_weights (3-array or float): regularization weights for (tx,ty,tz) respectively.
            zrange (2-tuple): If given, only use the portion of both volumes that is between zmin and zmax (inclusive).
            nz_thresh (float): a number above which the score value is used for registration.
        """

        self.labelIndexMap_m2f = labelIndexMap_m2f

        if isinstance(volume_m_, dict): # probabilistic volume
            labels_in_volume_m = set(np.unique(volume_m_.keys()))
        else:
            raise Exception("volume_m_ must be a dict.")
        # else: # annotation volume
        #     labels_in_volume_m = set(np.unique(volume_m_))

        if isinstance(volume_f_, dict): # probabilistic volume
            labels_in_volume_f = set(np.unique(volume_f_.keys()))
        else:
            raise Exception("volume_m_ must be a dict.")
        # else: # annotation volume
        #     labels_in_volume_f = set(np.unique(volume_f_))

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

        global volume_f, volume_f_origin

        if isinstance(volume_f_, dict): # probabilistic volume
            # volume_f = volume_f_
            volume_f = {i: volume_f_[i][0] for i in self.all_indices_f}
            volume_f_origin = {i: volume_f_[i][1].astype(np.int) for i in self.all_indices_f}

        # else: # annotation volume
        #     volume_f = {i: np.zeros_like(volume_f_, dtype=np.float16) for i in self.all_indices_f}
        #     for i in self.all_indices_f:
        #         mask = volume_f_ == i
        #         volume_f[i][mask] = 1.
        #         del mask

        # for i, v in volume_f.iteritems():
        #     volume_f[i] = gaussian(v, 3)

        global volume_m, volume_m_origin

        if isinstance(volume_m_, dict): # probabilistic volume
            # volume_m = volume_m_
            volume_m = {i: volume_m_[i][0] for i in self.all_indices_m}
            volume_m_origin = {i: volume_m_[i][1].astype(np.int) for i in self.all_indices_m}
        # else: # annotation volume; also convert to dict, treated the same as prob. volume
        #     volume_m = {i: np.zeros_like(volume_m_, dtype=np.float16) for i in self.all_indices_m}
        #     for i in self.all_indices_m:
        #         mask = volume_m_ == i
        #         volume_m[i][mask] = 1.
        #         del mask

#         # Slice volumes if arange is specified
#         if zrange is not None:
#             self.zl, self.zh = zrange
#             volume_f = {l: v[..., self.zl:self.zh+1] for l, v in volume_f.iteritems()}
#             volume_m = {l: v[..., self.zl:self.zh+1] for l, v in volume_m.iteritems()}

#             volume_f_sliced = {}
#             for l, v in volume_f.iteritems():
#                 if np.count_nonzero(v) == 0:
#                     sys.stderr.write('Fixed volume %(label)s at z=(%(zl)d,%(zh)d) is empty.\n' % dict(label=l, zl=self.zl, zh=self.zh))
#                 else:
#                     volume_f_sliced[l] = v

#             volume_m_sliced = {}
#             for l, v in volume_m.iteritems():
#                 if np.count_nonzero(v) == 0:
#                     sys.stderr.write('Moving volume %(label)s at z=(%(zl)d,%(zh)d) is empty.\n' % dict(label=l, zl=self.zl, zh=self.zh))
#                 else:
#                     volume_m_sliced[l] = v

#             if len(volume_f_sliced) == 0:
#                 raise Exception('All fixed volumes at z=(%(zl)d,%(zh)d) is empty.\n' % dict(zl=self.zl, zh=self.zh))

#             if len(volume_m_sliced) == 0:
#                 raise Exception('All moving volumes at z=(%(zl)d,%(zh)d) is empty.\n' % dict(zl=self.zl, zh=self.zh))

#             volume_f = volume_f_sliced
#             volume_m = volume_m_sliced

        # for i, v in volume_m.iteritems():
        #     volume_m[i] = gaussian(v, 3)

        assert volume_f is not None, 'Template volume is not specified.'
        assert volume_m is not None, 'Moving volume is not specified.'

        # self.ydim_m, self.xdim_m, self.zdim_m = volume_m.values()[0].shape
        
        # This assumes all volume_f have the same shape.
        # self.ydim_f, self.xdim_f, self.zdim_f = volume_f.values()[0].shape

        global nzvoxels_m
        if nzvoxels_m_ is None:

            # pool = Pool(16)
            # # nzvoxels_m_ = pool.map(lambda i: parallel_where_binary(volume_m[i] > 0, num_samples=int(1e6)),
            # #                         self.all_indices_m)
            # nzvoxels_m_ = pool.map(lambda i: parallel_where_binary(volume_m[i] > 0),
            #                         self.all_indices_m)
            # pool.terminate()
            # pool.join()

            # nzvoxels_m_ = [parallel_where_binary(volume_m[i] > nz_thresh) + volume_m_origin[i] for i in list(self.all_indices_m)]
            # nzvoxels_m = dict(zip(list(self.all_indices_m), nzvoxels_m_))
            nzvoxels_m = {ind_m: parallel_where_binary(volume_m[ind_m] > nz_thresh) + volume_m_origin[ind_m] 
                           for ind_m in self.all_indices_m}
        else:
            nzvoxels_m = nzvoxels_m_

        if label_weights is None:
            if not hasattr(self, 'label_weights'):
                sys.stderr.write('Label weights not set, default to 1 for all structures.\n')
                self.label_weights = {ind_m: 1 for ind_m in self.all_indices_m}
        else:
            self.label_weights = label_weights

        if reg_weights is None:
            # if not hasattr(self, 'reg_weights'):
            sys.stderr.write('Regularization weights not set, default to 0.\n')
            self.reg_weights = np.array([0,0,0])
            self.reg_weight = 0
        else:
            self.reg_weights = reg_weights

        self.inv_covar_mats_all_indices = {ind_m: np.eye(3) for ind_m in self.all_indices_m}

    def set_label_weights(self, label_weights):
        self.label_weights = label_weights

    def set_inverse_covar_mats_all_indices(self, inv_covar_mats):
        """
        Args:
            inv_covar_mats (dict {ind_m: (3,3)-ndarray}): inverse of covariance matrices.
        """
        self.inv_covar_mats_all_indices = inv_covar_mats

    def set_regularization_weights(self, reg_weights):
        """
        Args:
            reg_weights (float): If one scalar, this sets `self.reg_weight`; otherwise this sets `self.reg_weights`.
        """

        if isinstance(reg_weights, int) or isinstance(reg_weights, float):
            self.reg_weight = reg_weights
        else:
            self.reg_weights = reg_weights

    def set_bspline_grid_size(self, interval):
        """
        Set class internal variable `NuNvNw_allTestPts`.

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

            NuPx_allTestPts = compute_bspline_cp_contribution_to_test_pts(control_points=ctrl_x_intervals_centered/float(interval),
                                                                         test_points=test_pts[:,0]/float(interval))
            NvPy_allTestPts = compute_bspline_cp_contribution_to_test_pts(control_points=ctrl_y_intervals_centered/float(interval),
                                                                         test_points=test_pts[:,1]/float(interval))
            NwPz_allTestPts = compute_bspline_cp_contribution_to_test_pts(control_points=ctrl_z_intervals_centered/float(interval),
                                                                         test_points=test_pts[:,2]/float(interval))

            # NuPx_allTestPts = np.array([[N(ctrl_x/float(interval), x/float(interval))
            #                              for testPt_i, (x, y, z) in enumerate(test_pts)]
            #                             for ctrlXInterval_i, ctrl_x in enumerate(ctrl_x_intervals_centered)])
            # (n_ctrlx, n_all_nz_m)
            # NvPy_allTestPts = np.array([[N(ctrl_y/float(interval), y/float(interval))
            #                              for testPt_i, (x, y, z) in enumerate(test_pts)]
            #                             for ctrlYInterval_i, ctrl_y in enumerate(ctrl_y_intervals_centered)])
            # (n_ctrly, n_all_nz_m)
            # NwPz_allTestPts = np.array([[N(ctrl_z/float(interval), z/float(interval))
            #                              for testPt_i, (x, y, z) in enumerate(test_pts)]
            #                             for ctrlZInterval_i, ctrl_z in enumerate(ctrl_z_intervals_centered)])
            # (n_ctrlz, n_all_nz_m)
            # print 'NwPz_allTestPts', NwPz_allTestPts.shape

            sys.stderr.write("Compute NuPx/NvPy/NwPz: %.2f seconds.\n" % (time.time()-t) )

            t = time.time()

            # self.NuNvNw_allTestPts[ind_m] = np.array([np.ravel(np.tensordot(np.tensordot(NuPx_allTestPts[:,testPt_i],
            #                                                                       NvPy_allTestPts[:,testPt_i], 0),
            #                                                          NwPz_allTestPts[:,testPt_i], 0))
            #                       for testPt_i in range(len(test_pts))])

            self.NuNvNw_allTestPts[ind_m] = np.einsum('it,jt,kt->ijkt', NuPx_allTestPts, NvPy_allTestPts, NwPz_allTestPts).reshape((-1, NuPx_allTestPts.shape[-1])).T

            # print 'self.NuNvNw_allTestPts', self.NuNvNw_allTestPts[ind_m].shape
            # (n_all_nz_m, n_ctrl)

            sys.stderr.write("Compute every control point's contribution to every nonzero test point, 3 dimensions: %.2f seconds.\n" % (time.time()-t) )

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
                bboxes = np.array([volume_origin_to_bbox(volume_m[i], volume_m_origin[i]) for i in indices_m])
                # print bboxes
                xc = (bboxes[:,0].min(axis=0) + bboxes[:,1].max(axis=0)) / 2
                yc = (bboxes[:,2].min(axis=0) + bboxes[:,3].max(axis=0)) / 2
                zc = (bboxes[:,4].min(axis=0) + bboxes[:,5].max(axis=0)) / 2
                self.centroid_m = np.r_[xc, yc, zc]
            elif centroid_m == 'origin':
                self.centroid_m = np.zeros((3,))
            else:
                raise Exception('centroid_m not recognized.')
        else:
            self.centroid_m = centroid_m

        if isinstance(centroid_f, basestring):
            if centroid_f == 'centroid_m':
                self.centroid_f = self.centroid_m
            elif centroid_m == 'structure_centroid':
                self.centroid_f = np.array([np.mean(np.where(volume_f[self.labelIndexMap_m2f[i]]), axis=1)[[1,0,2]] + volume_f_origin[i] for i in indices_m]).mean(axis=0)
            elif centroid_f == 'volume_centroid':
                bboxes = np.array([volume_origin_to_bbox(volume_f[self.labelIndexMap_m2f[i]], volume_f_origin[self.labelIndexMap_m2f[i]]) for i in indices_m])
                xc = (bboxes[:,0].min(axis=0) + bboxes[:,1].max(axis=0)) / 2
                yc = (bboxes[:,2].min(axis=0) + bboxes[:,3].max(axis=0)) / 2
                zc = (bboxes[:,4].min(axis=0) + bboxes[:,5].max(axis=0)) / 2
                self.centroid_f = np.r_[xc, yc, zc]
            elif centroid_f == 'origin':
                self.centroid_f = np.zeros((3,))
            else:
                raise Exception('centroid_f not recognized.')
        else:
            self.centroid_f = centroid_f

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

    def load_gradient(self, gradient_filepath_map_f=None, indices_f=None, gradients=None, rescale=None):
        """Load gradients of fixed volumes.

        Need to pass gradient_filepath_map_f in from outside because Aligner class should be agnostic about structure names.

        Args:
            gradient_filepath_map_f (dict of str): path string that contains formatting parts and (suffix).
            gradients (dict of ((3,dimx,dimy,dimz) arrays, origin):
        """

        if indices_f is None:
            indices_f = set([self.labelIndexMap_m2f[ind_m] for ind_m in self.all_indices_m])
            sys.stderr.write('indices_f: %s\n' % indices_f)

        global grad_f, grad_f_origin

        if gradients is not None:
            grad_f = {i: g for i, (g, o) in gradients.iteritems()}
            grad_f_origin = {i: o.astype(np.int) for i, (g, o) in gradients.iteritems()}
        else:
            raise
#         else:
#             assert gradient_filepath_map_f is not None, 'gradient_filepath_map_f not specified.'
#             grad_f = {ind_f: np.zeros((3, self.ydim_f, self.xdim_f, self.zdim_f), dtype=np.float16) for ind_f in indices_f}

#             t1 = time.time()

#             for ind_f in indices_f:

#                 t = time.time()

#                 download_from_s3(gradient_filepath_map_f[ind_f] % {'suffix': 'gx'}, is_dir=False)
#                 download_from_s3(gradient_filepath_map_f[ind_f] % {'suffix': 'gy'}, is_dir=False)
#                 download_from_s3(gradient_filepath_map_f[ind_f] % {'suffix': 'gz'}, is_dir=False)

#                 if hasattr(self, 'zl'):
#                     if rescale is None:
#                         grad_f[ind_f][0] = bp.unpack_ndarray_file(gradient_filepath_map_f[ind_f] % {'suffix': 'gx'})[..., self.zl:self.zh+1]
#                         grad_f[ind_f][1] = bp.unpack_ndarray_file(gradient_filepath_map_f[ind_f] % {'suffix': 'gy'})[..., self.zl:self.zh+1]
#                         grad_f[ind_f][2] = bp.unpack_ndarray_file(gradient_filepath_map_f[ind_f] % {'suffix': 'gz'})[..., self.zl:self.zh+1]
#                     else:
#                         grad_f[ind_f][0] = rescale_volume_by_resampling(bp.unpack_ndarray_file(gradient_filepath_map_f[ind_f] % {'suffix': 'gx'})[..., self.zl:self.zh+1], rescale)
#                         grad_f[ind_f][0] = rescale_volume_by_resampling(bp.unpack_ndarray_file(gradient_filepath_map_f[ind_f] % {'suffix': 'gy'})[..., self.zl:self.zh+1], rescale)
#                         grad_f[ind_f][0] = rescale_volume_by_resampling(bp.unpack_ndarray_file(gradient_filepath_map_f[ind_f] % {'suffix': 'gz'})[..., self.zl:self.zh+1], rescale)
#                 else:
#                     if rescale is None:
#                         grad_f[ind_f][0] = bp.unpack_ndarray_file(gradient_filepath_map_f[ind_f] % {'suffix': 'gx'})
#                         grad_f[ind_f][1] = bp.unpack_ndarray_file(gradient_filepath_map_f[ind_f] % {'suffix': 'gy'})
#                         grad_f[ind_f][2] = bp.unpack_ndarray_file(gradient_filepath_map_f[ind_f] % {'suffix': 'gz'})
#                     else:
#                         grad_f[ind_f][0] = rescale_volume_by_resampling(bp.unpack_ndarray_file(gradient_filepath_map_f[ind_f] % {'suffix': 'gx'}), rescale)
#                         grad_f[ind_f][1] = rescale_volume_by_resampling(bp.unpack_ndarray_file(gradient_filepath_map_f[ind_f] % {'suffix': 'gy'}), rescale)
#                         grad_f[ind_f][2] = rescale_volume_by_resampling(bp.unpack_ndarray_file(gradient_filepath_map_f[ind_f] % {'suffix': 'gz'}), rescale)

#                 sys.stderr.write('load gradient %s: %f seconds\n' % (ind_f, time.time() - t)) # ~6s

#             sys.stderr.write('overall: %f seconds\n' % (time.time() - t1)) # ~100s


    def get_valid_voxels_after_transform(self, T, tf_type, ind_m, return_valid, n_sample=None):
        """
        Args:
            T (ndarray): transform parameter vector
            tf_type (str): rigid, affine or bspline.
            return_valid (bool): whether to return a boolean list indicating which nonzero moving voxels are valid.
        """

        ind_f = self.labelIndexMap_m2f[ind_m]
        
        # t = time.time()
        if n_sample is not None:

            num_nz = len(nzvoxels_centered_m[ind_m])
            assert num_nz > 0, "No valid pixel for ind_m = %d even before transform." % ind_m
            valid_moving_voxel_indicator = np.zeros((num_nz,), np.bool)

            # t = time.time()
            import random
            random_indices = np.array(sorted(random.sample(xrange(num_nz), min(num_nz, n_sample))))
            # NOTE: sorted is important
            # sys.stderr.write('random_indices: %.2f seconds\n' % (time.time() - t))

            if tf_type == 'affine' or tf_type == 'rigid':
                pts_prime_sampled = transform_points_affine(np.array(T),
                                            pts_centered=nzvoxels_centered_m[ind_m][random_indices],
                                            c_prime=self.centroid_f).astype(np.int16)
            elif tf_type == 'bspline':
                n_params = len(T)
                buvwx = T[:n_params/3]
                buvwy = T[n_params/3:n_params/3*2]
                buvwz = T[n_params/3*2:]
                pts_prime_sampled = transform_points_bspline(buvwx, buvwy, buvwz, \
                                                             pts_centered=nzvoxels_centered_m[ind_m][random_indices],
                                                             c_prime=self.centroid_f,
                            NuNvNw_allTestPts=self.NuNvNw_allTestPts[ind_m][random_indices]).astype(np.int16)
            else:
                raise

            xs_prime_sampled, ys_prime_sampled, zs_prime_sampled = pts_prime_sampled.T
            
            # print 'origin=', volume_f_origin[ind_f], 'volume_shape=', [volume_f[ind_f].shape[1], volume_f[ind_f].shape[0], volume_f[ind_f].shape[2]]
            
            valid_indicator_within_sampled = (xs_prime_sampled - volume_f_origin[ind_f][0] >= 0) & \
            (ys_prime_sampled - volume_f_origin[ind_f][1] >= 0) & \
            (zs_prime_sampled - volume_f_origin[ind_f][2] >= 0) & \
            (xs_prime_sampled - volume_f_origin[ind_f][0] < volume_f[ind_f].shape[1]) & \
            (ys_prime_sampled - volume_f_origin[ind_f][1] < volume_f[ind_f].shape[0]) & \
            (zs_prime_sampled - volume_f_origin[ind_f][2] < volume_f[ind_f].shape[2])
              
            valid_moving_voxel_indicator[random_indices[valid_indicator_within_sampled]] = 1
                
            xs_prime_valid = xs_prime_sampled[valid_indicator_within_sampled]
            ys_prime_valid = ys_prime_sampled[valid_indicator_within_sampled]
            zs_prime_valid = zs_prime_sampled[valid_indicator_within_sampled]
            
        else:
            if tf_type == 'affine' or tf_type == 'rigid':
                pts_prime = transform_points_affine(np.array(T),
                                            pts_centered=nzvoxels_centered_m[ind_m],
                                            c_prime=self.centroid_f).astype(np.int16)
            elif tf_type == 'bspline':
                n_params = len(T)
                buvwx = T[:n_params/3]
                buvwy = T[n_params/3:n_params/3*2]
                buvwz = T[n_params/3*2:]
                pts_prime = transform_points_bspline(buvwx, buvwy, buvwz,
                                                     pts_centered=nzvoxels_centered_m[ind_m], c_prime=self.centroid_f,
                                                    NuNvNw_allTestPts=self.NuNvNw_allTestPts[ind_m]).astype(np.int16)

            xs_prime, ys_prime, zs_prime = pts_prime.T
            
            valid_moving_voxel_indicator = (xs_prime - volume_f_origin[ind_f][0] >= 0) & \
            (ys_prime - volume_f_origin[ind_f][1] >= 0) & \
            (zs_prime - volume_f_origin[ind_f][2] >= 0) & \
            (xs_prime - volume_f_origin[ind_f][0] < volume_f[ind_f].shape[1]) & \
            (ys_prime - volume_f_origin[ind_f][1] < volume_f[ind_f].shape[0]) & \
            (zs_prime - volume_f_origin[ind_f][2] < volume_f[ind_f].shape[2])
            
#             (xs_prime >= 0) & (ys_prime >= 0) & (zs_prime >= 0) & \
#                     (xs_prime < self.xdim_f) & (ys_prime < self.ydim_f) & (zs_prime < self.zdim_f)

            xs_prime_valid = xs_prime[valid_moving_voxel_indicator]
            ys_prime_valid = ys_prime[valid_moving_voxel_indicator]
            zs_prime_valid = zs_prime[valid_moving_voxel_indicator]

        # sys.stderr.write("transform all points: %.2f s\n" % (time.time() - t))

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
            num_samples (int): if given, score is computed based on only the set of sampled voxels; if not given, use all non-zero voxels.
        """

        # t = time.time()

        score, xs_prime_valid, ys_prime_valid, zs_prime_valid, valid_moving_voxel_indicators = self.compute_score_one(T, tf_type=tf_type, ind_m=ind_m, return_valid=True, n_sample=num_samples)

        # sys.stderr.write('Valid voxels: %d\n' % np.count_nonzero(valid_moving_voxel_indicators))

        # sys.stderr.write("compute_score_one: %.2f s\n" % (time.time() - t))

        # Moving volume's valid voxel coordinates (not centralized).
        # t = time.time()
        xyzs_valid = nzvoxels_m[ind_m][valid_moving_voxel_indicators]
        # sys.stderr.write("fancy indexing into moving volume nz voxels: %.2f s\n" % (time.time() - t))
        # Moving volume's value at valid voxels. (n_valid_nz_m, )
        # t = time.time()
        S_m_valid_scores = volume_m[ind_m][xyzs_valid[:,1] - volume_m_origin[ind_m][1], 
                                           xyzs_valid[:,0] - volume_m_origin[ind_m][0], 
                                           xyzs_valid[:,2] - volume_m_origin[ind_m][2]]
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
        Sx = grad_f[ind_f][0, ys_prime_valid - grad_f_origin[ind_f][1], 
                           xs_prime_valid - grad_f_origin[ind_f][0], 
                           zs_prime_valid - grad_f_origin[ind_f][2]]
        Sy = grad_f[ind_f][1, ys_prime_valid - grad_f_origin[ind_f][1], 
                           xs_prime_valid - grad_f_origin[ind_f][0], 
                           zs_prime_valid - grad_f_origin[ind_f][2]]
        Sz = grad_f[ind_f][2, ys_prime_valid - grad_f_origin[ind_f][1], 
                           xs_prime_valid - grad_f_origin[ind_f][0], 
                           zs_prime_valid - grad_f_origin[ind_f][2]]
        if np.all(Sx == 0) and np.all(Sy == 0) and np.all(Sz == 0):
            raise Exception("Image gradient at all valid voxel is zero.")
        # sys.stderr.write("fancy indexing into fixed volume gradient: %.2f s\n" % (time.time() - t))

        xs_prime_valid = xs_prime_valid.astype(np.float)
        ys_prime_valid = ys_prime_valid.astype(np.float)
        zs_prime_valid = zs_prime_valid.astype(np.float)

#         t = time.time()

#         # Sample within valid voxels.
#         # Note that sampling takes time. Maybe it is better not sampling.
#         if num_samples is not None:

#             # t = time.time()
#             n_valid = np.count_nonzero(valid_moving_voxel_indicators)
#             # Typical n ranges from 63984 to 451341
#             # sys.stderr.write("count_nonzero: %.2f s\n" % (time.time() - t))
#             n_sample = min(num_samples, n_valid)
#             sys.stderr.write('%d: use %d samples out of %d valid\n' % (ind_m, n_sample, n_valid))
#             import random
#             ii = sorted(random.sample(range(n_valid), n_sample))

#             S_m_valid_scores = S_m_valid_scores[ii]
#             dxs = dxs[ii]
#             dys = dys[ii]
#             dzs = dzs[ii]
#             Sx = Sx[ii]
#             Sy = Sy[ii]
#             Sz = Sz[ii]
#             xs_prime_valid = xs_prime_valid[ii]
#             ys_prime_valid = ys_prime_valid[ii]
#             zs_prime_valid = zs_prime_valid[ii]

#             if tf_type == 'bspline':
#                 NuNvNw_allTestPts = NuNvNw_allTestPts[ii]
#         # else:
#         #     n_sample = n_valid

#         sys.stderr.write("sample: %.2f s\n" % (time.time() - t))

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

            # if tf_type == 'rigid':
            #     grad[0] = grad[0] - 2*self.reg_weights[0] * tx
            #     # print grad[0], 2*self.reg_weights[0] * tx
            #     grad[1] = grad[1] - 2*self.reg_weights[1] * ty
            #     grad[2] = grad[2] - 2*self.reg_weights[2] * tz
            # elif tf_type == 'affine':
            #     grad[3] = grad[3] - 2*self.reg_weights[0] * tx
            #     # print grad[3], 2*self.reg_weights[0] * tx
            #     grad[7] = grad[7] - 2*self.reg_weights[1] * ty
            #     grad[11] = grad[11] - 2*self.reg_weights[2] * tz

            # ref: https://math.stackexchange.com/questions/222894/how-to-take-the-gradient-of-the-quadratic-form
            if tf_type == 'rigid':
                # print grad[:3], - self.reg_weight * np.dot((self.inv_covar_mats_all_indices[ind_m] + self.inv_covar_mats_all_indices[ind_m].T), [tx,ty,tz])
                grad[:3] = grad[:3] - self.reg_weight * np.dot((self.inv_covar_mats_all_indices[ind_m] + self.inv_covar_mats_all_indices[ind_m].T), [tx,ty,tz])
            elif tf_type == 'affine':
                grad[[3,7,11]] = grad[[3,7,11]] - self.reg_weight * np.dot((self.inv_covar_mats_all_indices[ind_m] + self.inv_covar_mats_all_indices[ind_m].T), [tx,ty,tz])
        elif tf_type == 'bspline':
            pass

        # sys.stderr.write("3: %.2f s\n" % (time.time() - t))

        if tf_type == 'rigid' or tf_type == 'affine':
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
                grad += self.label_weights[ind_m] * grad_one
                score += self.label_weights[ind_m] * score_one

            except Exception as e:
                # raise e
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


    def compute_score_one(self, T, tf_type, ind_m, return_valid=False, n_sample=None):
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
        xs_prime_valid, ys_prime_valid, zs_prime_valid, valid_moving_voxel_indicator = \
        self.get_valid_voxels_after_transform(T, tf_type=tf_type, ind_m=ind_m, return_valid=True, n_sample=n_sample)
        # sys.stderr.write("Timing 2: get_valid_voxels_after_transform: %.2f seconds.\n" % (time.time()-t))

        if n_sample is None:
            n_total = len(nzvoxels_centered_m[ind_m])
        else:
            n_total = min(n_sample, len(nzvoxels_centered_m[ind_m]))
        n_valid = np.count_nonzero(valid_moving_voxel_indicator)
        n_invalid = n_total - n_valid
        # n_invalid = 0
        # print valid_moving_voxel_indicator
        if n_invalid > 0:
            sys.stderr.write('%d: %d valid, %d out-of-bound voxels after transform.\n' % (ind_m, n_valid, n_invalid))
        if n_valid == 0:
            raise Exception('%d: No valid voxels after transform.' % ind_m)

        ind_f = self.labelIndexMap_m2f[ind_m]

        # Reducing the scale of voxel value is important for keeping the sum (i.e. the following line) in the represeantable range of the chosen data type.
        # t = time.time()

        # This one does not consider the values of atlas voxels.
        # voxel_probs_valid = volume_f[ind_f][ys_prime_valid, xs_prime_valid, zs_prime_valid] / 1e6

        # This one considers the values of atlas voxels.
        xs_valid, ys_valid, zs_valid = nzvoxels_m[ind_m].astype(np.int16)[valid_moving_voxel_indicator].T
        voxel_probs_valid = volume_m[ind_m][ys_valid - volume_m_origin[ind_m][1], 
                                            xs_valid - volume_m_origin[ind_m][0], 
                                            zs_valid - volume_m_origin[ind_m][2]] * \
        volume_f[ind_f][ys_prime_valid - volume_f_origin[ind_f][1], 
                        xs_prime_valid - volume_f_origin[ind_f][0], 
                        zs_prime_valid - volume_f_origin[ind_f][2]] / 1e6

        # sys.stderr.write("Timing 2: fancy indexing valid voxels into fixed volume: %.2f seconds.\n" % (time.time()-t))

        # Penalize out-of-bound voxels, minus 1 for each such voxel
        s = voxel_probs_valid.sum() - np.sign(self.label_weights[ind_m]) * n_invalid / 1e6

        # Regularize
        if tf_type == 'affine' or tf_type == 'rigid':
            tx = T[3]
            ty = T[7]
            tz = T[11]
            # s_reg = self.reg_weights[0]*tx**2 + self.reg_weights[1]*ty**2 + self.reg_weights[2]*tz**2
            s_reg = self.reg_weight * np.dot([tx,ty,tz], np.dot(self.inv_covar_mats_all_indices[ind_m], [tx,ty,tz]))
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

    def compute_scores_neighborhood_grid(self, params, dxs, dys, dzs, dtheta_xys=None, indices_m=None, parallel=True):
        """
        Args:
            params ((12,)-array): the parameter vector around which the neighborhood is taken.
        """

        from itertools import product

        if parallel:
            #parallel
            pool = Pool(processes=12)
            # if dtheta_xys is None:
            scores = pool.map(lambda (dx, dy, dz): self.compute_score(params + np.array([0.,0.,0., dx, 0.,0.,0., dy, 0.,0.,0., dz]), indices_m=indices_m),
                            product(dxs, dys, dzs))
            # else:
            #     scores = pool.map(lambda (tx, ty, tz, theta_xy): self.compute_score(affine_components_to_vector(tx,ty,tz,theta_xy), indices_m=indices_m), product(dxs, dys, dzs, dtheta_xys))
            pool.close()
            pool.join()
        else:
            raise
            # scores = np.reshape([self.compute_score(params + (0.,0.,0., dx, 0.,0.,0., dy, 0.,0.,0., dz), indices_m=indices_m)
            #                 for dx, dy, dz in product(dxs, dys, dzs)],
            #                 (dxs.size, dys.size, dzs.size))

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
                    eta=3., stop_radius_voxel=10, init_T=None):
        """Grid search.

         Args:
             grid_search_iteration_number (int): number of iteration
             eta: sample number and sigma = initial value * np.exp(-iter/eta), default = 3.
         Returns:
             params_best_upToNow ((12,) float array): found parameters
        """

        # dx_best, dy_best, dz_best, dthetaxy_best = (0, 0, 0, 0)
        score_best_upToNow = -np.inf

        # dx_best, dy_best, dz_best, dthetaxy_best = ds_best_upToNow
        # d_best_mat = np.vstack([affine_components_to_vector(dx_best, dy_best, dz_best, dthetaxy_best).reshape((3,4)), (0,0,0,1)])

        if init_T is None:
            init_T = np.array([1,0,0,0,0,1,0,0,0,0,1,0])

        T_best_upToNow = init_T

        if indices_m is None:
            indices_m = self.all_indices_m

        for iteration in range(grid_search_iteration_number):

            # self.logger.info('grid search iteration %d', iteration)

            # init_tx, init_ty, init_tz, init_theta_xy = affine_components_to_vector(dx_best, dy_best, dz_best, dthetaxy_best)

            # n = int(init_n*np.exp(-iteration/eta))
            n = init_n

            sigma_tx = std_tx*np.exp(-iteration/eta)
            sigma_ty = std_ty*np.exp(-iteration/eta)
            sigma_tz = std_tz*np.exp(-iteration/eta)
            sigma_theta_xy = std_theta_xy*np.exp(-iteration/eta)

            sys.stderr.write('sigma_tx: %.2f (voxel), sigma_ty: %.2f, sigma_tz: %.2f, sigma_theta_xy: %.2f (deg), n:%d\n' % \
            (sigma_tx, sigma_ty, sigma_tz, np.rad2deg(sigma_theta_xy), n))

            dx_grid = sigma_tx * np.linspace(-1,1,n)
            dy_grid = sigma_ty * np.linspace(-1,1,n)
            dz_grid = sigma_tz * np.linspace(-1,1,n)
            # theta_xy_grid = init_theta_xy + sigma_theta_xy * np.linspace(-1,1,n)
            dthetaxy_grid = [0]

            # samples = np.c_[tx_grid, ty_grid, tz_grid, theta_xy_grid]

            #############

            t = time.time()

            scores = self.compute_scores_neighborhood_grid(T_best_upToNow,
                                                   dxs=dx_grid, dys=dy_grid, dzs=dz_grid, dtheta_xys=dthetaxy_grid,
                                                   indices_m=indices_m, parallel=parallel)
            i_best = np.argmax(scores)
            score_best = scores[i_best]
            i_tx, i_ty, i_tz, i_thetaxy = np.unravel_index(i_best, (len(dx_grid), len(dy_grid), len(dz_grid), len(dthetaxy_grid)))
            dx_best = dx_grid[i_tx]
            dy_best = dy_grid[i_ty]
            dz_best = dz_grid[i_tz]
            dthetaxy_best = dthetaxy_grid[i_thetaxy]

            sys.stderr.write('grid search: %f seconds\n' % (time.time() - t)) # ~23s

            if score_best > score_best_upToNow:
                # self.logger.info('%f %f', score_best_upToNow, score_best)
                sys.stderr.write('New best: %f %f\n' % (score_best_upToNow, score_best))

                score_best_upToNow = score_best
                # ds_best_upToNow = dx_best, dy_best, dz_best, dthetaxy_best
                d_best_mat = np.vstack([affine_components_to_vector(dx_best, dy_best, dz_best, dthetaxy_best).reshape((3,4)), (0,0,0,1)])
                T_best_upToNow_mat = np.vstack([np.reshape(T_best_upToNow, (3,4)), [0,0,0,1]])
                T_best_upToNow = np.dot(d_best_mat, T_best_upToNow_mat)[:3].flatten()

                # sys.stderr.write('dx_best: %.2f (voxel), dy_best: %.2f, dz_best: %.2f, dthetaxy_best: %.2f (deg), score=%f\n' % (dx_best, dy_best, dz_best, np.rad2deg(dthetaxy_best), score_best))

                sys.stderr.write('T_best_upToNow: %s\n' % str(np.reshape(T_best_upToNow, (3,4))))

            if sigma_tx < stop_radius_voxel and sigma_ty < stop_radius_voxel and sigma_tz < stop_radius_voxel:
                # if sigma is reduced to smaller than 10 voxels, abort
                break

            sys.stderr.write('\n')

                # self.logger.info('%f %f %f', tx_best, ty_best, tz_best)
        # sys.stderr.write('deviations_best_upToNow: %f %f %f %f\n' % (dx_best, dy_best, dz_best, dthetaxy_best))

        if return_best_score:
            return T_best_upToNow, score_best_upToNow
        else:
            return T_best_upToNow

    def do_grid_search(self, grid_search_iteration_number=10, grid_search_sample_number=1000,
                      std_tx=100, std_ty=100, std_tz=30, std_theta_xy=np.deg2rad(30),
                       grid_search_eta=3., stop_radius_voxel=10,
                      indices_m=None, parallel=True, init_T=None):

        if indices_m is None:
            indices_m = self.all_indices_m

        T, grid_search_score = self.grid_search(grid_search_iteration_number, indices_m=indices_m,
                                                    init_n=grid_search_sample_number,
                                                    std_tx=std_tx, std_ty=std_ty, std_tz=std_tz, std_theta_xy=std_theta_xy,
                                                                                         eta=grid_search_eta, stop_radius_voxel=stop_radius_voxel,
                                                    return_best_score=True,
                                                                                        parallel=parallel, init_T=init_T)
        # T = np.r_[1,0,0, tx_best, 0,1,0, ty_best, 0,0,1, tz_best]
        # T = affine_components_to_vector(tx_best, ty_best, tz_best, theta_xy_best)
        # d_mat = np.vstack([affine_components_to_vector(dx_best, dy_best, dz_best, dthetaxy_best).reshape((3,4)), (0,0,0,1)])
        # init_T_mat = np.vstack([init_T.reshape((3,4)), (0,0,0,1)])
        # T = np.dot(d_mat, init_T_mat)[:3].flatten()
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
                affine_scaling_limits=None,
                bspline_deformation_limit=None):
        """Optimize.
        # Objective = texture score - reg_weights[0] * tx**2 - reg_weights[1] * ty**2 - reg_weights[2] * tz**2
        Objective = texture score - reg_weights * [tx, ty, tz]^T * CovMat^{-1} * [tx, ty, tz]

        Args:
            reg_weights: penalty for translation vector (tx,ty,tz). If just one scalar, this sets `self.reg_weight`; otherwise this sets `self.reg_weights`.
            affine_scaling_limits (2 tuple of float): min/max for the diagonal elements of affine matrix.
            bspline_deformation_limit (float): maximum deformation of any bspline control point in any direction.

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
        best_gradient_descent_params = T

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
                                                                           num_samples=grad_computation_sample_number,
                                                                               bspline_deformation_limit=bspline_deformation_limit)

            else:
                raise Exception('Type must be either rigid or affine.')

            sys.stderr.write('step: %.2f seconds\n' % (time.time() - t))
            sys.stderr.write('current score: %f\n' % s)

            if tf_type == 'rigid' or tf_type == 'affine':
                # sys.stderr.write('new_T: %s\n' % new_T[[3,7,11]])
                sys.stderr.write('new_T: %s\n' % new_T)
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
            elif tf_type == 'bspline':
                if iteration > history_len:
                    if np.all([np.std(Ts[iteration-history_len:iteration, :], axis=0) < terminate_thresh_trans]):
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
        One optimization step over the manifold SE(3).

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
        # Here grad is dObjective/d\epsilon. epsilon is the small adjustment in the linearization of manifold at current estimate.

        # print 'score:', score
        # print 'grad:', grad

        # # AdaGrad Rule
        grad_historical += grad**2
        grad_adjusted = grad / np.sqrt(grad_historical + epsilon)
        # Note: It is wrong to do: grad_adjusted = grad /  (np.sqrt(grad_historical) + epsilon)
        # sys.stderr.write('Norm of gradient = %f\n' % np.linalg.norm(grad_adjusted))
        sys.stderr.write('Norm of gradient (translation) = %f\n' % np.linalg.norm(grad_adjusted[:3]))
        sys.stderr.write('Norm of gradient (rotation) = %f\n' % np.linalg.norm(grad_adjusted[3:]))

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

        ########### New ############

#         epsilon_opt = lr * grad_adjusted # no minus sign because we are maximizing objective instead of minimizing.
#         # epsilon_opt is the optimal small adjustment in the linearization of manifold from the current estimate.

#         exp_w_skew, Vt = matrix_exp_v(epsilon_opt) # 3x3, 3x1
#         exp_epsilon_opt = np.vstack([np.c_[exp_w_skew, Vt], [0,0,0,1]]) # 4x4
#         T4x4 = np.vstack([np.reshape(T, (3,4)), [0,0,0,1]]) # 4x4
#         newT4x4 = np.dot(exp_epsilon_opt, T4x4)
#         R_new = newT4x4[:3, :3]
#         t_new = newT4x4[:3, 3]

        ############ Old ############
        v_opt = lr * grad_adjusted # no minus sign because we are maximizing objective instead of minimizing.

        theta = np.sqrt(np.sum(v_opt[3:]**2))
        assert theta < np.pi

        exp_w, Vt = matrix_exp_v(v_opt)
        # print 'Vt', Vt
        Tm = np.reshape(T, (3,4))
        t = Tm[:, 3]
        # print 't', t
        R = Tm[:, :3]
        R_new = np.dot(exp_w, R)
        t_new = np.dot(exp_w, t) + Vt
        # t_new = t + Vt
        # print 't_new', t_new
        ###########################

        euler_angles_R_new = rotationMatrixToEulerAngles(R_new) # (around x, around y, around z)
        sys.stderr.write("around x=%.2f; around y=%.2f; around z=%.2f\n" % \
        (np.rad2deg(euler_angles_R_new[0]), np.rad2deg(euler_angles_R_new[1]), np.rad2deg(euler_angles_R_new[2])))

        if euler_angles_R_new[2] > np.pi/4:
            R_new = eulerAnglesToRotationMatrix([euler_angles_R_new[0],euler_angles_R_new[1], np.pi/4.])
            sys.stderr.write("Constrain around-z angle. Force to < 45 degree.\n")
        elif euler_angles_R_new[2] < -np.pi/4:
            R_new = eulerAnglesToRotationMatrix([euler_angles_R_new[0],euler_angles_R_new[1], -np.pi/4.])
            sys.stderr.write("Constrain around-z angle. Force to < 45 degree\n")

        if euler_angles_R_new[1] > np.pi/4:
            R_new = eulerAnglesToRotationMatrix([euler_angles_R_new[0], np.pi/4., euler_angles_R_new[2]])
            sys.stderr.write("Constrain around-y angle. Force to < 45 degree\n")
        elif euler_angles_R_new[1] < -np.pi/4:
            R_new = eulerAnglesToRotationMatrix([euler_angles_R_new[0], -np.pi/4., euler_angles_R_new[2]])
            sys.stderr.write("Constrain around-y angle. Force to < 45 degree\n")

        if euler_angles_R_new[0] > np.pi/4:
            R_new = eulerAnglesToRotationMatrix([np.pi/4., euler_angles_R_new[1],euler_angles_R_new[2]])
            sys.stderr.write("Constrain around-x angle. Force to < 45 degree\n")
        elif euler_angles_R_new[0] < -np.pi/4:
            R_new = eulerAnglesToRotationMatrix([-np.pi/4., euler_angles_R_new[1],euler_angles_R_new[2]])
            sys.stderr.write("Constrain around-x angle. Force to < 45 degree\n")

        # Constrain the amount of rotation around ANY axis.

        return np.column_stack([R_new, t_new]).flatten(), score, grad_historical, sq_updates_historical


    def step_gd(self, T, lr, grad_historical, sq_updates_historical, tf_type, surround=False, surround_weight=2., num_samples=None, indices_m=None, scaling_limits=None, bspline_deformation_limit=100):
        """
        One optimization step using gradient descent with Adagrad.

        Args:
            T ((12,) vector): flattened vector of 3x4 transform matrix.
            lr ((12,) vector): learning rate
            dMdA_historical ((12,) vector): accumulated gradiant magnitude, for Adagrad

            scaling_limits (2-tuple of float): applicable only to affine registration; the minimum/maximum values for any of the three diagonal elements of the affine matrix.
            bspline_deformation_limit (float): applicable only to bspline registration; the maximum deformation in any of x,y,z directions allowed for any control point.

        Returns:
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
        sys.stderr.write('Norm of gradient = %f\n' % np.linalg.norm(grad_adjusted))
        new_T = T + lr*grad_adjusted

        # Constrain the transform
        if tf_type == 'bspline':
            # Limit the deformation at all control points to be less than bspline_deformation_limit.
            new_T = np.sign(new_T) * np.minimum(np.abs(new_T), bspline_deformation_limit)
        elif tf_type == 'affine':
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

        if tf_type == 'affine' or tf_type == 'rigid':
            sys.stderr.write("in T: %.2f %.2f %.2f, out T: %.2f %.2f %.2f\n" % (T[3], T[7], T[11], new_T[3], new_T[7], new_T[11]))
        elif tf_type == 'bspline':
            sys.stderr.write("in T: min %.2f, max %.2f; out T: min %.2f, max %.2f\n" % (T.min(), T.max(), new_T.min(), new_T.max()))
        return new_T, score, grad_historical, sq_updates_historical
    
    
    
    
def global_registration(global_alignment_spec, structures_m):
    global_aligner_parameters = generate_aligner_parameters_v2(alignment_spec=global_alignment_spec, structures_m=structures_m)

    volume_fixed = global_aligner_parameters['volume_fixed']
    volume_moving = global_aligner_parameters['volume_moving']

    aligner = Aligner(volume_fixed, volume_moving, labelIndexMap_m2f=global_aligner_parameters['label_mapping_m2f'])

    aligner.set_centroid(centroid_m='volume_centroid', centroid_f='centroid_m')
    aligner.set_label_weights(label_weights=global_aligner_parameters['label_weights_m'])

    # If no any structures overlap after centroid alignment, we need to do grid search to obtain a better initial transform.
    # grid_search_T, grid_search_score = aligner.do_grid_search(grid_search_iteration_number=1,
    #                                    grid_search_sample_number=5,
    #                 parallel=True,
    #                 std_tx=50, std_ty=50, std_tz=30)

    gradients_f = compute_gradient_v2(volume_fixed, smooth_first=True)
    aligner.load_gradient(gradients=gradients_f)

    # Tuning learning rate:
    # lr1=10., if lucky converges much faster than lr1=1., but sometimes stuck in local maxima
    # If lr1=1., grad_computation_sample_number=1e5 is sufficient.

    trial_num = 1

    T_all_trials = []
    scores_all_trials = []
    traj_all_trials = []

    for _ in range(trial_num):

        try:
            T, scores = aligner.optimize(tf_type=global_aligner_parameters['transform_type'],
                                         max_iter_num=1000,
                                         history_len=20,
                                         terminate_thresh_rot=.001,
                                         terminate_thresh_trans=.1,
                                         grad_computation_sample_number=global_aligner_parameters['grad_computation_sample_number'],
                                         lr1=10, lr2=.1,
                                        # init_T=grid_search_T,
                                         affine_scaling_limits=(.9, 1.2)
                                        )
            T_all_trials.append(T)
            scores_all_trials.append(scores)
            traj_all_trials.append(aligner.Ts)

        except Exception as e:
            sys.stderr.write('%s\n' % e)

    Ts = np.array(aligner.Ts)

    plt.plot(Ts[:, [0,1,2,4,5,6,8,9,10]]);
    plt.title('rotational params');
    plt.xlabel('Iteration');
    plt.show();

    plt.plot(Ts[:, [3,7,11]]);
    plt.title('translation params');
    plt.xlabel('Iteration');
    plt.show();

    best_trial = np.argsort([np.max(scores) for scores in scores_all_trials])[-1]
    # T = T_all_trials[best_trial]
    T = traj_all_trials[best_trial][-1]
    scores = scores_all_trials[best_trial]
    print 'Best trial:', best_trial
    print max(scores), scores[-1]

    print T.reshape((3,4))
    plt.figure();
    plt.plot(scores);
    plt.show();


    transform_parameters = {
    'parameters': T_all_trials[best_trial],
    'centroid_m_wrt_wholebrain': aligner.centroid_m,
    'centroid_f_wrt_wholebrain': aligner.centroid_f,
    'resolution_um': global_aligner_parameters['resolution_um'] 
    }

    DataManager.save_alignment_results_v2(transform_parameters=transform_parameters,
                       score_traj=scores_all_trials[best_trial],
                       parameter_traj=traj_all_trials[best_trial],
                      alignment_spec=global_alignment_spec)

    transform_parameters = DataManager.load_alignment_parameters_v3(alignment_spec=global_alignment_spec)

    ####################################
    
    stack_m_spec = global_alignment_spec['stack_m']
    
    scale_registration_resol_to_original_resol = transform_parameters['resolution_um'] / convert_resolution_string_to_voxel_size(resolution=stack_m_spec['resolution'], stack=stack_m_spec['name'])
    # scale to original resolution of the moving volume
    print 'scale_registration_resol_to_original_resol =', scale_registration_resol_to_original_resol

    parameters_rescaled_to_originalResol = transform_parameters['parameters'].copy()
    parameters_rescaled_to_originalResol[[3,7,11]] = np.array(transform_parameters['parameters'])[[3,7,11]] * scale_registration_resol_to_original_resol

    transform_parameters_originalResol = {'centroid_f_wrt_wholebrain': transform_parameters['centroid_f_wrt_wholebrain'] * scale_registration_resol_to_original_resol, 
                                 'centroid_m_wrt_wholebrain': transform_parameters['centroid_m_wrt_wholebrain'] * scale_registration_resol_to_original_resol,
                                 'parameters': parameters_rescaled_to_originalResol.flatten()
                                }    

    for name_s in all_known_structures_sided_with_surround:

        volume, bbox_wrt_atlasSpace = \
        DataManager.load_original_volume_v2(stack_m_spec, structure=name_s, bbox_wrt='atlasSpace')

        transformed_vol, transformed_vol_box_wrt_fixedWholebrain = \
        transform_volume_by_alignment_parameters_v2(volume, bbox=bbox_wrt_atlasSpace, 
                                                 transform_parameters=transform_parameters_originalResol)
    #     print transformed_vol.shape, transformed_vol_box_wrt_fixedWholebrain

        DataManager.save_transformed_volume(volume=transformed_vol, 
                                            bbox=transformed_vol_box_wrt_fixedWholebrain, 
                                            alignment_spec=global_alignment_spec, 
                                            resolution=stack_m_spec['resolution'], 
                                            structure=name_s)
        del volume, transformed_vol


def local_registration(local_alignment_spec, global_alignment_spec):

    structure_m = local_alignment_spec['stack_m']['structure']

    local_aligner_parameters = generate_aligner_parameters_v2(alignment_spec=local_alignment_spec,
                                                           structures_m=[structure_m])

    global_transform_parameters = DataManager.load_alignment_parameters_v3(global_alignment_spec)

    volume_fixed = local_aligner_parameters['volume_fixed']
    volume_moving = local_aligner_parameters['volume_moving']

    v, o = volume_moving[local_aligner_parameters['structure_to_label_moving'][structure_m]]
    init_shift_wrt_movingWholebrain = get_centroid_3d(v) + o
    init_shift_wrt_fixedWholebrain = transform_points_by_transform_parameters_v2(pts=[init_shift_wrt_movingWholebrain],
    transform_parameters=global_transform_parameters)[0]

    print init_shift_wrt_movingWholebrain, init_shift_wrt_fixedWholebrain

    aligner = Aligner(volume_fixed, volume_moving, labelIndexMap_m2f=local_aligner_parameters['label_mapping_m2f'])
    aligner.set_centroid(centroid_m=init_shift_wrt_movingWholebrain, centroid_f=init_shift_wrt_fixedWholebrain)
    aligner.set_label_weights(label_weights=local_aligner_parameters['label_weights_m'])

    gradients_f = compute_gradient_v2(volume_fixed, smooth_first=True)

    # gradients_f = {}
    # for ind_f, struct_f in local_aligner_parameters['label_to_structure_fixed'].iteritems():
    #     tmpl = DataManager.get_volume_gradient_filepath_template_v3(stack_spec=local_alignment_spec['stack_f'], structure=struct_f)
    #     gradients_f[ind_f] = np.asarray([bp.unpack_ndarray_file(tmpl % {'suffix': 'gx'}),
    #                          bp.unpack_ndarray_file(tmpl % {'suffix': 'gy'}),
    #                          bp.unpack_ndarray_file(tmpl % {'suffix': 'gz'})])

    aligner.load_gradient(gradients=gradients_f)


    init_T = transform_parameters_to_init_T_v2(global_transform_parameters, local_aligner_parameters,
                                   centroid_m=init_shift_wrt_movingWholebrain, 
                                    centroid_f=init_shift_wrt_fixedWholebrain)
    print 'init_T', init_T

    # Tuning learning rate:
    # lr1=10., if lucky converges much faster than lr1=1., but sometimes stuck in local maxima
    # If lr1=1., grad_computation_sample_number=1e5 is sufficient.

    trial_num = 1

    T_all_trials = []
    scores_all_trials = []
    traj_all_trials = []

    for _ in range(trial_num):

        try:
            T, scores = aligner.optimize(tf_type=local_aligner_parameters['transform_type'],
                                         max_iter_num=10000,
                                         history_len=20,
                                         terminate_thresh_rot=.001,
                                         terminate_thresh_trans=.01,
                                    grad_computation_sample_number=local_aligner_parameters['grad_computation_sample_number'],
                                         lr1=1, lr2=.01,
                                         init_T=init_T.flatten()
                                        )
            T_all_trials.append(T)
            scores_all_trials.append(scores)
            traj_all_trials.append(aligner.Ts)

        except Exception as e:
            sys.stderr.write('%s\n' % e)



    Ts = np.array(aligner.Ts)

    plt.plot(Ts[:, [0,1,2,4,5,6,8,9,10]]);
    plt.title('rotational params');
    plt.xlabel('Iteration');
    plt.show();

    plt.plot(Ts[:, [3,7,11]]);
    plt.title('translation params');
    plt.xlabel('Iteration');
    plt.show();

    best_trial = np.argsort([np.max(scores) for scores in scores_all_trials])[-1]
    # T = T_all_trials[best_trial]
    T = traj_all_trials[best_trial][-1]
    scores = scores_all_trials[best_trial]
    print 'Best trial:', best_trial
    print max(scores), scores[-1]

    print T.reshape((3,4))
    plt.figure();
    plt.plot(scores);
    plt.show();

    transform_parameters = {
        'parameters': T_all_trials[best_trial],
        'centroid_m_wrt_wholebrain': aligner.centroid_m,
        'centroid_f_wrt_wholebrain': aligner.centroid_f,
    }

    DataManager.save_alignment_results_v2(transform_parameters=transform_parameters,
                       score_traj=scores_all_trials[best_trial],
                       parameter_traj=traj_all_trials[best_trial],
                      alignment_spec=local_alignment_spec)

    transform_parameters = DataManager.load_alignment_parameters_v3(alignment_spec=local_alignment_spec)

    
    stack_m_spec = local_alignment_spec['stack_m']
    
    for name_s in [structure_m]:

        volume, bbox_wrt_atlasSpace = \
            DataManager.load_original_volume_v2(stack_m_spec, structure=name_s, bbox_wrt='atlasSpace')

        transformed_vol, transformed_vol_box_wrt_fixedWholebrain = transform_volume_by_alignment_parameters_v2(volume, bbox=bbox_wrt_atlasSpace, transform_parameters=transform_parameters)
        print transformed_vol.shape, transformed_vol_box_wrt_fixedWholebrain
        DataManager.save_transformed_volume(volume=transformed_vol, 
                                            bbox=transformed_vol_box_wrt_fixedWholebrain, 
                                            alignment_spec=local_alignment_spec, 
                                            resolution='%.1fum' % local_aligner_parameters['resolution_um'], 
                                            structure=name_s)
