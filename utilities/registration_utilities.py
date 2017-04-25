"""Functions related to registration."""

import numpy as np
import sys
import os

from skimage.morphology import binary_closing, disk, binary_dilation, binary_erosion
from skimage.measure import grid_points_in_poly, subdivide_polygon, approximate_polygon
from skimage.measure import find_contours, regionprops
from shapely.geometry import Polygon
import cv2

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from distributed_utilities import download_from_s3_to_ec2

def parallel_where_binary(binary_volume, num_samples=None):

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

def affine_components_to_vector(tx,ty,tz,theta_xy,theta_yz=None,theta_xz=None,c=(0,0,0)):
    """
    Args:
        theta_xy (float):
            in radian.
    """
    cos_theta = np.cos(theta_xy)
    sin_theta = np.sin(theta_xy)
    Rz = np.array([[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]])
    Rx = np.array([[0, 0, 0], [0, cos_theta, -sin_theta], [0, sin_theta, cos_theta]])
    Ry = np.array([[cos_theta, 0, -sin_theta], [0, 0, 0], [sin_theta, 0, cos_theta]])
    tt = np.dot(Rz, np.r_[tx,ty,tz]-c) + c
    return np.ravel(np.c_[Rz, tt])

def rotate_transform_vector(v, theta_xy=None,theta_yz=None,theta_xz=None,c=(0,0,0)):
    """
    v is 12-length parameter.
    """
    cos_theta_z = np.cos(theta_xy)
    sin_theta_z = np.sin(theta_xy)
    Rz = np.array([[cos_theta_z, -sin_theta_z, 0], [sin_theta_z, cos_theta_z, 0], [0, 0, 1]])
    cos_theta_x = np.cos(theta_yz)
    sin_theta_x = np.sin(theta_yz)
    Rx = np.array([[0, 0, 0], [0, cos_theta_x, -sin_theta_x], [0, sin_theta_x, cos_theta_x]])
    cos_theta_y = np.cos(theta_xz)
    sin_theta_y = np.sin(theta_xz)
    Ry = np.array([[cos_theta_y, 0, -sin_theta_y], [0, 0, 0], [sin_theta_y, 0, cos_theta_y]])

    R = np.zeros((3,3))
    R[0] = v[:3]
    R[1] = v[4:7]
    R[2] = v[8:11]
    R_new = np.dot(Rx, np.dot(Ry, np.dot(Rz, R)))
    t_new = t + c - np.dot(R_new, c)
    return np.ravel(np.c_[R_new, t_new])

# from joblib import Parallel, delayed
import time
from lie import matrix_exp_v
import logging

from multiprocess import Pool

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
                labelIndexMap_m2f=None, label_weights=None, reg_weights=None, zrange=None):
        """
        Variant that takes in two probabilistic volumes.

        zrange: tuple
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

            nzvoxels_m_ = [parallel_where_binary(volume_m[i] > 0) for i in list(self.all_indices_m)]
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

    def set_centroid(self, centroid_m=None, centroid_f=None, indices_m=None):

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
            elif centroid_f == 'volume_centroid':
                self.centroid_f = np.r_[.5*self.xdim_f, .5*self.ydim_f, .5*self.zdim_f]
            elif centroid_f == 'origin':
                self.centroid_f = np.zeros((3,))
            else:
                raise Exception('centroid_f not recognized.')

        print 'm:', self.centroid_m, 'f:', self.centroid_f

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

    def load_gradient(self, gradient_filepath_map_f, indices_f=None):
        """Load gradients.

        Need to pass gradient_filepath_map_f in from outside because Aligner class should be agnostic about structure names.

        Args:
            gradient_filepath_map_f (dict of str): path string that contains formatting parts and (suffix).

        """

        if indices_f is None:
            indices_f = set([self.labelIndexMap_m2f[ind_m] for ind_m in self.all_indices_m])
            print indices_f

        global grad_f
        grad_f = {ind_f: np.zeros((3, self.ydim_f, self.xdim_f, self.zdim_f), dtype=np.float16) for ind_f in indices_f}

        t1 = time.time()

        for ind_f in indices_f:

            t = time.time()
            
            download_from_s3_to_ec2(gradient_filepath_map_f[ind_f] % {'suffix': 'gx'}, is_dir=False)
            download_from_s3_to_ec2(gradient_filepath_map_f[ind_f] % {'suffix': 'gy'}, is_dir=False)
            download_from_s3_to_ec2(gradient_filepath_map_f[ind_f] % {'suffix': 'gz'}, is_dir=False)
            
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

    def get_valid_voxels_after_transform(self, T, ind_m, return_valid):

        pts_prime = transform_points(np.array(T), pts_centered=nzvoxels_centered_m[ind_m], c_prime=self.centroid_f).astype(np.int16)

        # print 'before'
        # print nzvoxels_centered_m[ind_m]
        # print 'after'
        # print pts_prime

        xs_prime, ys_prime, zs_prime = pts_prime.T
        valid = (xs_prime >= 0) & (ys_prime >= 0) & (zs_prime >= 0) & \
                (xs_prime < self.xdim_f) & (ys_prime < self.ydim_f) & (zs_prime < self.zdim_f)

        # print pts_prime.max(axis=0), pts_prime.min(axis=0)
        # print self.xdim_f, self.ydim_f, self.zdim_f

        # print len(pts_prime), np.count_nonzero(valid)

        if np.any(valid):
            xs_prime_valid = xs_prime[valid]
            ys_prime_valid = ys_prime[valid]
            zs_prime_valid = zs_prime[valid]

            if return_valid:
                return xs_prime_valid, ys_prime_valid, zs_prime_valid, valid
            else:
                return xs_prime_valid, ys_prime_valid, zs_prime_valid


    def compute_score_and_gradient_one(self, T, num_samples=None, wrt_v=False, ind_m=None):

        try:
            s, xs_prime_valid, ys_prime_valid, zs_prime_valid, valid_moving_voxel_indices = self.compute_score_one(T, ind_m, return_valid=True)
        except Exception as e:
            sys.stderr.write(e.message + '\n')
            return 0, 0

        xs_valid, ys_valid, zs_valid = nzvoxels_m[ind_m][valid_moving_voxel_indices].T
        S_m_valid_scores = volume_m[ind_m][ys_valid, xs_valid, zs_valid]

        # score += label_weights[name] * voxel_probs_valid.sum()
        # score += s

        ind_f = self.labelIndexMap_m2f[ind_m]
        Sx = grad_f[ind_f][0, ys_prime_valid, xs_prime_valid, zs_prime_valid]
        Sy = grad_f[ind_f][1, ys_prime_valid, xs_prime_valid, zs_prime_valid]
        Sz = grad_f[ind_f][2, ys_prime_valid, xs_prime_valid, zs_prime_valid]
        dxs, dys, dzs = nzvoxels_centered_m[ind_m][valid_moving_voxel_indices].T

        xs_prime_valid = xs_prime_valid.astype(np.float)
        ys_prime_valid = ys_prime_valid.astype(np.float)
        zs_prime_valid = zs_prime_valid.astype(np.float)

        if num_samples is not None:
            n = np.count_nonzero(valid_moving_voxel_indices)
            n_sample = min(int(num_samples), n)
            ii = np.random.choice(range(n), n_sample, replace=False)
            # sys.stderr.write('%d samples\n' % n_sample)
            Sx = Sx[ii]
            Sy = Sy[ii]
            Sz = Sz[ii]
            xs_prime_valid = xs_prime_valid[ii]
            ys_prime_valid = ys_prime_valid[ii]
            zs_prime_valid = zs_prime_valid[ii]
            dxs = dxs[ii]
            dys = dys[ii]
            dzs = dzs[ii]

            S_m_valid_scores = S_m_valid_scores[ii]

        if wrt_v:
            q = np.c_[Sx, Sy, Sz,
                    -Sy*zs_prime_valid + Sz*ys_prime_valid,
                    Sx*zs_prime_valid - Sz*xs_prime_valid,
                    -Sx*ys_prime_valid + Sy*xs_prime_valid]
        else:
            q = np.c_[Sx*dxs, Sx*dys, Sx*dzs, Sx, Sy*dxs, Sy*dys, Sy*dzs, Sy, Sz*dxs, Sz*dys, Sz*dzs, Sz]

        # dMdA += label_weights[name] * q.sum(axis=0)
        # grad += q.sum(axis=0)
        # grad += (S_m_valid_scores[:,None] * q).sum(axis=0)
        # grad = q.sum(axis=0)

        # Whether to scale gradient to match the scores' scale depends on whether AdaGrad is used;
        # if used, then the scale will be automatically adapted so the scaling does not matter
        grad = q.sum(axis=0) / 1e6

        # regularized version
        tx = T[3]
        ty = T[7]
        tz = T[11]

        if wrt_v:
            grad[0] = grad[0] - 2*self.reg_weights[0] * tx
            # print grad[0], 2*self.reg_weights[0] * tx
            grad[1] = grad[1] - 2*self.reg_weights[1] * ty
            grad[2] = grad[2] - 2*self.reg_weights[2] * tz
        else:
            grad[3] = grad[3] - 2*self.reg_weights[0] * tx
            # print grad[3], 2*self.reg_weights[0] * tx
            grad[7] = grad[7] - 2*self.reg_weights[1] * ty
            grad[11] = grad[11] - 2*self.reg_weights[2] * tz

        # del q, Sx, Sy, Sz, dxs, dys, dzs, xs_prime_valid, ys_prime_valid, zs_prime_valid
        del q, Sx, Sy, Sz, dxs, dys, dzs, xs_prime_valid, ys_prime_valid, zs_prime_valid, \
            xs_valid, ys_valid, zs_valid, S_m_valid_scores

        return s, grad

    def compute_score_and_gradient(self, T, num_samples=None, wrt_v=False, indices_m=None):
        """
        Compute score and gradient.
        v is update on the Lie space.

        Args:
            T ((12,) vector): transform parameters
            num_samples (int): Number of sample points to compute gradient.
            wrt_v (bool): if true, compute gradient with respect to (tx,ty,tz,w1,w2,w3);
                            otherwise, compute gradient with respect to 12 parameters.
            indices_m (integer list):

        Returns:
            (tuple): tuple containing:

            - score (int): score
            - grad (float): gradient
        """

        score = 0

        if wrt_v:
            grad = np.zeros((6,))
        else:
            grad = np.zeros((12,))

        if indices_m is None:
            indices_m = self.all_indices_m

        # serial
        for ind_m in indices_m:
            score_one, grad_one = self.compute_score_and_gradient_one(T, num_samples, wrt_v, ind_m)
            # grad += grad_one
            # score += score_one
            grad += self.label_weights[ind_m] * grad_one
            score += self.label_weights[ind_m] * score_one

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


    def compute_score_one(self, T, ind_m, return_valid=False):
        """
        Compute score for one label.
        Notice that raw overlap score is divided by 1e6 before returned.

        Args:
            T ((12,) vector): transform parameters
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

        res = self.get_valid_voxels_after_transform(T, ind_m, return_valid=True)

        if res is None:
            # sys.stderr.write('No valid voxels after transform.\n')
            raise Exception('No valid voxels after transform: ind_m = %d' % ind_m)

        # if return_valid:
        xs_prime_valid, ys_prime_valid, zs_prime_valid, valid_moving_voxel_indices = res
        # else:
        #     xs_prime_valid, ys_prime_valid, zs_prime_valid = res

        ind_f = self.labelIndexMap_m2f[ind_m]

        # Reducing the scale of voxel value is important for keeping the sum in the represeantable range of the chosen data type.
        voxel_probs_valid = volume_f[ind_f][ys_prime_valid, xs_prime_valid, zs_prime_valid] / 1e6

        # sys.stderr.write('%d\n' % len(voxel_probs_valid))

        # xs_valid, ys_valid, zs_valid = nzvoxels_m[ind_m][valid_moving_voxel_indices].T
        # voxel_probs_valid = volume_m[ind_m][ys_valid, xs_valid, zs_valid] * volume_f[ind_f][ys_prime_valid, xs_prime_valid, zs_prime_valid] / 1e6

        # print np.histogram(voxel_probs_valid)

        # Unregularized version
        # s = voxel_probs_valid.sum()

        # Regularized version
        s = voxel_probs_valid.sum()

        tx = T[3]
        ty = T[7]
        tz = T[11]
        s_reg = self.reg_weights[0]*tx**2 + self.reg_weights[1]*ty**2 + self.reg_weights[2]*tz**2
        s = s - s_reg

        # sys.stderr.write('score %f\n' % s)

        if return_valid:
            return s, xs_prime_valid, ys_prime_valid, zs_prime_valid, valid_moving_voxel_indices
        else:
            return s

    def compute_score(self, T, indices_m=None, return_individual_score=False):
        """Compute score.

        Returns:
            pass
        """

        if indices_m is None:
            indices_m = self.all_indices_m

        score_all_landmarks = {}
        for ind_m in indices_m:
            try:
                score_all_landmarks[ind_m] = self.compute_score_one(T, ind_m, return_valid=False)
            except Exception as e:
                # sys.stderr.write(e.message + '\n')
                score_all_landmarks[ind_m] = 0

        # score = np.sum(score_all_landmarks.values())

        score = 0
        for ind_m, score_one in score_all_landmarks.iteritems():
            score += self.label_weights[ind_m] * score_one

        if return_individual_score:
            return score, score_all_landmarks
        else:
            return score


    def compute_scores_neighborhood_grid(self, params, dxs, dys, dzs, indices_m=None):

        from itertools import product

        # scores = np.reshape([self.compute_score(params + (0.,0.,0., dx, 0.,0.,0., dy, 0.,0.,0., dz), indices_m=indices_m)
        #                     for dx, dy, dz in product(dxs, dys, dzs)],
        #                     (dxs.size, dys.size, dzs.size))

        #parallel
        pool = Pool(processes=12)
        scores = pool.map(lambda (dx, dy, dz): self.compute_score(params + (0.,0.,0., dx, 0.,0.,0., dy, 0.,0.,0., dz), indices_m=indices_m),
                        product(dxs, dys, dzs))
        pool.close()
        pool.join()

        # scores = np.reshape(Parallel(n_jobs=12)(delayed(compute_score)(params + (0.,0.,0., dx, 0.,0.,0., dy, 0.,0.,0., dz))
        #                                         for dx, dy, dz in product(dxs, dys, dzs)),
        #                     (dxs.size, dys.size, dzs.size))

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
                    eta=3.):
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

            n = int(init_n*np.exp(-iteration/eta))

            sigma_tx = std_tx*np.exp(-iteration/eta)
            sigma_ty = std_ty*np.exp(-iteration/eta)
            sigma_tz = std_tz*np.exp(-iteration/eta)
            sigma_theta_xy = std_theta_xy*np.exp(-iteration/eta)

            tx_grid = init_tx + sigma_tx * np.r_[0, (2 * np.random.random(n) - 1)]
            ty_grid = init_ty + sigma_ty * np.r_[0, (2 * np.random.random(n) - 1)]
            tz_grid = init_tz + sigma_tz * np.r_[0, (2 * np.random.random(n) - 1)]
            theta_xy_grid = init_theta_xy + sigma_theta_xy * np.r_[0, (2 * np.random.random(n) - 1)]

            samples = np.c_[tx_grid, ty_grid, tz_grid, theta_xy_grid]

            t = time.time()

            # empirical speedup 7x
            # parallel
            if parallel:
                pool = Pool(processes=8)
                scores = pool.map(lambda (tx, ty, tz, theta_xy): self.compute_score(affine_components_to_vector(tx,ty,tz,theta_xy),
                                                        indices_m=indices_m), samples)
                pool.close()
                pool.join()
            else:
            # serial
                scores = [self.compute_score(affine_components_to_vector(tx,ty,tz,theta_xy), indices_m=indices_m)
                            for tx, ty, tz, theta_xy in samples]

            sys.stderr.write('grid search: %f seconds\n' % (time.time() - t)) # ~23s

            score_best = np.max(scores)

            tx_best, ty_best, tz_best, theta_xy_best = samples[np.argmax(scores)]

            sys.stderr.write('tx_best: %.2f (voxel), ty_best: %.2f, tz_best: %.2f, theta_xy_best: %.2f (deg)\n' % \
            (tx_best, ty_best, tz_best, np.rad2deg(theta_xy_best)))
            sys.stderr.write('sigma_tx: %.2f (voxel), sigma_ty: %.2f, sigma_tz: %.2f, sigma_theta_xy: %.2f (deg)\n' % \
            (sigma_tx, sigma_ty, sigma_tz, np.rad2deg(sigma_theta_xy)))

            if score_best > score_best_upToNow:
                # self.logger.info('%f %f', score_best_upToNow, score_best)
                sys.stderr.write('%f %f\n' % (score_best_upToNow, score_best))

                score_best_upToNow = score_best
                params_best_upToNow = tx_best, ty_best, tz_best, theta_xy_best

            if sigma_tx < 10:
                break

                # self.logger.info('%f %f %f', tx_best, ty_best, tz_best)
        print 'params_best_upToNow', np.ravel(params_best_upToNow)

        if return_best_score:
            return params_best_upToNow, score_best_upToNow
        else:
            return params_best_upToNow


    def optimize(self, type='rigid', init_T=None, label_weights=None, \
                grid_search_iteration_number=0, grid_search_sample_number=1000,
                grad_computation_sample_number=10000,
                max_iter_num=1000, history_len=200, terminate_thresh=.1, \
                indices_m=None, lr1=None, lr2=None, full_lr=None,
                std_tx=100, std_ty=100, std_tz=30, std_theta_xy=np.deg2rad(30),
                reg_weights=None,
                epsilon=1e-8):
        """Optimize

        reg_weights is for (tx,ty,tz)
        obj = texture score - reg_weights[0] * tx**2 - reg_weights[1] * ty**2 - reg_weights[2] * tz**2
        """

        if indices_m is None:
            indices_m = self.all_indices_m

        if label_weights is not None:
            self.set_label_weights(label_weights)

        if reg_weights is not None:
            self.set_regularization_weights(reg_weights)

        if type == 'rigid':
            grad_historical = np.zeros((6,))
            sq_updates_historical = np.zeros((6,))
            if lr1 is None:
                lr1 = 10.
            if lr2 is None:
                lr2 = 1e-1 # for Lie optimization, lr2 cannot be zero, otherwise causes error in computing scores.
        elif type == 'affine':
            grad_historical = np.zeros((12,))
            sq_updates_historical = np.zeros((12,))
            if lr1 is None:
                lr1 = 10
            if lr2 is None:
                lr2 = 1e-1
        else:
            raise Exception('Type must be either rigid or affine.')

        if grid_search_iteration_number > 0:
            (tx_best, ty_best, tz_best, theta_xy_best), grid_search_score = self.grid_search(grid_search_iteration_number, indices_m=indices_m,
                                                        init_n=grid_search_sample_number,
                                                        std_tx=std_tx, std_ty=std_ty, std_tz=std_tz, std_theta_xy=std_theta_xy,
                                                        return_best_score=True)
            # T = np.r_[1,0,0, tx_best, 0,1,0, ty_best, 0,0,1, tz_best]
            T = affine_components_to_vector(tx_best, ty_best, tz_best, theta_xy_best)
        elif init_T is None:
            T = np.r_[1,0,0,0,0,1,0,0,0,0,1,0]
        else:
            T = init_T

        score_best = -np.inf
        scores = []
        self.Ts = []

        for iteration in range(max_iter_num):

            # t = time.time()

            # self.logger.info('iteration %d', iteration)
            sys.stderr.write('iteration %d\n' % iteration)

            t = time.time()

            if type == 'rigid':
                # lr1, lr2 = (.1, 1e-2) # lr2 cannot be zero, otherwise causes error in computing scores.

                if full_lr is not None:
                    lr = full_lr
                else:
                    lr = np.r_[lr1,lr1,lr1,lr2,lr2,lr2]

                T, s, grad_historical, sq_updates_historical = self.step_lie(T, lr=lr,
                    grad_historical=grad_historical, sq_updates_historical=sq_updates_historical,
                    verbose=False, num_samples=grad_computation_sample_number,
                    indices_m=indices_m,
                    epsilon=epsilon)

                # print 'New T:', np.ravel(T)

            elif type == 'affine':

                if full_lr is not None:
                    lr = full_lr
                else:
                    lr = np.r_[lr2, lr2, lr2, lr1, lr2, lr2, lr2, lr1, lr2, lr2, lr2, lr1]

                T, s, grad_historical, sq_updates_historical = self.step_gd(T, lr=lr, \
                                grad_historical=grad_historical, sq_updates_historical=sq_updates_historical,
                                indices_m=indices_m)

            else:
                raise Exception('Type must be either rigid or affine.')

            sys.stderr.write('step: %.2f seconds\n' % (time.time() - t))

            # self.logger.info('score: %f', s)
            sys.stderr.write('score: %f\n' % s)
            scores.append(s)

            self.Ts.append(T)

            # sys.stderr.write('%f seconds\n' % (time.time()-t)) # 1.77s/iteration

            if iteration > 2*history_len:
                if np.abs(np.mean(scores[iteration-history_len:iteration]) - \
                          np.mean(scores[iteration-2*history_len:iteration-history_len])) < terminate_thresh:
                    break

            if s > score_best:
                best_gradient_descent_params = T
                score_best = s

        if grid_search_iteration_number > 0:
            if scores[-1] <= grid_search_score:
                # raise Exception('Gradient descent does not converge to higher than grid search score. Likely stuck at local minima.')
                sys.stderr.write('Gradient descent does not converge to higher than grid search score. Likely stuck at local minima.\n')

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
        score, grad = self.compute_score_and_gradient(T, num_samples=num_samples, wrt_v=True, indices_m=indices_m)

        # print 'score:', score
        # print 'grad:', grad

        # # AdaGrad Rule
        grad_historical += grad**2
        grad_adjusted = grad /  np.sqrt(grad_historical + epsilon)
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

        if verbose:
            print 'v_opt:', v_opt

        theta = np.sqrt(np.sum(v_opt[3:]**2))
        if verbose:
            print 'theta:', theta
        assert theta < np.pi

        exp_w, Vt = matrix_exp_v(v_opt)

        if verbose:
            print 'Vt:' , Vt

        Tm = np.reshape(T, (3,4))
        t = Tm[:, 3]
        R = Tm[:, :3]

        R_new = np.dot(exp_w, R)
        t_new = np.dot(exp_w, t) + Vt

        if verbose:
            print '\n'

        return np.column_stack([R_new, t_new]).flatten(), score, grad_historical, sq_updates_historical

    def step_gd(self, T, lr, grad_historical, sq_updates_historical, surround=False, surround_weight=2., num_samples=None, indices_m=None):
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

        score, grad = self.compute_score_and_gradient(T, num_samples=num_samples, indices_m=indices_m)

        # if surround:
        #     s_surr, dMdA_surr = compute_score_and_gradient(T, name, surround=True, num_samples=num_samples)
        #     dMdA -= surround_weight * dMdA_surr
        #     score -= surround_weight * s_surr

        # AdaGrad Rule
        grad_historical += grad**2
        grad_adjusted = grad / np.sqrt(grad_historical + 1e-10)
        new_T = T + lr*grad_adjusted

        # AdaDelta Rule
        # gamma = .9
        # epsilon = 1e-10
        # grad_historical = gamma * grad_historical + (1-gamma) * grad**2
        # update = np.sqrt(sq_updates_historical + epsilon)/np.sqrt(grad_historical + epsilon)*grad
        # new_T = T + update
        # sq_updates_historical = gamma * sq_updates_historical + (1-gamma) * update**2

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


def find_contour_points(labelmap, sample_every=10, min_length=10, padding=5):
    """
    The value of sample_every can be interpreted as distance between points.
    return is (x,y)
    Note the returned result is a dict of lists.
    """

    regions = regionprops(labelmap.astype(np.int))

    contour_points = {}

    for r in regions:

        (min_row, min_col, max_row, max_col) = r.bbox

        padded = np.pad(r.filled_image, ((padding,padding),(padding,padding)),
                        mode='constant', constant_values=0)

        contours = find_contours(padded, .5, fully_connected='high')
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
    cnts = find_contours(levelset, .5)
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


def extract_contours_from_labeled_volume(stack, volume,
                            section_z_map=None,
                            downsample_factor=None,
                            volume_limits=None,
                            labels=None, extrapolate_num_section=0,
                            force=True, filepath=None):
    """
    Extract labeled contours from a labeled volume.
    """

    if volume == 'localAdjusted':
        volume = bp.unpack_ndarray_file(volume_dir + '/%(stack)s/%(stack)s_localAdjustedVolume.bp'%{'stack':stack})
    elif volume == 'globalAligned':
        volume = bp.unpack_ndarray_file(volume_dir + '/%(stack)s/%(stack)s_atlasProjectedVolume.bp'%{'stack':stack})
    else:
        raise 'Volume unknown.'

    if filepath is None:
        filepath = volume_dir + '/initCntsAllSecs_%s.pkl' % stack

    if os.path.exists(filepath) and not force:
        init_cnts_allSecs = pickle.load(open(filepath, 'r'))
    else:
        if volume_limits is None:
            volume_xmin, volume_xmax, volume_ymin, volume_ymax, volume_zmin, volume_zmax = \
            np.loadtxt(os.path.join(volume_dir, '%(stack)s/%(stack)s_scoreVolume_limits.txt' % {'stack': stack}), dtype=np.int)

        if section_z_map is None:
            assert downsample_factor is not None, 'Because section_z_map is not given, must specify downsample_factor.'
            z_section_map = find_z_section_map(stack, volume_zmin, downsample_factor=downsample_factor)
            section_z_map = {sec: z for z, sec in z_section_map.iteritems()}

        init_cnts_allSecs = {}

        first_detect_sec, last_detect_sec = detect_bbox_range_lookup[stack]

        for sec in range(first_detect_sec, last_detect_sec+1):

            z = section_z_map[sec]
            projected_annotation_labelmap = volume[..., z]

            init_cnts = find_contour_points(projected_annotation_labelmap) # downsampled 16
            init_cnts = dict([(labels[label_ind], (cnts[0]+(volume_xmin, volume_ymin))*2)
                              for label_ind, cnts in init_cnts.iteritems()])

            # extend contour to copy annotations of undetected classes from neighbors
            if extrapolate_num_section > 0:

                sss = np.empty((2*extrapolate_num_section,), np.int)
                sss[1::2] = -np.arange(1, extrapolate_num_section+1)
                sss[::2] = np.arange(1, extrapolate_num_section+1)

                Ls = []
                for ss in sss:
                    sec2 = sec + ss
                    z2 = section_z_map[sec2]
                    if z2 >= volume.shape[2] or z2 < 0:
                        continue

                    init_cnts2 = find_contour_points(volume[..., z2]) # downsampled 16
                    init_cnts2 = dict([(labels[label_ind], (cnts[0]+(volume_xmin, volume_ymin))*2)
                                      for label_ind, cnts in init_cnts2.iteritems()])
                    Ls.append(init_cnts2)

                for ll in Ls:
                    for l, c in ll.iteritems():
                        if l not in init_cnts:
                            init_cnts[l] = c

            init_cnts_allSecs[sec] = init_cnts

        pickle.dump(init_cnts_allSecs, open(filepath, 'w'))

    return init_cnts_allSecs


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
    # print Tm
    t = Tm[:, 2]
    A = Tm[:, :2]

    pts_prime = np.dot(A, pts_centered.T) + (t + c_prime)[:,None]

    # print t, c_prime
    # print pts_centered
    # print pts_prime.T

    return pts_prime.T


def transform_points(T, pts=None, c=None, pts_centered=None, c_prime=0):
    '''
    T: 1x12 vector
    c: center of volume 1
    c_prime: center of volume 2
    pts: nx3
    '''

    if pts_centered is None:
        pts_centered = pts - c

    Tm = np.reshape(T, (3,4))
    # print Tm
    t = Tm[:, 3]
    A = Tm[:, :3]

    pts_prime = np.dot(A, pts_centered.T) + (t + c_prime)[:,None]

    # print t, c_prime
    # print pts_centered
    # print pts_prime.T

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

def transform_points_inverse(T, pts_prime=None, c_prime=None, pts_prime_centered=None, c=0):
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

def transform_volume_polyrigid(vol, rigid_param_list, anchor_points, sigmas, weights):

    nzvoxels_m_temp = parallel_where_binary(vol > 0)

    if sigmas[0].ndim == 2: # sigma is covariance matrix
        nzvoxels_weights = np.array([w*np.exp(-mahalanobis_distance_sq(nzvoxels_m_temp, ap, sigma))
                            for ap, sigma, w in zip(anchor_points, sigmas, weights)])
    elif sigmas[0].ndim == 1: # sigma is a single scalar
        nzvoxels_weights = np.array([w*np.exp(-np.sum((nzvoxels_m_temp - ap)**2, axis=1)/sigma**2) \
                            for ap, sigma, w in zip(anchor_points, sigmas, weights)])
    nzvoxels_weights = nzvoxels_weights/nzvoxels_weights.sum(axis=0)
    # nzvoxels_weights[nzvoxels_weights < 1e-1] = 0
    # nzvoxels_weights = nzvoxels_weights/nzvoxels_weights.sum(axis=0)
    # n_components x n_voxels

    nzs_m_aligned_to_f = np.zeros((len(nzvoxels_m_temp), 3), dtype=np.float16)

    for i, rigid_params in enumerate(rigid_param_list):

        params, centroid_m, centroid_f, \
        xdim_m, ydim_m, zdim_m, \
        xdim_f, ydim_f, zdim_f = rigid_params

        nzs_m_aligned_to_f += nzvoxels_weights[i][:,None] * transform_points(params, pts=nzvoxels_m_temp,
                            c=centroid_m, c_prime=centroid_f).astype(np.float16)

    nzs_m_aligned_to_f = nzs_m_aligned_to_f.astype(np.int16)
    xs_f, ys_f, zs_f = nzs_m_aligned_to_f.T

    volume_m_aligned_to_f = np.zeros((ydim_f, xdim_f, zdim_f), vol.dtype)

    valid = (xs_f >= 0) & (ys_f >= 0) & (zs_f >= 0) & \
    (xs_f < xdim_f) & (ys_f < ydim_f) & (zs_f < zdim_f)

    xs_m, ys_m, zs_m = nzvoxels_m_temp.T

    volume_m_aligned_to_f[ys_f[valid], xs_f[valid], zs_f[valid]] = \
    vol[ys_m[valid], xs_m[valid], zs_m[valid]]

    if np.issubdtype(volume_m_aligned_to_f.dtype, np.float):
        # score volume
        dense_volume = fill_sparse_score_volume(volume_m_aligned_to_f)
    elif np.issubdtype(volume_m_aligned_to_f.dtype, np.integer):
        dense_volume = fill_sparse_volume(volume_m_aligned_to_f)
    else:
        raise Exception('transform_volume: Volume must be either float or int.')

    return dense_volume


def transform_volume(vol, global_params, centroid_m, centroid_f, xdim_f, ydim_f, zdim_f):

    nzvoxels_m_temp = parallel_where_binary(vol > 0)
    # "_temp" is appended to avoid name conflict with module level variable defined in registration.py

    nzs_m_aligned_to_f = transform_points(global_params, pts=nzvoxels_m_temp,
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
        # score volume
        dense_volume = fill_sparse_score_volume(volume_m_aligned_to_f)
    elif np.issubdtype(volume_m_aligned_to_f.dtype, np.integer):
        dense_volume = fill_sparse_volume(volume_m_aligned_to_f)
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
        # score volume
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

    from registration_utilities import find_contour_points
    from annotation_utilities import points_inside_contour

    volume = np.zeros_like(volume_sparse, np.int8)

    for z in range(volume_sparse.shape[-1]):
        for ind, cnts in find_contour_points(volume_sparse[..., z]).iteritems():
            cnt = cnts[np.argsort(map(len, cnts))[-1]]
            pts = points_inside_contour(cnt)
            volume[pts[:,1], pts[:,0], z] = ind
    return volume


def annotation_volume_to_score_volume(ann_vol, label_to_structure):
    all_indices = set(np.unique(ann_vol)) - {0}
    volume_f = {label_to_structure[i]: np.zeros_like(ann_vol, dtype=np.float16) for i in all_indices}
    for i in all_indices:
        mask = ann_vol == i
        volume_f[label_to_structure[i]][mask] = 1.
        del mask
    return volume_f
