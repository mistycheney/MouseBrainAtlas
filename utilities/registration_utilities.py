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


from joblib import Parallel, delayed
import time
from lie import matrix_exp_v
import logging

from multiprocess import Pool

volume_f = None
volume_m = None
nzvoxels_m = None
nzvoxels_centered_m = None
grad_f = None

#########################################################################

class Aligner3(object):
    def __init__(self, volume_f_, volume_m_=None, nzvoxels_m_=None, centroid_f=None, centroid_m=None, \
                label_weights=None, labelIndexMap_m2f=None, considered_indices_m=None):
        """
        Find the optimal transform of volume2 that aligns it with volume1.

        Args:
            volume_f (dict of 3d float array): the fixed volume(s) - subjects' score volumes, a probabilistic volume.
            volume_m (3d integer array): the moving volume - the atlas, an annotation volume. alternative is to provide `nzvoxels_m`.
            nzvoxels_m ((n,3) array): indices of active voxels in the moving volume
        """

        self.labelIndexMap_m2f = labelIndexMap_m2f
        self.all_indices_m = list(set(self.labelIndexMap_m2f.keys()) & set(np.unique(volume_m_)))
        self.all_indices_f = set([self.labelIndexMap_m2f[ind_m] for ind_m in self.all_indices_m])

        # If use self.logger, Pool will incur "illegal seek" error
        # self.logger = logging.getLogger(__name__)
        # self.logger.setLevel(logging.INFO)
        #
        global volume_f

        if isinstance(volume_f_, dict): # probabilistic volume
            volume_f = volume_f_
        else: # annotation volume
            volume_f = {i: np.zeros_like(volume_f_) for i in self.all_indices_f}
            for i in self.all_indices_f:
                mask = volume_f_ == i
                volume_f[i][mask] = volume_f_[mask]
                del mask

        global volume_m
        volume_m = volume_m_

        # self.volume_f = volume_f
        # self.volume_m = volume_m

        assert volume_f is not None, 'Template volume is not specified.'
        assert volume_m is not None or nzvoxels_m_ is not None, 'Moving volume is not specified.'

        self.ydim_m, self.xdim_m, self.zdim_m = volume_m.shape
        self.ydim_f, self.xdim_f, self.zdim_f = volume_f.values()[0].shape

        # if considered_indices_m is None:
        #     self.considered_indices_m = all_indices_m
        # else:
        #     self.considered_indices_m = considered_indices_m

        # self.considered_indices_f = [self.labelIndexMap_m2f[ind_m] for ind_m in self.considered_indices_m]

        global nzvoxels_m
        if nzvoxels_m_ is None:
            # nzvoxels_m_ = Parallel(n_jobs=16)(delayed(parallel_where)(volume_m, i, num_samples=int(1e5))
            #                             for i in self.all_indices_m)

            pool = Pool(16)
            nzvoxels_m_ = pool.map(lambda i: parallel_where(volume_m, i, num_samples=int(1e5)),
                                    self.all_indices_m)
            pool.close()
            pool.join()

            nzvoxels_m = dict(zip(self.all_indices_m, nzvoxels_m_))
        else:
            nzvoxels_m = nzvoxels_m_

        # self.set_centroid(centroid_m=centroid_m, centroid_f=centroid_f, indices_m=self.considered_indices_m)

    def set_centroid(self, centroid_m=None, centroid_f=None, indices_m=None):

        if indices_m is None:
            indices_m = self.all_indices_m

        if isinstance(centroid_m, basestring):
            if centroid_m == 'structure_centroid':
                self.centroid_m = np.concatenate([nzvoxels_m[i] for i in indices_m]).mean(axis=0)
            elif centroid_m == 'volume_centroid':
                self.centroid_m = np.r_[.5*self.xdim_m, .5*self.ydim_m, .5*self.zdim_m]
            else:
                raise Exception('centroid_m not recognized.')

        if isinstance(centroid_f, basestring):
            if centroid_f == 'centroid_m':
                self.centroid_f = self.centroid_m
            elif centroid_f == 'volume_centroid':
                self.centroid_f = np.r_[.5*self.xdim_f, .5*self.ydim_f, .5*self.zdim_f]
            else:
                raise Exception('centroid_f not recognized.')

        global nzvoxels_centered_m
        nzvoxels_centered_m = {ind_m: nzvs - self.centroid_m for ind_m, nzvs in nzvoxels_m.iteritems()}

    def load_gradient(self, gradient_filepath_map_f=None, indices_f=None):
        """Load gradients.

        Args:
            gradient_filepath_map_f (dict of str): path string that contains formatting parts and (suffix).
            If None, gradients will be computed.
        """

        if indices_f is None:
            indices_f = self.all_indices_f

        global grad_f
        grad_f = {ind_f: np.empty((3, self.ydim_f, self.xdim_f, self.zdim_f), dtype=np.float16) for ind_f in indices_f}

        t1 = time.time()

        for ind_f in indices_f:

            t = time.time()

            if gradient_filepath_map_f is None:
                gy_gx_gz = np.gradient(volume_f[ind_f].astype(np.float32), 3, 3, 3)
                grad_f[ind_f][0] = gy_gx_gz[1]
                grad_f[ind_f][1] = gy_gx_gz[0]
                grad_f[ind_f][2] = gy_gx_gz[2]

            else:

                assert gradient_filepath_map_f is not None
                grad_f[ind_f][0] = bp.unpack_ndarray_file(gradient_filepath_map_f[ind_f] % {'suffix': 'gx'})
                grad_f[ind_f][1] = bp.unpack_ndarray_file(gradient_filepath_map_f[ind_f] % {'suffix': 'gy'})
                grad_f[ind_f][2] = bp.unpack_ndarray_file(gradient_filepath_map_f[ind_f] % {'suffix': 'gz'})

            sys.stderr.write('load gradient %s: %f seconds\n' % (ind_f, time.time() - t)) # ~6s

        sys.stderr.write('overall: %f seconds\n' % (time.time() - t1)) # ~100s

    def get_valid_voxels_after_transform(self, T, ind_m, return_valid):

        pts_prime = transform_points(np.array(T), pts_centered=nzvoxels_centered_m[ind_m], c_prime=self.centroid_f).astype(np.int16)
        xs_prime, ys_prime, zs_prime = pts_prime.T
        valid = (xs_prime >= 0) & (ys_prime >= 0) & (zs_prime >= 0) & \
                (xs_prime < self.xdim_f) & (ys_prime < self.ydim_f) & (zs_prime < self.zdim_f)

        if np.any(valid):
            xs_prime_valid = xs_prime[valid]
            ys_prime_valid = ys_prime[valid]
            zs_prime_valid = zs_prime[valid]

            if return_valid:
                return xs_prime_valid, ys_prime_valid, zs_prime_valid, valid
            else:
                return xs_prime_valid, ys_prime_valid, zs_prime_valid

    def compute_score_and_gradient(self, T, num_samples=None, wrt_v=False, indices_m=None):
        """
        Compute score and gradient.

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

        for ind_m in indices_m:

            s, xs_prime_valid, ys_prime_valid, zs_prime_valid, valid = self.compute_score_one(T, ind_m, return_valid=True)

            # score += label_weights[name] * voxel_probs_valid.sum()
            score += s

            ind_f = self.labelIndexMap_m2f[ind_m]
            Sx = grad_f[ind_f][0, ys_prime_valid, xs_prime_valid, zs_prime_valid]
            Sy = grad_f[ind_f][1, ys_prime_valid, xs_prime_valid, zs_prime_valid]
            Sz = grad_f[ind_f][2, ys_prime_valid, xs_prime_valid, zs_prime_valid]
            dxs, dys, dzs = nzvoxels_centered_m[ind_m][valid].T

            xs_prime_valid = xs_prime_valid.astype(np.float)
            ys_prime_valid = ys_prime_valid.astype(np.float)
            zs_prime_valid = zs_prime_valid.astype(np.float)

            if num_samples is not None:
                n = np.count_nonzero(valid)
                ii = np.random.choice(range(n), min(int(num_samples), n), replace=False)
                Sx = Sx[ii]
                Sy = Sy[ii]
                Sz = Sz[ii]
                xs_prime_valid = xs_prime_valid[ii]
                ys_prime_valid = ys_prime_valid[ii]
                zs_prime_valid = zs_prime_valid[ii]
                dxs = dxs[ii]
                dys = dys[ii]
                dzs = dzs[ii]

            if wrt_v:
                q = np.c_[Sx, Sy, Sz,
                -Sy*zs_prime_valid + Sz*ys_prime_valid,
                Sx*zs_prime_valid - Sz*xs_prime_valid,
                -Sx*ys_prime_valid + Sy*xs_prime_valid]
            else:
                q = np.c_[Sx*dxs, Sx*dys, Sx*dzs, Sx, Sy*dxs, Sy*dys, Sy*dzs, Sy, Sz*dxs, Sz*dys, Sz*dzs, Sz]

            # dMdA += label_weights[name] * q.sum(axis=0)
            grad += q.sum(axis=0)

            del q, Sx, Sy, Sz, dxs, dys, dzs, xs_prime_valid, ys_prime_valid, zs_prime_valid

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

        res = self.get_valid_voxels_after_transform(T, ind_m, return_valid)

        if res is None:
            return 0

        if return_valid:
            xs_prime_valid, ys_prime_valid, zs_prime_valid, valid = res
        else:
            xs_prime_valid, ys_prime_valid, zs_prime_valid = res

        ind_f = self.labelIndexMap_m2f[ind_m]

        voxel_probs_valid = volume_f[ind_f][ys_prime_valid, xs_prime_valid, zs_prime_valid] / 1e6
        s = voxel_probs_valid.sum()

        if return_valid:
            return s, xs_prime_valid, ys_prime_valid, zs_prime_valid, valid
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
            score_all_landmarks[ind_m] = self.compute_score_one(T, ind_m, return_valid=False)

        score = np.sum(score_all_landmarks.values())

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

    def grid_search(self, grid_search_iteration_number, indices_m=None, init_n=1000, parallel=True):
        """Grid search.

        Args:
            grid_search_iteration_number (int): number of iteration

        Returns:
            params_best_upToNow ((12,) float array): found parameters

        """
        params_best_upToNow = (0, 0, 0)
        score_best_upToNow = 0

        if indices_m is None:
            indices_m = self.all_indices_m

        for iteration in range(grid_search_iteration_number):

            # self.logger.info('grid search iteration %d', iteration)

            init_tx, init_ty, init_tz  = params_best_upToNow

            n = int(init_n*np.exp(-iteration/3.))

            sigma_tx = 300*np.exp(-iteration/3.)
            sigma_ty = 300*np.exp(-iteration/3.)
            sigma_tz = 100*np.exp(-iteration/3.)

            tx_grid = init_tx + sigma_tx * (2 * np.random.random(n) - 1)
            ty_grid = init_ty + sigma_ty * (2 * np.random.random(n) - 1)
            tz_grid = init_tz + sigma_tz * (2 * np.random.random(n) - 1)

            samples = np.c_[tx_grid, ty_grid, tz_grid]

            t = time.time()

            # empirical speedup 7x
            # parallel
            if parallel:
                pool = Pool(processes=12)
                scores = pool.map(lambda (tx, ty, tz): self.compute_score(np.r_[1, 0, 0, tx, 0, 1, 0, ty, 0, 0, 1, tz], indices_m=indices_m), samples)
                pool.close()
                pool.join()
            else:
            # serial
                scores = [self.compute_score(np.r_[1, 0, 0, tx, 0, 1, 0, ty, 0, 0, 1, tz], indices_m=indices_m)
                            for tx, ty, tz in samples]

            sys.stderr.write('grid search: %f seconds\n' % (time.time() - t)) # ~23s

            score_best = np.max(scores)

            tx_best, ty_best, tz_best = samples[np.argmax(scores)]

            if score_best > score_best_upToNow:
                # self.logger.info('%f %f', score_best_upToNow, score_best)

                score_best_upToNow = score_best
                params_best_upToNow = tx_best, ty_best, tz_best

                # self.logger.info('%f %f %f', tx_best, ty_best, tz_best)

        return params_best_upToNow


    def optimize(self, type='rigid', init_T=None, label_weights=None, \
                grid_search_iteration_number=0, grid_search_sample_number=1000,
                grad_computation_sample_number=10000,
                max_iter_num=1000, history_len=200, terminate_thresh=.1, \
                indices_m=None):
        """Optimize"""

        if type == 'rigid':
            grad_historical = np.zeros((6,))
        elif type == 'affine':
            grad_historical = np.zeros((12,))
        else:
            raise Exception('Type must be either rigid or affine.')

        if indices_m is None:
            indices_m = self.all_indices_m

        if grid_search_iteration_number > 0:
            tx_best, ty_best, tz_best = self.grid_search(grid_search_iteration_number, indices_m=indices_m, init_n=grid_search_sample_number)
            T = np.r_[1,0,0, tx_best, 0,1,0, ty_best, 0,0,1, tz_best]
        elif init_T is None:
            T = np.r_[1,0,0,0,0,1,0,0,0,0,1,0]
        else:
            T = init_T

        score_best = 0
        scores = []

        for iteration in range(max_iter_num):

            # self.logger.info('iteration %d', iteration)

            if type == 'rigid':
                lr1, lr2 = (1., 1e-2)
                T, s, grad_historical = self.step_lie(T, lr=np.r_[lr1,lr1,lr1,lr2,lr2,lr2],
                    grad_historical=grad_historical, verbose=False, num_samples=grad_computation_sample_number,
                    indices_m=indices_m)
            elif type == 'affine':
                lr1, lr2 = (10., 1e-1)
                T, s, grad_historical = self.step_gd(T, lr=np.r_[lr2, lr2, lr2, lr1, lr2, lr2, lr2, lr1, lr2, lr2, lr2, lr1], \
                                                grad_historical=grad_historical,
                                                indices_m=indices_m)
            else:
                raise Exception('Type must be either rigid or affine.')

            # self.logger.info('score: %f', s)
            scores.append(s)

            if iteration > 2*history_len:
                if np.abs(np.mean(scores[iteration-history_len:iteration]) - \
                          np.mean(scores[iteration-2*history_len:iteration-history_len])) < terminate_thresh:
                    break

            if s > score_best:
                best_gradient_descent_params = T
                score_best = s

        return best_gradient_descent_params, scores

    def step_lie(self, T, lr, grad_historical, verbose=False, num_samples=1000, indices_m=None):
        """
        One optimization step over Lie group SE(3).

        Args:
            T ((12,) vector): flattened vector of 3x4 transform matrix
            lr ((12,) vector): learning rate
            grad_historical ((12,) vector): accumulated gradiant magnitude, for Adagrad

        Returns:
            (tuple): tuple containing:

                new_T ((12,) vector): the new parameters
                score (float): current score
                grad_historical ((12,) vector): new accumulated gradient magnitude, used for Adagrad
        """

        if indices_m is None:
            indices_m = self.all_indices_m

        score, grad = self.compute_score_and_gradient(T, num_samples=num_samples, wrt_v=True, indices_m=indices_m)

        grad_historical += grad**2
        grad_adjusted = grad / (1e-10 + np.sqrt(grad_historical))
        v_opt = lr * grad_adjusted # no minus sign because maximizing

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

        return np.column_stack([R_new, t_new]).flatten(), score, grad_historical

    def step_gd(self, T, lr, dMdA_historical, surround=False, surround_weight=2., num_samples=None, indices_m=None):
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

        grad_historical += grad**2
        grad_adjusted = grad / (1e-10 + np.sqrt(grad_historical))

        new_T = T + lr*grad_adjusted

        return new_T, score, grad_historical


#######################################################

class Aligner2(object):
    def __init__(self, volume_f_, volume_m_=None, nzvoxels_m_=None, centroid_f=None, centroid_m=None, \
                label_weights=None, labelIndexMap_m2f=None, considered_indices_m=None):
        """
        Find the optimal transform of volume2 that aligns it with volume1.

        Args:
            volume_f (dict of 3d float array): the fixed volume(s) - subjects' score volumes, a probabilistic volume.
            volume_m (3d integer array): the moving volume - the atlas, an annotation volume. alternative is to provide `nzvoxels_m`.
            nzvoxels_m ((n,3) array): indices of active voxels in the moving volume
        """

        # If use self.logger, Pool will incur "illegal seek" error
        # self.logger = logging.getLogger(__name__)
        # self.logger.setLevel(logging.INFO)
        #
        global volume_f
        volume_f = volume_f_

        global volume_m
        volume_m = volume_m_

        # self.volume_f = volume_f
        # self.volume_m = volume_m

        assert volume_f is not None, 'Template volume is not specified.'
        assert volume_m is not None or nzvoxels_m_ is not None, 'Moving volume is not specified.'

        self.ydim_m, self.xdim_m, self.zdim_m = volume_m.shape
        self.ydim_f, self.xdim_f, self.zdim_f = volume_f.values()[0].shape

        self.labelIndexMap_m2f = labelIndexMap_m2f

        self.all_indices_m = list(set(self.labelIndexMap_m2f.keys()) & set(np.unique(volume_m)))

        # if considered_indices_m is None:
        #     self.considered_indices_m = all_indices_m
        # else:
        #     self.considered_indices_m = considered_indices_m

        # self.considered_indices_f = [self.labelIndexMap_m2f[ind_m] for ind_m in self.considered_indices_m]

        global nzvoxels_m
        if nzvoxels_m_ is None:
            nzvoxels_m_ = Parallel(n_jobs=16)(delayed(parallel_where)(volume_m, i, num_samples=int(1e5))
                                        for i in self.all_indices_m)
            nzvoxels_m = dict(zip(self.all_indices_m, nzvoxels_m_))
        else:
            nzvoxels_m = nzvoxels_m_

        # self.set_centroid(centroid_m=centroid_m, centroid_f=centroid_f, indices_m=self.considered_indices_m)

    def set_centroid(self, centroid_m=None, centroid_f=None, indices_m=None):

        if indices_m is None:
            indices_m = self.all_indices_m

        if isinstance(centroid_m, basestring):
            if centroid_m == 'structure_centroid':
                self.centroid_m = np.concatenate([nzvoxels_m[i] for i in indices_m]).mean(axis=0)
            elif centroid_m == 'volume_centroid':
                self.centroid_m = np.r_[.5*self.xdim_m, .5*self.ydim_m, .5*self.zdim_m]
            else:
                raise Exception('centroid_m not recognized.')

        if isinstance(centroid_f, basestring):
            if centroid_f == 'centroid_m':
                self.centroid_f = self.centroid_m
            elif centroid_f == 'volume_centroid':
                self.centroid_f = np.r_[.5*self.xdim_f, .5*self.ydim_f, .5*self.zdim_f]
            else:
                raise Exception('centroid_f not recognized.')

        global nzvoxels_centered_m
        nzvoxels_centered_m = {ind_m: nzvs - self.centroid_m for ind_m, nzvs in nzvoxels_m.iteritems()}

    def load_gradient(self, gradient_filepath_map_f, indices_f=None):
        """Load gradients.

        Args:
            gradient_filepath_map_f (dict of str): path string that contains formatting parts and (suffix).
        """

        if indices_f is None:
            indices_f = set([self.labelIndexMap_m2f[ind_m] for ind_m in self.all_indices_m])

        global grad_f
        grad_f = {ind_f: np.empty((3, self.ydim_f, self.xdim_f, self.zdim_f), dtype=np.float16) for ind_f in indices_f}

        t1 = time.time()

        for ind_f in indices_f:

            t = time.time()

            assert gradient_filepath_map_f is not None
            grad_f[ind_f][0] = bp.unpack_ndarray_file(gradient_filepath_map_f[ind_f] % {'suffix': 'gx'})
            grad_f[ind_f][1] = bp.unpack_ndarray_file(gradient_filepath_map_f[ind_f] % {'suffix': 'gy'})
            grad_f[ind_f][2] = bp.unpack_ndarray_file(gradient_filepath_map_f[ind_f] % {'suffix': 'gz'})

            sys.stderr.write('load gradient %s: %f seconds\n' % (ind_f, time.time() - t)) # ~6s

        sys.stderr.write('overall: %f seconds\n' % (time.time() - t1)) # ~100s

    def get_valid_voxels_after_transform(self, T, ind_m, return_valid):

        pts_prime = transform_points(np.array(T), pts_centered=nzvoxels_centered_m[ind_m], c_prime=self.centroid_f).astype(np.int16)
        xs_prime, ys_prime, zs_prime = pts_prime.T
        valid = (xs_prime >= 0) & (ys_prime >= 0) & (zs_prime >= 0) & \
                (xs_prime < self.xdim_f) & (ys_prime < self.ydim_f) & (zs_prime < self.zdim_f)

        if np.any(valid):
            xs_prime_valid = xs_prime[valid]
            ys_prime_valid = ys_prime[valid]
            zs_prime_valid = zs_prime[valid]

            if return_valid:
                return xs_prime_valid, ys_prime_valid, zs_prime_valid, valid
            else:
                return xs_prime_valid, ys_prime_valid, zs_prime_valid

    def compute_score_and_gradient(self, T, num_samples=None, wrt_v=False, indices_m=None):
        """
        Compute score and gradient.

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

        for ind_m in indices_m:

            s, xs_prime_valid, ys_prime_valid, zs_prime_valid, valid = self.compute_score_one(T, ind_m, return_valid=True)

            # score += label_weights[name] * voxel_probs_valid.sum()
            score += s

            ind_f = self.labelIndexMap_m2f[ind_m]
            Sx = grad_f[ind_f][0, ys_prime_valid, xs_prime_valid, zs_prime_valid]
            Sy = grad_f[ind_f][1, ys_prime_valid, xs_prime_valid, zs_prime_valid]
            Sz = grad_f[ind_f][2, ys_prime_valid, xs_prime_valid, zs_prime_valid]
            dxs, dys, dzs = nzvoxels_centered_m[ind_m][valid].T

            xs_prime_valid = xs_prime_valid.astype(np.float)
            ys_prime_valid = ys_prime_valid.astype(np.float)
            zs_prime_valid = zs_prime_valid.astype(np.float)

            if num_samples is not None:
                n = np.count_nonzero(valid)
                ii = np.random.choice(range(n), min(int(num_samples), n), replace=False)
                Sx = Sx[ii]
                Sy = Sy[ii]
                Sz = Sz[ii]
                xs_prime_valid = xs_prime_valid[ii]
                ys_prime_valid = ys_prime_valid[ii]
                zs_prime_valid = zs_prime_valid[ii]
                dxs = dxs[ii]
                dys = dys[ii]
                dzs = dzs[ii]

            if wrt_v:
                q = np.c_[Sx, Sy, Sz,
                -Sy*zs_prime_valid + Sz*ys_prime_valid,
                Sx*zs_prime_valid - Sz*xs_prime_valid,
                -Sx*ys_prime_valid + Sy*xs_prime_valid]
            else:
                q = np.c_[Sx*dxs, Sx*dys, Sx*dzs, Sx, Sy*dxs, Sy*dys, Sy*dzs, Sy, Sz*dxs, Sz*dys, Sz*dzs, Sz]

            # dMdA += label_weights[name] * q.sum(axis=0)
            grad += q.sum(axis=0)

            del q, Sx, Sy, Sz, dxs, dys, dzs, xs_prime_valid, ys_prime_valid, zs_prime_valid

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

        res = self.get_valid_voxels_after_transform(T, ind_m, return_valid)

        if res is None:
            return 0

        if return_valid:
            xs_prime_valid, ys_prime_valid, zs_prime_valid, valid = res
        else:
            xs_prime_valid, ys_prime_valid, zs_prime_valid = res

        ind_f = self.labelIndexMap_m2f[ind_m]
        voxel_probs_valid = volume_f[ind_f][ys_prime_valid, xs_prime_valid, zs_prime_valid] / 1e6
        s = voxel_probs_valid.sum()

        if return_valid:
            return s, xs_prime_valid, ys_prime_valid, zs_prime_valid, valid
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
            score_all_landmarks[ind_m] = self.compute_score_one(T, ind_m, return_valid=False)

        score = np.sum(score_all_landmarks.values())

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

    def grid_search(self, grid_search_iteration_number, indices_m=None, init_n=1000, parallel=True):
        """Grid search.

        Args:
            grid_search_iteration_number (int): number of iteration

        Returns:
            params_best_upToNow ((12,) float array): found parameters

        """
        params_best_upToNow = (0, 0, 0)
        score_best_upToNow = 0

        if indices_m is None:
            indices_m = self.all_indices_m

        for iteration in range(grid_search_iteration_number):

            # self.logger.info('grid search iteration %d', iteration)

            init_tx, init_ty, init_tz  = params_best_upToNow

            n = int(init_n*np.exp(-iteration/3.))

            sigma_tx = 300*np.exp(-iteration/3.)
            sigma_ty = 300*np.exp(-iteration/3.)
            sigma_tz = 100*np.exp(-iteration/3.)

            tx_grid = init_tx + sigma_tx * (2 * np.random.random(n) - 1)
            ty_grid = init_ty + sigma_ty * (2 * np.random.random(n) - 1)
            tz_grid = init_tz + sigma_tz * (2 * np.random.random(n) - 1)

            samples = np.c_[tx_grid, ty_grid, tz_grid]

            t = time.time()

            # empirical speedup 7x
            # parallel
            if parallel:
                pool = Pool(processes=12)
                scores = pool.map(lambda (tx, ty, tz): self.compute_score(np.r_[1, 0, 0, tx, 0, 1, 0, ty, 0, 0, 1, tz], indices_m=indices_m), samples)
                pool.close()
                pool.join()
            else:
            # serial
                scores = [self.compute_score(np.r_[1, 0, 0, tx, 0, 1, 0, ty, 0, 0, 1, tz], indices_m=indices_m)
                            for tx, ty, tz in samples]

            sys.stderr.write('grid search: %f seconds\n' % (time.time() - t)) # ~23s

            score_best = np.max(scores)

            tx_best, ty_best, tz_best = samples[np.argmax(scores)]

            if score_best > score_best_upToNow:
                # self.logger.info('%f %f', score_best_upToNow, score_best)

                score_best_upToNow = score_best
                params_best_upToNow = tx_best, ty_best, tz_best

                # self.logger.info('%f %f %f', tx_best, ty_best, tz_best)

        return params_best_upToNow


    def optimize(self, type='rigid', init_T=None, label_weights=None, \
                grid_search_iteration_number=0, grid_search_sample_number=1000,
                grad_computation_sample_number=10000,
                max_iter_num=1000, history_len=200, terminate_thresh=.1, \
                indices_m=None):
        """Optimize"""

        if type == 'rigid':
            grad_historical = np.zeros((6,))
        elif type == 'affine':
            grad_historical = np.zeros((12,))
        else:
            raise Exception('Type must be either rigid or affine.')

        if indices_m is None:
            indices_m = self.all_indices_m

        if grid_search_iteration_number > 0:
            tx_best, ty_best, tz_best = self.grid_search(grid_search_iteration_number, indices_m=indices_m, init_n=grid_search_sample_number)
            T = np.r_[1,0,0, tx_best, 0,1,0, ty_best, 0,0,1, tz_best]
        elif init_T is None:
            T = np.r_[1,0,0,0,0,1,0,0,0,0,1,0]
        else:
            T = init_T

        score_best = 0
        scores = []

        for iteration in range(max_iter_num):

            # self.logger.info('iteration %d', iteration)

            if type == 'rigid':
                lr1, lr2 = (1., 1e-2)
                T, s, grad_historical = self.step_lie(T, lr=np.r_[lr1,lr1,lr1,lr2,lr2,lr2],
                    grad_historical=grad_historical, verbose=False, num_samples=grad_computation_sample_number,
                    indices_m=indices_m)
            elif type == 'affine':
                lr1, lr2 = (10., 1e-1)
                T, s, grad_historical = self.step_gd(T, lr=np.r_[lr2, lr2, lr2, lr1, lr2, lr2, lr2, lr1, lr2, lr2, lr2, lr1], \
                                                grad_historical=grad_historical,
                                                indices_m=indices_m)
            else:
                raise Exception('Type must be either rigid or affine.')

            # self.logger.info('score: %f', s)
            scores.append(s)

            if iteration > 2*history_len:
                if np.abs(np.mean(scores[iteration-history_len:iteration]) - \
                          np.mean(scores[iteration-2*history_len:iteration-history_len])) < terminate_thresh:
                    break

            if s > score_best:
                best_gradient_descent_params = T
                score_best = s

        return best_gradient_descent_params, scores

    def step_lie(self, T, lr, grad_historical, verbose=False, num_samples=1000, indices_m=None):
        """
        One optimization step over Lie group SE(3).

        Args:
            T ((12,) vector): flattened vector of 3x4 transform matrix
            lr ((12,) vector): learning rate
            grad_historical ((12,) vector): accumulated gradiant magnitude, for Adagrad

        Returns:
            (tuple): tuple containing:

                new_T ((12,) vector): the new parameters
                score (float): current score
                grad_historical ((12,) vector): new accumulated gradient magnitude, used for Adagrad
        """

        if indices_m is None:
            indices_m = self.all_indices_m

        score, grad = self.compute_score_and_gradient(T, num_samples=num_samples, wrt_v=True, indices_m=indices_m)

        grad_historical += grad**2
        grad_adjusted = grad / (1e-10 + np.sqrt(grad_historical))
        v_opt = lr * grad_adjusted # no minus sign because maximizing

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

        return np.column_stack([R_new, t_new]).flatten(), score, grad_historical

    def step_gd(self, T, lr, dMdA_historical, surround=False, surround_weight=2., num_samples=None, indices_m=None):
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

        grad_historical += grad**2
        grad_adjusted = grad / (1e-10 + np.sqrt(grad_historical))

        new_T = T + lr*grad_adjusted

        return new_T, score, grad_historical



class Aligner(object):
    def __init__(self, volume_f, volume_m=None, nzvoxels_m=None, centroid_f=None, centroid_m=None, \
                label_weights=None, labelIndexMap_m2f=None, considered_indices_m=None):
        """
        Find the optimal transform of volume2 that aligns it with volume1.

        Args:
            volume_f (dict of 3d float array): the fixed volume(s) - subjects' score volumes, a probabilistic volume.
            volume_m (3d integer array): the moving volume - the atlas, an annotation volume. alternative is to provide `nzvoxels_m`.
            nzvoxels_m ((n,3) array): indices of active voxels in the moving volume
        """

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.volume_f = volume_f
        self.volume_m = volume_m

        assert volume_f is not None, 'Template volume is not specified.'
        assert volume_m is not None or nzvoxels_m is not None, 'Moving volume is not specified.'

        self.ydim_m, self.xdim_m, self.zdim_m = volume_m.shape
        self.ydim_f, self.xdim_f, self.zdim_f = volume_f.values()[0].shape

        self.labelIndexMap_m2f = labelIndexMap_m2f

        self.all_indices_m = list(set(self.labelIndexMap_m2f.keys()) & set(np.unique(volume_m)))

        # if considered_indices_m is None:
        #     self.considered_indices_m = all_indices_m
        # else:
        #     self.considered_indices_m = considered_indices_m

        # self.considered_indices_f = [self.labelIndexMap_m2f[ind_m] for ind_m in self.considered_indices_m]

        if nzvoxels_m is None:
            self.nzvoxels_m = Parallel(n_jobs=16)(delayed(parallel_where)(volume_m, i, num_samples=int(1e5))
                                        for i in self.all_indices_m)
            self.nzvoxels_m = dict(zip(self.all_indices_m, self.nzvoxels_m))
        else:
            self.nzvoxels_m = nzvoxels_m

        # self.set_centroid(centroid_m=centroid_m, centroid_f=centroid_f, indices_m=self.considered_indices_m)

    def set_centroid(self, centroid_m=None, centroid_f=None, indices_m=None):

        if indices_m is None:
            indices_m = self.all_indices_m

        if isinstance(centroid_m, basestring):
            if centroid_m == 'structure_centroid':
                self.centroid_m = np.concatenate([self.nzvoxels_m[i] for i in indices_m]).mean(axis=0)
            elif centroid_m == 'volume_centroid':
                self.centroid_m = np.r_[.5*self.xdim_m, .5*self.ydim_m, .5*self.zdim_m]
            else:
                raise Exception('centroid_m not recognized.')

        if isinstance(centroid_f, basestring):
            if centroid_f == 'centroid_m':
                self.centroid_f = self.centroid_m
            elif centroid_f == 'volume_centroid':
                self.centroid_f = np.r_[.5*self.xdim_f, .5*self.ydim_f, .5*self.zdim_f]
            else:
                raise Exception('centroid_f not recognized.')

        self.nzvoxels_centered_m = {ind_m: nzvs - self.centroid_m for ind_m, nzvs in self.nzvoxels_m.iteritems()}


    def load_gradient(self, gradient_filepath_map_f, indices_f=None):
        """Load gradients.

        Args:
            gradient_filepath_map_f (dict of str): path string that contains formatting parts and (suffix).
        """

        if indices_f is None:
            indices_f = set([self.labelIndexMap_m2f[ind_m] for ind_m in self.all_indices_m])

        self.grad_f = {ind_f: np.empty((3, self.ydim_f, self.xdim_f, self.zdim_f), dtype=np.float16) for ind_f in indices_f}

        t1 = time.time()

        for ind_f in indices_f:

            t = time.time()

            assert gradient_filepath_map_f is not None
            self.grad_f[ind_f][0] = bp.unpack_ndarray_file(gradient_filepath_map_f[ind_f] % {'suffix': 'gx'})
            self.grad_f[ind_f][1] = bp.unpack_ndarray_file(gradient_filepath_map_f[ind_f] % {'suffix': 'gy'})
            self.grad_f[ind_f][2] = bp.unpack_ndarray_file(gradient_filepath_map_f[ind_f] % {'suffix': 'gz'})

            sys.stderr.write('load gradient %s: %f seconds\n' % (ind_f, time.time() - t)) # ~6s

        sys.stderr.write('overall: %f seconds\n' % (time.time() - t1)) # ~100s

    def get_valid_voxels_after_transform(self, T, ind_m, return_valid):

        pts_prime = transform_points(np.array(T), pts_centered=self.nzvoxels_centered_m[ind_m], c_prime=self.centroid_f).astype(np.int16)
        xs_prime, ys_prime, zs_prime = pts_prime.T
        valid = (xs_prime >= 0) & (ys_prime >= 0) & (zs_prime >= 0) & \
                (xs_prime < self.xdim_f) & (ys_prime < self.ydim_f) & (zs_prime < self.zdim_f)

        if np.any(valid):
            xs_prime_valid = xs_prime[valid]
            ys_prime_valid = ys_prime[valid]
            zs_prime_valid = zs_prime[valid]

            if return_valid:
                return xs_prime_valid, ys_prime_valid, zs_prime_valid, valid
            else:
                return xs_prime_valid, ys_prime_valid, zs_prime_valid

    def compute_score_and_gradient(self, T, num_samples=None, wrt_v=False, indices_m=None):
        """
        Compute score and gradient.

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

        for ind_m in indices_m:

            s, xs_prime_valid, ys_prime_valid, zs_prime_valid, valid = self.compute_score_one(T, ind_m, return_valid=True)

            # score += label_weights[name] * voxel_probs_valid.sum()
            score += s

            ind_f = self.labelIndexMap_m2f[ind_m]
            Sx = self.grad_f[ind_f][0, ys_prime_valid, xs_prime_valid, zs_prime_valid]
            Sy = self.grad_f[ind_f][1, ys_prime_valid, xs_prime_valid, zs_prime_valid]
            Sz = self.grad_f[ind_f][2, ys_prime_valid, xs_prime_valid, zs_prime_valid]
            dxs, dys, dzs = self.nzvoxels_centered_m[ind_m][valid].T

            xs_prime_valid = xs_prime_valid.astype(np.float)
            ys_prime_valid = ys_prime_valid.astype(np.float)
            zs_prime_valid = zs_prime_valid.astype(np.float)

            if num_samples is not None:
                n = np.count_nonzero(valid)
                ii = np.random.choice(range(n), min(int(num_samples), n), replace=False)
                Sx = Sx[ii]
                Sy = Sy[ii]
                Sz = Sz[ii]
                xs_prime_valid = xs_prime_valid[ii]
                ys_prime_valid = ys_prime_valid[ii]
                zs_prime_valid = zs_prime_valid[ii]
                dxs = dxs[ii]
                dys = dys[ii]
                dzs = dzs[ii]

            if wrt_v:
                q = np.c_[Sx, Sy, Sz,
                -Sy*zs_prime_valid + Sz*ys_prime_valid,
                Sx*zs_prime_valid - Sz*xs_prime_valid,
                -Sx*ys_prime_valid + Sy*xs_prime_valid]
            else:
                q = np.c_[Sx*dxs, Sx*dys, Sx*dzs, Sx, Sy*dxs, Sy*dys, Sy*dzs, Sy, Sz*dxs, Sz*dys, Sz*dzs, Sz]

            # dMdA += label_weights[name] * q.sum(axis=0)
            grad += q.sum(axis=0)

            del q, Sx, Sy, Sz, dxs, dys, dzs, xs_prime_valid, ys_prime_valid, zs_prime_valid

        return score, grad

    def compute_score_one(self, T, ind_m, return_valid=False):
        """
        Compute score for one label.

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

        res = self.get_valid_voxels_after_transform(T, ind_m, return_valid)

        if res is None:
            return 0

        if return_valid:
            xs_prime_valid, ys_prime_valid, zs_prime_valid, valid = res
        else:
            xs_prime_valid, ys_prime_valid, zs_prime_valid = res

        ind_f = self.labelIndexMap_m2f[ind_m]
        voxel_probs_valid = self.volume_f[ind_f][ys_prime_valid, xs_prime_valid, zs_prime_valid] / 1e6
        s = voxel_probs_valid.sum()

        if return_valid:
            return s, xs_prime_valid, ys_prime_valid, zs_prime_valid, valid
        else:
            return s

    def compute_score(self, T, indices_m=None):
        """Compute score."""

        if indices_m is None:
            indices_m = self.all_indices_m

        score = 0
        for ind_m in indices_m:
            s = self.compute_score_one(T, ind_m, return_valid=False)
            score += s
        return score


    def compute_scores_neighborhood_grid(self, params, dxs, dys, dzs, indices_m=None):

        from itertools import product

        # parallelism not working yet, unless put large instance members in global variable

        scores = np.reshape([self.compute_score(params + (0.,0.,0., dx, 0.,0.,0., dy, 0.,0.,0., dz), indices_m=indices_m)
                            for dx, dy, dz in product(dxs, dys, dzs)],
                            (dxs.size, dys.size, dzs.size))

        # scores = np.reshape(Parallel(n_jobs=12)(delayed(compute_score)(params + (0.,0.,0., dx, 0.,0.,0., dy, 0.,0.,0., dz))
        #                                         for dx, dy, dz in product(dxs, dys, dzs)),
        #                     (dxs.size, dys.size, dzs.size))

        return scores

    def compute_scores_neighborhood_random(self, params, n, stds, indices_m=None):

        dparams = np.random.uniform(-1., 1., (n, len(stds))) * stds
        scores = [self.compute_score(params + dp, indices_m=indices_m) for dp in dparams]
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

    def grid_search(self, grid_search_iteration_number, indices_m=None, init_n=1000):
        """Grid search.

        Args:
            grid_search_iteration_number (int): number of iteration

        Returns:
            params_best_upToNow ((12,) float array): found parameters

        """
        params_best_upToNow = (0, 0, 0)
        score_best_upToNow = 0

        if indices_m is None:
            indices_m = self.all_indices_m

        for iteration in range(grid_search_iteration_number):

            self.logger.info('grid search iteration %d', iteration)

            init_tx, init_ty, init_tz  = params_best_upToNow

            n = int(init_n*np.exp(-iteration/3.))

            sigma_tx = 300*np.exp(-iteration/3.)
            sigma_ty = 300*np.exp(-iteration/3.)
            sigma_tz = 100*np.exp(-iteration/3.)

            tx_grid = init_tx + sigma_tx * (2 * np.random.random(n) - 1)
            ty_grid = init_ty + sigma_ty * (2 * np.random.random(n) - 1)
            tz_grid = init_tz + sigma_tz * (2 * np.random.random(n) - 1)

            samples = np.c_[tx_grid, ty_grid, tz_grid]

            t = time.time()

            scores = [self.compute_score(np.r_[1, 0, 0, tx, 0, 1, 0, ty, 0, 0, 1, tz], indices_m=indices_m)
                        for tx, ty, tz in samples]

            sys.stderr.write('grid search: %f seconds\n' % (time.time() - t)) # ~23s

            score_best = np.max(scores)

            tx_best, ty_best, tz_best = samples[np.argmax(scores)]

            if score_best > score_best_upToNow:
                self.logger.info('%f %f', score_best_upToNow, score_best)

                score_best_upToNow = score_best
                params_best_upToNow = tx_best, ty_best, tz_best

                self.logger.info('%f %f %f', tx_best, ty_best, tz_best)

        return params_best_upToNow


    def optimize(self, type='rigid', init_T=None, label_weights=None, \
                grid_search_iteration_number=0, max_iter_num=1000, history_len=200, terminate_thresh=.1, \
                indices_m=None):
        """Optimize"""

        if type == 'rigid':
            grad_historical = np.zeros((6,))
        elif type == 'affine':
            grad_historical = np.zeros((12,))
        else:
            raise Exception('Type must be either rigid or affine.')

        if indices_m is None:
            indices_m = self.all_indices_m

        if grid_search_iteration_number > 0:
            tx_best, ty_best, tz_best = self.grid_search(grid_search_iteration_number, indices_m=indices_m)
            T = np.r_[1,0,0, tx_best, 0,1,0, ty_best, 0,0,1, tz_best]
        elif init_T is None:
            T = np.r_[1,0,0,0,0,1,0,0,0,0,1,0]
        else:
            T = init_T

        score_best = 0
        scores = []

        for iteration in range(max_iter_num):

            self.logger.info('iteration %d', iteration)

            if type == 'rigid':
                lr1, lr2 = (1., 1e-2)
                T, s, grad_historical = self.step_lie(T, lr=np.r_[lr1,lr1,lr1,lr2,lr2,lr2],
                    grad_historical=grad_historical, verbose=False, num_samples=10000,
                    indices_m=indices_m)
            elif type == 'affine':
                lr1, lr2 = (10., 1e-1)
                T, s, grad_historical = self.step_gd(T, lr=np.r_[lr2, lr2, lr2, lr1, lr2, lr2, lr2, lr1, lr2, lr2, lr2, lr1], \
                                                grad_historical=grad_historical,
                                                indices_m=indices_m)
            else:
                raise Exception('Type must be either rigid or affine.')

            self.logger.info('score: %f', s)
            scores.append(s)

            if iteration > 2*history_len:
                if np.abs(np.mean(scores[iteration-history_len:iteration]) - \
                          np.mean(scores[iteration-2*history_len:iteration-history_len])) < terminate_thresh:
                    break

            if s > score_best:
                best_gradient_descent_params = T
                score_best = s

        return best_gradient_descent_params, scores

    def step_lie(self, T, lr, grad_historical, verbose=False, num_samples=1000, indices_m=None):
        """
        One optimization step over Lie group SE(3).

        Args:
            T ((12,) vector): flattened vector of 3x4 transform matrix
            lr ((12,) vector): learning rate
            grad_historical ((12,) vector): accumulated gradiant magnitude, for Adagrad

        Returns:
            (tuple): tuple containing:

                new_T ((12,) vector): the new parameters
                score (float): current score
                grad_historical ((12,) vector): new accumulated gradient magnitude, used for Adagrad
        """

        if indices_m is None:
            indices_m = self.all_indices_m

        score, grad = self.compute_score_and_gradient(T, num_samples=num_samples, wrt_v=True, indices_m=indices_m)

        grad_historical += grad**2
        grad_adjusted = grad / (1e-10 + np.sqrt(grad_historical))
        v_opt = lr * grad_adjusted # no minus sign because maximizing

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

        return np.column_stack([R_new, t_new]).flatten(), score, grad_historical

    def step_gd(self, T, lr, dMdA_historical, surround=False, surround_weight=2., num_samples=None, indices_m=None):
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

        grad_historical += grad**2
        grad_adjusted = grad / (1e-10 + np.sqrt(grad_historical))

        new_T = T + lr*grad_adjusted

        return new_T, score, grad_historical


############################################################

# def align(volume_f, volume_m=None, type='rigid', nzvoxels_m=None, centroid_f=None, centroid_m=None, \
#         grad_f=None, grad_m=None, init_T=None, label_weights=None, gradient_filepath_map_f=None, labelIndexMap_m2f=None, \
#         grid_search_iteration_number=0, max_iter_num=1000, history_len=200, terminate_thresh=.1):
#     """Find the optimal transform of volume2 that aligns it with volume1.
#
#     Args:
#         volume_f (dict of 3d float array): the fixed volume(s) - subjects' score volumes, a probabilistic volume.
#         volume_m (3d integer array): the moving volume - the atlas, an annotation volume. alternative is to provide `nzvoxels_m`.
#         nzvoxels_m ((n,3) array): indices of active voxels in the moving volume
#         gradient_filepath_map_f (dict of str): path string that contains formatting parts and (suffix).
#
#     Returns:
#         (tuple): tuple containing:
#
#             - T ((12,) vector): optimal transform parameters
#             - scores (float array): score history
#     """
#
#     import time
#
#     import logging
#     logger = logging.getLogger(__name__)
#     logger.setLevel(logging.INFO)
#
#     assert volume_f is not None, 'Template volume is not specified.'
#     assert volume_m is not None or nzvoxels_m is not None, 'Moving volume is not specified.'
#
#     ydim_m, xdim_m, zdim_m = volume_m.shape
#     ydim_f, xdim_f, zdim_f = volume_f.values()[0].shape
#
#     considered_indices_m = set(labelIndexMap_m2f.keys()) & set(np.unique(volume_m))
#
#     if nzvoxels_m is None:
#         nzvoxels_m = Parallel(n_jobs=16)(delayed(parallel_where)(volume_m, i, num_samples=int(1e5))
#                                     for i in considered_indices_m)
#         nzvoxels_m = dict(zip(considered_indices_m, nzvoxels_m))
#
#     if isinstance(centroid_m, basestring):
#         if centroid_m == 'structure_centroid':
#             centroid_m = np.concatenate([nzvoxels_m[i] for i in considered_indices_m]).mean(axis=0)
#         elif centroid_m == 'volume_centroid':
#             centroid_m = np.r_[.5*xdim_m, .5*ydim_m, .5*zdim_m]
#         else:
#             raise Exception('centroid_m not recognized.')
#
#     if isinstance(centroid_f, basestring):
#         if centroid_f == 'centroid_m':
#             centroid_f = centroid_m
#         elif centroid_f == 'volume_centroid':
#             centroid_f = np.r_[.5*xdim_f, .5*ydim_f, .5*zdim_f]
#         else:
#             raise Exception('centroid_f not recognized.')
#
#     nzvoxels_centered_m = {ind_m: nzvoxels_m[ind_m] - centroid_m for ind_m in labelIndexMap_m2f.iterkeys()}
#
#     considered_indices_f = set(labelIndexMap_m2f.values())
#
#     # if label_weights is None:
#     #     label_weights = {name: .1 if name == 'outerContour' else 1. for name in available_labels_unsided}
#
#     # Load Gradients
#     if grad_f is None:
#
#         grad_f = {ind_f: np.empty((3, ydim_f, xdim_f, zdim_f), dtype=np.float16) for ind_f in considered_indices_f}
#
#         t1 = time.time()
#
#         for ind_f in considered_indices_f:
#
#             t = time.time()
#
#             assert gradient_filepath_map_f is not None
#             grad_f[ind_f][0] = bp.unpack_ndarray_file(gradient_filepath_map_f[ind_f] % {'suffix': 'gx'})
#             grad_f[ind_f][1] = bp.unpack_ndarray_file(gradient_filepath_map_f[ind_f] % {'suffix': 'gy'})
#             grad_f[ind_f][2] = bp.unpack_ndarray_file(gradient_filepath_map_f[ind_f] % {'suffix': 'gz'})
#
#             sys.stderr.write('load gradient %s: %f seconds\n' % (ind_f, time.time() - t)) # ~6s
#
#         sys.stderr.write('overall: %f seconds\n' % (time.time() - t1)) # ~100s
#
#
#     def get_valid_voxels_after_transform(T, ind_m, return_valid):
#
#         pts_prime = transform_points(np.array(T), pts_centered=nzvoxels_centered_m[ind_m], c_prime=centroid_f).astype(np.int16)
#         xs_prime, ys_prime, zs_prime = pts_prime.T
#         valid = (xs_prime >= 0) & (ys_prime >= 0) & (zs_prime >= 0) & \
#                 (xs_prime < xdim_f) & (ys_prime < ydim_f) & (zs_prime < zdim_f)
#
#         if np.any(valid):
#             xs_prime_valid = xs_prime[valid]
#             ys_prime_valid = ys_prime[valid]
#             zs_prime_valid = zs_prime[valid]
#
#             if return_valid:
#                 return xs_prime_valid, ys_prime_valid, zs_prime_valid, valid
#             else:
#                 return xs_prime_valid, ys_prime_valid, zs_prime_valid
#
#
#     def compute_score_and_gradient(T, num_samples=None, wrt_v=False):
#         """
#         Compute score and gradient, based on `considered_indices_m`.
#
#         Args:
#             T ((12,) vector): transform parameters
#             num_samples (int): Number of sample points to compute gradient.
#
#         Returns:
#             (tuple): tuple containing:
#
#             - score (int): score
#             - grad (float): gradient
#         """
#
#         score = 0
#
#         if wrt_v:
#             grad = np.zeros((6,))
#         else:
#             grad = np.zeros((12,))
#
#         for ind_m in considered_indices_m:
#
#             s, xs_prime_valid, ys_prime_valid, zs_prime_valid, valid = compute_score_one(T, ind_m, return_valid=True)
#
#             # score += label_weights[name] * voxel_probs_valid.sum()
#             score += s
#
#             ind_f = labelIndexMap_m2f[ind_m]
#             Sx = grad_f[ind_f][0, ys_prime_valid, xs_prime_valid, zs_prime_valid]
#             Sy = grad_f[ind_f][1, ys_prime_valid, xs_prime_valid, zs_prime_valid]
#             Sz = grad_f[ind_f][2, ys_prime_valid, xs_prime_valid, zs_prime_valid]
#             dxs, dys, dzs = nzvoxels_centered_m[ind_m][valid].T
#
#             xs_prime_valid = xs_prime_valid.astype(np.float)
#             ys_prime_valid = ys_prime_valid.astype(np.float)
#             zs_prime_valid = zs_prime_valid.astype(np.float)
#
#             if num_samples is not None:
#                 n = np.count_nonzero(valid)
#                 ii = np.random.choice(range(n), min(int(num_samples), n), replace=False)
#                 Sx = Sx[ii]
#                 Sy = Sy[ii]
#                 Sz = Sz[ii]
#                 xs_prime_valid = xs_prime_valid[ii]
#                 ys_prime_valid = ys_prime_valid[ii]
#                 zs_prime_valid = zs_prime_valid[ii]
#                 dxs = dxs[ii]
#                 dys = dys[ii]
#                 dzs = dzs[ii]
#
#             if wrt_v:
#                 q = np.c_[Sx, Sy, Sz,
#                 -Sy*zs_prime_valid + Sz*ys_prime_valid,
#                 Sx*zs_prime_valid - Sz*xs_prime_valid,
#                 -Sx*ys_prime_valid + Sy*xs_prime_valid]
#             else:
#                 q = np.c_[Sx*dxs, Sx*dys, Sx*dzs, Sx, Sy*dxs, Sy*dys, Sy*dzs, Sy, Sz*dxs, Sz*dys, Sz*dzs, Sz]
#
#             # dMdA += label_weights[name] * q.sum(axis=0)
#             grad += q.sum(axis=0)
#
#             del q, Sx, Sy, Sz, dxs, dys, dzs, xs_prime_valid, ys_prime_valid, zs_prime_valid
#
#         return score, grad
#
#
#     def compute_score_one(T, ind_m, return_valid=False):
#         """
#         Compute score for one label.
#
#         Args:
#             T ((12,) vector): transform parameters
#             ind_m (int): label on the moving volume
#             return_valid (bool): whether to return valid voxels
#
#         Returns:
#             (float or tuple): if `return_valid` is true, return a tuple containing:
#
#             - score (int): score
#             - xs_prime_valid (array):
#             - ys_prime_valid (array):
#             - zs_prime_valid (array):
#             - valid (boolean array):
#         """
#
#         res = get_valid_voxels_after_transform(T, ind_m, return_valid)
#
#         if res is None:
#             return 0
#
#         if return_valid:
#             xs_prime_valid, ys_prime_valid, zs_prime_valid, valid = res
#         else:
#             xs_prime_valid, ys_prime_valid, zs_prime_valid = res
#
#         ind_f = labelIndexMap_m2f[ind_m]
#         voxel_probs_valid = volume_f[ind_f][ys_prime_valid, xs_prime_valid, zs_prime_valid] / 1e6
#         s = voxel_probs_valid.sum()
#
#         if return_valid:
#             return s, xs_prime_valid, ys_prime_valid, zs_prime_valid, valid
#         else:
#             return s
#
#     def compute_score(T, indices_m=considered_indices_m):
#         """Compute score."""
#
#         score = 0
#         for ind_m in indices_m:
#             s = compute_score_one(T, ind_m, return_valid=False)
#             score += s
#         return score
#
#
#     # Grid search
#
#     params_best_upToNow = (0, 0, 0)
#     score_best_upToNow = 0
#
#     init_n = 1000
#
#     for iteration in range(grid_search_iteration_number):
#
#         logger.info('grid search iteration %d', iteration)
#
#         init_tx, init_ty, init_tz  = params_best_upToNow
#
#         n = int(init_n*np.exp(-iteration/3.))
#
#         sigma_tx = 300*np.exp(-iteration/3.)
#         sigma_ty = 300*np.exp(-iteration/3.)
#         sigma_tz = 100*np.exp(-iteration/3.)
#
#         tx_grid = init_tx + sigma_tx * (2 * np.random.random(n) - 1)
#         ty_grid = init_ty + sigma_ty * (2 * np.random.random(n) - 1)
#         tz_grid = init_tz + sigma_tz * (2 * np.random.random(n) - 1)
#
#         samples = np.c_[tx_grid, ty_grid, tz_grid]
#
#         import time
#
#         t = time.time()
#         # scores = Parallel(n_jobs=8)(delayed(compute_score)(np.r_[1, 0, 0, tx, 0, 1, 0, ty, 0, 0, 1, tz])
#         #                             for tx, ty, tz in samples)
#
#         scores = [compute_score(np.r_[1, 0, 0, tx, 0, 1, 0, ty, 0, 0, 1, tz]) for tx, ty, tz in samples]
#
#         # scores = Parallel(n_jobs=8)(delayed(compute_score)(np.r_[1, 0, 0, tx, 0, 1, 0, ty, 0, 0, 1, tz])
#         #                             for tx, ty, tz in samples)
#
#         sys.stderr.write('grid search: %f seconds\n' % (time.time() - t)) # ~23s
#
#         score_best = np.max(scores)
#
#         tx_best, ty_best, tz_best = samples[np.argmax(scores)]
#
#         if score_best > score_best_upToNow:
#             logger.info('%f %f', score_best_upToNow, score_best)
#
#             score_best_upToNow = score_best
#             params_best_upToNow = tx_best, ty_best, tz_best
#
#             logger.info('%f %f %f', tx_best, ty_best, tz_best)
#
#         logger.info('\n')
#
#     # Gradient descent
#
#     def step_gd(T, lr, dMdA_historical, surround=False, surround_weight=2., num_samples=None):
#         """
#         One optimization step using gradient descent with Adagrad.
#
#         Args:
#             T ((12,) vector): flattened vector of 3x4 transform matrix.
#             lr ((12,) vector): learning rate
#             dMdA_historical ((12,) vector): accumulated gradiant magnitude, for Adagrad
#
#         Returns:
#             (tuple): tuple containing:
#
#                 new_T ((12,) vector): the new parameters
#                 score (float): current score
#                 dMdv_historical ((12,) vector): new accumulated gradient magnitude, used for Adagrad
#
#         """
#
#         score, grad = compute_score_and_gradient(T, num_samples=num_samples)
#
#         # if surround:
#         #     s_surr, dMdA_surr = compute_score_and_gradient(T, name, surround=True, num_samples=num_samples)
#         #     dMdA -= surround_weight * dMdA_surr
#         #     score -= surround_weight * s_surr
#
#         grad_historical += grad**2
#         grad_adjusted = grad / (1e-10 + np.sqrt(grad_historical))
#
#         new_T = T + lr*grad_adjusted
#
#         return new_T, score, grad_historical
#
#
#     from lie import matrix_exp_v
#     def step_lie(T, lr, grad_historical, verbose=False, num_samples=1000):
#         """
#         One optimization step over Lie group SE(3).
#
#         Args:
#             T ((12,) vector): flattened vector of 3x4 transform matrix
#             lr ((12,) vector): learning rate
#             grad_historical ((12,) vector): accumulated gradiant magnitude, for Adagrad
#
#         Returns:
#             (tuple): tuple containing:
#
#                 new_T ((12,) vector): the new parameters
#                 score (float): current score
#                 grad_historical ((12,) vector): new accumulated gradient magnitude, used for Adagrad
#         """
#
#         score, grad = compute_score_and_gradient(T, num_samples=num_samples, wrt_v=True)
#
#         grad_historical += grad**2
#         grad_adjusted = grad / (1e-10 + np.sqrt(grad_historical))
#         v_opt = lr * grad_adjusted # no minus sign because maximizing
#
#         if verbose:
#             print 'v_opt:', v_opt
#
#         theta = np.sqrt(np.sum(v_opt[3:]**2))
#         if verbose:
#             print 'theta:', theta
#         assert theta < np.pi
#
#         exp_w, Vt = matrix_exp_v(v_opt)
#
#         if verbose:
#             print 'Vt:' , Vt
#
#         Tm = np.reshape(T, (3,4))
#         t = Tm[:, 3]
#         R = Tm[:, :3]
#
#         R_new = np.dot(exp_w, R)
#         t_new = np.dot(exp_w, t) + Vt
#
#         if verbose:
#             print '\n'
#
#         return np.column_stack([R_new, t_new]).flatten(), score, grad_historical
#
#
#     #########################################################
#
#     if type == 'rigid':
#         grad_historical = np.zeros((6,))
#     elif type == 'affine':
#         grad_historical = np.zeros((12,))
#     else:
#         raise Exception('Type must be either rigid or affine.')
#
#     if grid_search_iteration_number > 0:
#         tx_best, ty_best, tz_best = params_best_upToNow
#         T = np.r_[1,0,0, tx_best, 0,1,0, ty_best, 0,0,1, tz_best]
#     elif init_T is None:
#         T = np.r_[1,0,0,0,0,1,0,0,0,0,1,0]
#     else:
#         T = init_T
#
#     score_best = 0
#     scores = []
#
#     for iteration in range(max_iter_num):
#
#         logger.info('iteration %d', iteration)
#
#         if type == 'rigid':
#             lr1, lr2 = (1., 1e-2)
#             T, s, grad_historical = step_lie(T, lr=np.r_[lr1,lr1,lr1,lr2,lr2,lr2],
#                 grad_historical=grad_historical, verbose=False, num_samples=10000)
#         elif type == 'affine':
#             lr1, lr2 = (10., 1e-1)
#             T, s, grad_historical = step_gd(T, lr=np.r_[lr2, lr2, lr2, lr1, lr2, lr2, lr2, lr1, lr2, lr2, lr2, lr1], \
#                                             grad_historical=grad_historical)
#         else:
#             raise Exception('Type must be either rigid or affine.')
#
#         logger.info('score: %f', s)
#         scores.append(s)
#
#         if iteration > 2*history_len:
#             if np.abs(np.mean(scores[iteration-history_len:iteration]) - \
#                       np.mean(scores[iteration-2*history_len:iteration-history_len])) < terminate_thresh:
#                 break
#
#         if s > score_best:
#             best_gradient_descent_params = T
#             score_best = s
#
#     return best_gradient_descent_params, scores

###########################################################################

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


def find_contour_points(labelmap, sample_every=10):
    """
    The value of sample_every can be interpreted as distance between points.
    return is (x,y)
    Note the returned result is a dict of lists.
    """

    regions = regionprops(labelmap)

    contour_points = {}

    for r in regions:

        (min_row, min_col, max_row, max_col) = r.bbox

        padded = np.pad(r.filled_image, ((5,5),(5,5)), mode='constant', constant_values=0)

        contours = find_contours(padded, .5, fully_connected='high')
        contours = [cnt.astype(np.int) for cnt in contours if len(cnt) > 10]
        if len(contours) > 0:
#             if len(contours) > 1:
#                 sys.stderr.write('%d: region has more than one part\n' % r.label)

            contours = sorted(contours, key=lambda c: len(c), reverse=True)
            contours_list = [c-(5,5) for c in contours]
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
    t = Tm[:, 3]
    A = Tm[:, :3]

    pts_prime = np.dot(A, pts_centered.T) + (t + c_prime)[:,None]

    return pts_prime.T

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
