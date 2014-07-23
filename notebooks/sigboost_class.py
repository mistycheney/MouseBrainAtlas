import numpy as np
import cv2

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import random
import itertools
import sys
import os
from multiprocessing import Pool
import json
import glob
import re
import subprocess
import argparse
import pprint

from skimage.segmentation import slic, mark_boundaries
from skimage.measure import regionprops
from skimage.util import img_as_ubyte
from skimage.color import hsv2rgb, label2rgb, gray2rgb
from skimage.morphology import disk
from skimage.filter.rank import gradient
from skimage.filter import gabor_kernel
from skimage.transform import rescale, resize

from scipy.ndimage import gaussian_filter, measurements
from scipy.sparse import coo_matrix
from scipy.spatial.distance import pdist, squareform, euclidean, cdist
from scipy.signal import fftconvolve

from IPython.display import FileLink, Image, FileLinks

import utilities

from joblib import Parallel, delayed


class sigboost(object):

    def __init__(self, img, segmentation, sp_texton_hist_normalized, sp_texton_hist_normalized, labeling,
    					bg_superpixels, neighbors, result_dir, param):
    	self.img = img
    	self.segmentation = segmentation
    	self.labeling = labeling

	    self.D_texton_null = np.squeeze(
	    	cdist(sp_texton_hist_normalized, [overall_texton_hist_normalized], chi2))
	    self.D_dir_null = np.squeeze(
	        cdist(sp_dir_hist_normalized, [overall_dir_hist_normalized], chi2))
	    self.p = sp_texton_hist_normalized
	    self.q = sp_dir_hist_normalized

	    self.bg_superpixels = bg_superpixels

		self.result_dir = result_dir

	    self.n_models = param['n_models']
	    self.frontier_contrast_diff_thresh = param['frontier_contrast_diff_thresh']
	    self.lr_grow_thresh = param['lr_grow_thresh']
	    self.beta = param['beta']
		self.lr_decision_thresh = param['lr_decision_thresh']

	    self.re_thresh_min = 0.01
	    self.re_thresh_max = 0.8

        self.n_superpixels = segmentation.max() + 1

    def chi2(u, v):
        r = np.nansum((u - v) ** 2 / (u + v))
        return r

    def grow_cluster_relative_entropy(seed, debug=False,
                                      frontier_contrast_diff_thresh=0.1,
                                      max_cluster_size=100):
        '''
        find the connected cluster of superpixels that have similar texture, starting from a superpixel as seed
        '''

        bg_set = set(self.bg_superpixels.tolist())

        if seed in bg_set:
            return [], -1

        prev_frontier_contrast = np.inf
        for re_thresh in np.arange(self.re_thresh_min, self.re_thresh_max, .01):

            curr_cluster = set([seed])
            frontier = [seed]

            while len(frontier) > 0:
                u = frontier.pop(-1)
                for v in neighbors[u]:
                    if v in self.bg_superpixels or v in curr_cluster:
                        continue

                    if chi2(self.p[v], self.p[seed]) < re_thresh:
                        curr_cluster.add(v)
                        frontier.append(v)

            surround = set.union(
                *[self.neighbors[i] for i in curr_cluster]) - set.union(curr_cluster, bg_set)
            if len(surround) == 0:
                return curr_cluster, re_thresh

            frontier_in_cluster = set.intersection(
                set.union(*[self.neighbors[i] for i in surround]), curr_cluster)
            frontier_contrasts = [np.nanmax([chi2(p[i], p[j]) for j in self.neighbors[i] if j not in bg_set]) for i in frontier_in_cluster]
            frontier_contrast = np.max(frontier_contrasts)

            if len(curr_cluster) > max_cluster_size or \
                    frontier_contrast - prev_frontier_contrast > frontier_contrast_diff_thresh:
                return curr_cluster, re_thresh

            prev_frontier_contrast = frontier_contrast
            prev_cluster = curr_cluster
            prev_re_thresh = re_thresh

        return curr_cluster, re_thresh

    def grow_cluster_likelihood_ratio(seed, texton_model, dir_model, debug=False, lr_grow_thresh=0.1):
        '''
        find the connected cluster of superpixels that are more likely to be explained by given model than by null, starting from a superpixel as seed
        '''

        if seed in self.bg_superpixels:
            return [], -1

        curr_cluster = set([seed])
        frontier = [seed]

        while len(frontier) > 0:
            u = frontier.pop(-1)
            for v in self.neighbors[u]:
                if v in self.bg_superpixels or v in curr_cluster:
                    continue

                ratio_v = self.D_texton_null[v] - chi2(p[v], texton_model) +\
                    		self.D_dir_null[v] - chi2(q[v], dir_model)

                if ratio_v > lr_grow_thresh:
                    curr_cluster.add(v)
                    frontier.append(v)

        return curr_cluster, lr_grow_thresh

    def grow_cluster_likelihood_ratio_precomputed(seed, D_texton_model, D_dir_model, debug=False, lr_grow_thresh=0.1):
        '''
        find the connected cluster of superpixels that are more likely to be explained by given model than by null, starting from a superpixel as seed
        using pre-computed distances between model and superpixels
        '''

        if seed in self.bg_superpixels:
            return [], -1

        curr_cluster = set([seed])
        frontier = [seed]

        while len(frontier) > 0:
            u = frontier.pop(-1)
            for v in self.neighbors[u]:
                if v in self.bg_superpixels or v in curr_cluster:
                    continue

                ratio_v = self.D_texton_null[v] - D_texton_model[v] +\
                    self.D_dir_null[v] - D_dir_model[v]

                if ratio_v > lr_grow_thresh:
                    curr_cluster.add(v)
                    frontier.append(v)

        return curr_cluster, lr_grow_thresh

    def visualize_cluster(self, scores, cluster='all', title='', filename=None):
    	'''
    	Generate black and white image with the cluster of superpixels highlighted
    	'''
        vis = scores[self.segmentation]
        if cluster != 'all':
            cluster_selection = np.equal.outer(
                self.segmentation, cluster).any(axis=2)
            vis[~cluster_selection] = 0

        plt.matshow(vis, cmap=plt.cm.Greys_r)
        plt.axis('off')
        plt.title(title)
        if filename is not None:
            plt.savefig(
                os.path.join(self.result_dir, 'stages', filename + '.png'), bbox_inches='tight')
    #     plt.show()
        plt.close()

    def paint_cluster_on_img(self, cluster, title, filename=None):
    	'''
    	Highlight the cluster of superpixels on the real image
    	'''
        cluster_map = -1 * np.ones_like(self.segmentation)
        for s in cluster:
            cluster_map[self.segmentation == s] = 1
        vis = label2rgb(cluster_map, image=img)
        plt.imshow(vis, cmap=plt.cm.Greys_r)
        plt.axis('off')
        plt.title(title)
        if filename is not None:
            plt.savefig(
                os.path.join(self.result_dir, 'stages', filename + '.png'), bbox_inches='tight')
    #     plt.show()
        plt.close()

    def paint_clusters_on_img(self, clusters, title, filename=None):
    	'''
    	Highlight the clusters of superpixels on the real image
    	'''
        cluster_map = -1 * np.ones_like(self.segmentation)
        for i, cluster in enumerate(clusters):
            for j in cluster:
                cluster_map[segmentation == j] = i
        vis = label2rgb(cluster_map, image=img)
        plt.imshow(vis, cmap=plt.cm.Greys_r)
        plt.axis('off')
        plt.title(title)
        if filename is not None:
            plt.savefig(
                os.path.join(self.result_dir, 'stages', filename + '.png'), bbox_inches='tight')
    #     plt.show()
        plt.close()


    def boost(self):
        '''
        perform sigboost
        '''

        # create output directory
	    f = os.path.join(self.result_dir, 'stages')
	    if not os.path.exists(f):
	        os.makedirs(f)


        # compute RE-clusters of every superpixel
        r = Parallel(n_jobs=16)(delayed(grow_cluster_relative_entropy)(i, frontier_contrast_diff_thresh=self.frontier_contrast_diff_thresh)
                                for i in range(self.n_superpixels))
        clusters = [list(c) for c, t in r]
        print 'RE-clusters computed'


        # initialize models
	    self.texton_models = np.zeros((n_models, n_texton))
	    self.dir_models = np.zeros((n_models, n_angle))

	    self.seed_indices = np.zeros((n_models,))

	    weights = np.ones((self.n_superpixels, )) / self.n_superpixels
	    weights[self.bg_superpixels] = 0

        # begin boosting loop; learn one model at each iteration
	    for t in range(n_models):

	        print 'model %d' % (t)

            # Compute significance scores for every superpixel;
            # the significance score is defined as the average log likelihood ratio in a superpixel's RE-cluster
	        sig_score = np.zeros((self.n_superpixels, ))
	        for i in self.fg_superpixels:
	            cluster = clusters[i]
	            sig_score[i] = np.mean(weights[cluster] *
	                                   (D_texton_null[cluster] - np.array([chi2(p[j], p[i]) for j in cluster]) +
	                                    D_dir_null[cluster] - np.array([chi2(q[j], q[i]) for j in cluster])))

            
            # Pick the most significant superpixel
	        seed_sp = sig_score.argsort()[-1]
	        print "most significant superpixel", seed_sp

	        visualize_cluster(
	            sig_score, 'all', title='significance score for each superpixel', filename='sigscore%d' % t)

	        curr_cluster = clusters[seed_sp]
	        visualize_cluster(
	            sig_score, curr_cluster, title='cluster growed based on relative entropy', filename='re_cluster%d' % t)

            # models are the average of the distributions in the chosen superpixel's RE-cluster
	        model_texton = sp_texton_hist_normalized[curr_cluster].mean(axis=0)
	        model_dir = sp_dir_hist_normalized[curr_cluster].mean(axis=0)

            # Compute log likelihood ratio of this model against the null, for every superpixel

	        # RE(pj|pm)
	        D_texton_model = np.empty((self.n_superpixels,))
	        D_texton_model[self.fg_superpixels] = np.array(
	            [chi2(sp_texton_hist_normalized[i], model_texton) for i in self.fg_superpixels])
	        D_texton_model[self.bg_superpixels] = np.nan

	        # RE(qj|qm)
	        D_dir_model = np.empty((self.n_superpixels,))
	        D_dir_model[self.fg_superpixels] = np.array(
	            [chi2(sp_dir_hist_normalized[i], model_dir) for i in self.fg_superpixels])
	        D_dir_model[self.bg_superpixels] = np.nan

	        # RE(pj|p0)-RE(pj|pm) + RE(qj|q0)-RE(qj|qm)
	        match_scores = np.empty((self.n_superpixels,))
	        match_scores[self.fg_superpixels] = D_texton_null[self.fg_superpixels] - D_texton_model[self.fg_superpixels] +\
	            D_dir_model[self.fg_superpixels] - D_dir_model[self.fg_superpixels]
	        match_scores[self.bg_superpixels] = 0

	        visualize_cluster(
	            match_scores, 'all', title='match score', filename='match_score%d' % t)

            # Find the cluster growed from seed based on log likelihood ratio. Refer to this cluster as the LR-cluster
	        matched, _ = grow_cluster_likelihood_ratio_precomputed(seed_sp, D_texton_model, D_dir_model, lr_grow_thresh=self.lr_grow_thresh)
	        matched = list(matched)

	        visualize_cluster(
	            match_scores, matched, title='cluster growed based on likelihood ratio', filename='lr_cluster%d' % t)

            # Reduce the weights of superpixels in LR-cluster
	        weights[matched] = weights[matched] * np.exp(-5 * (D_texton_null[matched] - D_texton_model[matched] +
	                                                           D_dir_null[matched] - D_dir_model[matched]) ** self.beta)
	        weights[self.bg_superpixels] = 0
	        weights = weights / weights.sum()
	        visualize_cluster((weights - weights.min()) / (weights.max() - weights.min()), 'all',
	                          title='updated superpixel weights', filename='weight%d' % t)

	        labels = -1 * np.ones_like(self.segmentation)
	        for i in matched:
	            labels[self.segmentation == i] = 1
	        real_image = label2rgb(labels, img)
	        save_img(real_image, os.path.join('stage', 'real_image_model%d' % t))

            # record the model found at this round
	        self.seed_indices[t] = seed_sp
	        self.texton_models[t] = model_texton
	        self.dir_models[t] = model_dir


    def find_best_model_per_proc(self, i):
    	'''
		Worker function for finding the best models for every superpixel on the current image.
		Best model is the one with the highest likelihood ratio against the null distribution.
		'''
        model_score = np.empty((n_models, ))

        if i in self.bg_superpixels:
            return -1
        else:
            for m in range(n_models):
                matched, _ = grow_cluster_likelihood_ratio_precomputed(i, D_texton_model[m], D_dir_model[m],
                                                                       lr_grow_thresh=self.lr_grow_thresh)
                matched = list(matched)
                model_score[m] = np.mean(D_texton_null[matched] - D_texton_model[m, matched] +
                                         D_dir_null[matched] - D_dir_model[m, matched])

            best_sig = model_score.max()
            # sp whose sig is smaller than this is assigned null
            if best_sig > self.lr_decision_thresh:
                return model_score.argmax()
        return -1

	def apply_models_curr_img(self):
		'''
		Find the best models for every superpixel on the current image.
		Best model is the one with the highest likelihood ratio against the null distribution.
		'''

        # Compute the distances between every model and every superpixel
        D_texton_model = -1 * np.ones((n_models, self.n_superpixels))
        D_dir_model = -1 * np.ones((n_models, self.n_superpixels))
        D_texton_model[:, self.fg_superpixels] = cdist(
            sp_texton_hist_normalized[self.fg_superpixels], self.texton_models, chi2).T
        D_dir_model[:, self.fg_superpixels] = cdist(
            sp_dir_hist_normalized[self.fg_superpixels], self.dir_models, chi2).T

        # Compute the likelihood ratio for every model on every superpixel, and return the model with the highest ratio
	    best_model = Parallel(n_jobs=16)(delayed(self.find_best_model_per_proc)(i) for i in range(self.n_superpixels))
	    labels = np.array(best_model, dtype=np.int)
	    save_array(labels, 'labels')

	    labelmap = labels[self.segmentation]
	    save_array(labelmap, 'labelmap')

	    labelmap_rgb = label2rgb(labelmap.astype(np.int), image=img)
	    save_img(labelmap_rgb, 'labelmap')
