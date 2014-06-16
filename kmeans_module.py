# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from sklearn.cluster import MiniBatchKMeans 

from skimage.segmentation import slic, mark_boundaries
from skimage.measure import regionprops
from skimage.util import img_as_ubyte
from skimage.color import hsv2rgb, label2rgb, gray2rgb
from skimage.morphology import disk
from skimage.filter.rank import gradient

from scipy.sparse import coo_matrix
from scipy.spatial.distance import pdist, squareform

import numpy as np
import cv2
import matplotlib.pyplot as plt

from utilities import *

class SaliencyDetector(object):
    
    def __init__(self, img, mask, features_input):
        self.img = img
        self.features = np.load(features_input)
        self.n_feat = self.features.shape[-1]
                
        self.features = self.features[100:-100, 100:-100, :]
        self.img = self.img[100:-100, 100:-100]
        self.mask = mask[100:-100, 100:-100]
        
#         self.features = self.features.view(np.ma.MaskedArray)
#         self.features[~self.mask,:] = np.ma.masked        
#         self.img = self.img.view(np.ma.MaskedArray)
#         self.img[~self.mask,:] = np.ma.masked
    
    def visualize_features(self, output=None):
        fig, axes = plt.subplots(ncols=4, nrows=self.features.shape[-1]/4, figsize=(20,10))
        for i, ax in zip(range(self.features.shape[-1]), axes.flat):
            ax.matshow(self.features[...,i])
            ax.set_title('feature %d'%i, fontsize=0.5)
            ax.axis('off')
        plt.close();
        
        if output is not None:
            fig.savefig(output, bbox_inches='tight')
    
    @timeit
    def compute_texton(self, num_textons = 10):
        self.num_textons = num_textons
        kmeans_model = MiniBatchKMeans(num_textons)
        kmeans_model.fit(self.features[self.mask,:].reshape(-1, self.n_feat))
        self.textonmap = kmeans_model.predict(self.features.reshape(-1, self.n_feat)).reshape(self.features.shape[:2])
    
    @timeit
    def segment_superpixels(self, compactness=5, sigma=10):
        self.segmentation = slic(gray2rgb(self.img), compactness=compactness, sigma=sigma, enforce_connectivity=True)
        self.n_superpixels = len(np.unique(self.segmentation))

        sp_props = regionprops(self.segmentation+1, intensity_image=self.img, cache=True)
        self.sp_centroids = np.array([s.centroid for s in sp_props])
        sp_centroid_dist = pdist(self.sp_centroids)
        self.sp_centroid_dist_matrix = squareform(sp_centroid_dist)

        self.sp_mean_intensity = np.array([s.mean_intensity for s in sp_props])
        sp_areas = np.array([s.area for s in sp_props])
        
        superpixels_bg_count = np.array([(~self.mask[self.segmentation==i]).sum() for i in range(self.n_superpixels)])
        self.bg_superpixels = np.nonzero((superpixels_bg_count/sp_areas) > .8)[0]
        
#         superpixels_fg_count = np.array([self.mask[self.segmentation==i].sum() for i in range(self.n_superpixels)])
#         self.bg_superpixels = np.nonzero((superpixels_fg_count/sp_areas) < 0.3)[0]
    
    def visualize_superpixels(self, output=None):
        img_superpixelized = mark_boundaries(gray2rgb(self.img), self.segmentation)
        img_superpixelized_text = img_as_ubyte(img_superpixelized)
        for s in range(self.n_superpixels):
            img_superpixelized_text = cv2.putText(img_superpixelized_text, str(s), 
                                                  tuple(np.floor(self.sp_centroids[s][::-1]).astype(np.int)), 
                                                  cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                  0.8, ((255,0,255)), 1)
        img_superpixelized_text = img_superpixelized_text/255.
        
        if output is not None:
            cv2.imwrite(output, img_as_ubyte(img_superpixelized_text))

    def visualize_textonmap_superpixels(self, output=None):
        textonmap_rgb = label2rgb(self.textonmap, image=None, colors=None, alpha=0.3, image_alpha=1)
        
        img_superpixelized = mark_boundaries(gray2rgb(self.img), self.segmentation)
        img_superpixelized_text = img_as_ubyte(img_superpixelized)
        for s in range(self.n_superpixels):
            img_superpixelized_text = cv2.putText(img_superpixelized_text, str(s), 
                                                  tuple(np.floor(self.sp_centroids[s][::-1]).astype(np.int)), 
                                                  cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                  0.8, ((255,0,255)), 1)
        img_superpixelized_text = img_superpixelized_text/255.
        
        if output is not None:
            cv2.imwrite(output, img_as_ubyte(.5*textonmap_rgb + .5*img_superpixelized_text))

        
    def _kl(self, u,v):
        eps = 0.001
        return np.sum((u+eps)*np.log((u+eps)/(v+eps)))

    def _kl_no_eps(self, u,v):
        return np.sum(u*np.log(u/v))
    
    def _chi2(self, u,v):
        return np.sum(np.where(u+v!=0, (u-v)**2/(u+v), 0))

    @timeit
    def compute_distance_matrix(self):
        sample_interval = 1
        gridy, gridx = np.mgrid[:self.img.shape[0]:sample_interval, :self.img.shape[1]:sample_interval]

        all_seg = self.segmentation[gridy.ravel(), gridx.ravel()]
        all_texton = self.textonmap[gridy.ravel(), gridx.ravel()]
        sp_texton_hist = np.array([np.bincount(all_texton[all_seg == s], minlength=self.num_textons) 
                         for s in range(self.n_superpixels)])

        row_sums = sp_texton_hist.sum(axis=1)
        self.sp_texton_hist_normalized = sp_texton_hist.astype(np.float) / row_sums[:, np.newaxis]
        D = pdist(self.sp_texton_hist_normalized, self._kl)
        self.hist_distance_matrix = squareform(D)

    @timeit
    def compute_connectivity(self):
        edge_map = gradient(self.segmentation.astype(np.uint8), disk(3))
        self.neighbors = [set() for i in range(self.n_superpixels)]
        for y,x in zip(*np.nonzero(edge_map)):
            self.neighbors[self.segmentation[y,x]] |= set(self.segmentation[y-2:y+2,x-2:x+2].ravel())

        rows = np.hstack([s*np.ones((len(self.neighbors[s]),), dtype=np.int) for s in range(self.n_superpixels)])
        cols = np.hstack([list(self.neighbors[s]) for s in range(self.n_superpixels)])
        data = np.ones((cols.size, ), dtype=np.bool)
        self.connectivity_matrix = coo_matrix((data, (rows, cols)), shape=(self.n_superpixels, self.n_superpixels))
    
    @timeit
    def compute_saliency_map(self, neighbor_term_weight=1.):
        overall_texton_hist = np.bincount(self.textonmap[self.mask].flat)
        overall_texton_hist_normalized = overall_texton_hist.astype(np.float) / overall_texton_hist.sum()

        individual_saliency_score = np.array([self._kl(sp_hist, overall_texton_hist_normalized) for sp_hist in self.sp_texton_hist_normalized])

        self.saliency_score = np.zeros((self.n_superpixels,))
        for i, sp_hist in enumerate(self.sp_texton_hist_normalized):
            if i in self.bg_superpixels: continue
            self.saliency_score[i] = individual_saliency_score[i]
            neighbor_term = 0
            for j in self.neighbors[i]:
                if j!=i and j not in self.bg_superpixels:
                    neighbor_term += np.exp(-self.hist_distance_matrix[i,j]) * individual_saliency_score[j]
            self.saliency_score[i] += neighbor_term_weight*neighbor_term/(len(self.neighbors[i])-1)

        self.saliency_map = self.saliency_score[self.segmentation]
    
    def visualize_saliency_map(self, output=None):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        fig, ax = plt.subplots()
        im = ax.matshow(self.saliency_map, cmap=plt.cm.Greys_r)
        ax.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.close();
        if output is not None:
            fig.savefig(output, bbox_inches='tight')
    
    @timeit    
    def find_salient_clusters(self, dist_thresh = 0.5, n_top_clusters=10):
    
        chosen_superpixels = set([])
        self.clusters = []

        for t in range(n_top_clusters):
            for i in self.saliency_score.argsort()[::-1]:
                if i not in chosen_superpixels and i not in self.bg_superpixels:
                    break

            curr_cluster = np.array([i], dtype=np.int)
            frontier = [i]
            while len(frontier) > 0:
                i = frontier.pop(-1)
                for j in self.neighbors[i]:
                    if j != i and j not in curr_cluster and j not in chosen_superpixels\
                    and self.hist_distance_matrix[curr_cluster,j].mean() < dist_thresh\
                    and i not in self.bg_superpixels and j not in self.bg_superpixels:
                        curr_cluster = np.append(curr_cluster, j)
                        frontier.append(j)

            self.clusters.append(curr_cluster)
            chosen_superpixels |= set(curr_cluster)
    
    def visualize_salient_clusters(self, output=None):
        segmentation_copy = np.zeros_like(self.segmentation)

        for i, c in enumerate(self.clusters):
            propagate_selection = np.equal.outer(self.segmentation, c).any(axis=2)
            segmentation_copy[propagate_selection] = i + 1

        selection_rgb = label2rgb(segmentation_copy, self.img, 
                                  bg_label=0, bg_color=(1,1,1), 
                                  colors=None)
        if output is not None:
            cv2.imwrite(output, img_as_ubyte(selection_rgb))

