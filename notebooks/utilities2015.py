from skimage.filters import threshold_otsu, threshold_adaptive, gaussian_filter
from skimage.color import color_dict, gray2rgb, label2rgb, rgb2gray
from skimage.segmentation import clear_border
from skimage.morphology import binary_dilation, binary_erosion, watershed, remove_small_objects
from skimage.measure import regionprops, label
from skimage.restoration import denoise_bilateral
from skimage.util import img_as_ubyte, img_as_float
from skimage.io import imread, imsave
from scipy.spatial.distance import cdist, pdist
import numpy as np
import os
import csv
import sys
from operator import itemgetter
import json
import cPickle as pickle

import cv2

from tables import *

from subprocess import check_output, call

import matplotlib.pyplot as plt

def create_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def execute_command(cmd):
    print cmd

    try:
        retcode = call(cmd, shell=True)
        if retcode < 0:
            print >>sys.stderr, "Child was terminated by signal", -retcode
        else:
            print >>sys.stderr, "Child returned", retcode
    except OSError as e:
        print >>sys.stderr, "Execution failed:", e
        raise e

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
    
    if len(scores) > min_size:
    
        scores_shifted = scores[min_size-1:]
        scores_shifted_positive = scores_shifted - scores_shifted.min()

        peaks_shifted = np.atleast_1d(np.squeeze(peak_local_max(scores_shifted_positive, 
                                    min_distance=min_distance, threshold_abs=threshold_abs-scores_shifted.min(), exclude_border=False)))
        peaks_shifted = peaks_shifted[scores_shifted[peaks_shifted] >= np.max(scores_shifted) - threshold_rel]

        peaks_shifted = np.unique(np.r_[peaks_shifted, np.argmax(scores_shifted)])

        if verbose:
            print 'raw peaks', np.atleast_1d(np.squeeze(min_size - 1 + peaks_shifted))

        if len(peaks_shifted) > 0:
            peaks = min_size - 1 + peaks_shifted
        else:
            peaks = np.array([np.argmax(scores[min_size-1:]) + min_size-1])

        peakedness = np.array([scores[p]-np.mean(np.r_[scores[max(min_size-1, p-peakedness_radius):p], 
                                                       scores[p+1:min(len(scores), p+1+peakedness_radius)]]) for p in peaks])

        if verbose:
            print 'peakedness', peakedness
            print 'filtered peaks', np.atleast_1d(np.squeeze(peaks))

        high_peaks = peaks[peakedness > peakedness_lim]
        high_peaks = np.unique(np.r_[high_peaks, min_size - 1 + np.argmax(scores_shifted)])

        high_peaks_order = scores[high_peaks].argsort()[::-1]
        high_peaks_sorted = high_peaks[high_peaks_order]

        high_peaks_peakedness = np.array([scores[p]-np.mean(np.r_[scores[max(min_size-1, p-peakedness_radius):p], 
                                                                scores[p+1:min(len(scores), p+1+peakedness_radius)]]) 
                                for p in high_peaks_sorted])

    else:
        high_peaks_sorted = np.array([np.argmax(scores)])
        high_peaks_peakedness = np.inf
        
    return high_peaks_sorted, high_peaks_peakedness    


class DataManager(object):

    def __init__(self, data_dir=os.environ['GORDON_DATA_DIR'], 
                 repo_dir=os.environ['GORDON_REPO_DIR'], 
                 result_dir=os.environ['GORDON_RESULT_DIR'], 
                 labeling_dir=os.environ['GORDON_LABELING_DIR'],
                 gabor_params_id=None, 
                 segm_params_id=None, 
                 vq_params_id=None,
                 stack=None,
                 resol='lossless',
                 section=None,
                 load_mask=True):

        self.data_dir = data_dir
        self.repo_dir = repo_dir
        self.params_dir = os.path.join(repo_dir, 'params')

        self.root_labelings_dir = labeling_dir
        self.labelnames_path = os.path.join(labeling_dir, 'labelnames.txt')
    
        if os.path.isfile(self.labelnames_path):
            with open(self.labelnames_path, 'r') as f:
                self.labelnames = [n.strip() for n in f.readlines()]
                self.labelnames = [n for n in self.labelnames if len(n) > 0]
        else:
            self.labelnames = []

        self.root_results_dir = result_dir

        self.slice_ind = None
        self.image_name = None

        if gabor_params_id is None:
            self.set_gabor_params('blueNisslWide')
        else:
            self.set_gabor_params(gabor_params_id)

        if segm_params_id is None:
            self.set_segmentation_params('blueNisslRegular')
        else:
            self.set_segmentation_params(segm_params_id)

        if vq_params_id is None:
            self.set_vq_params('blueNissl')
        else:
            self.set_vq_params(vq_params_id)
            
        if stack is not None:
            self.set_stack(stack)

        if resol is not None:
            self.set_resol(resol)

        if section is not None:
            self.set_slice(section)

        if load_mask:
            self._load_mask()

#     def set_labelnames(self, labelnames):
#         self.labelnames = labelnames

#         with open(self.labelnames_path, 'w') as f:
#             for n in labelnames:
#                 f.write('%s\n' % n)

    def set_stack(self, stack):
        self.stack = stack
#         self.stack_path = os.path.join(self.data_dir, self.stack)
#         self.slice_ind = None
        
    def set_resol(self, resol):
        self.resol = resol
    

    def _load_mask(self):

        self.mask = np.zeros((self.image_height, self.image_width), np.bool)

        if self.stack == 'MD593':
            self.mask[1848:1848+4807, 924:924+10186] = True
        elif self.stack == 'MD594':
            self.mask[1081:1081+6049, 552:552+12443] = True
        else:
            raise 'mask is not specified'

        xs_valid = np.any(self.mask, axis=0)
        ys_valid = np.any(self.mask, axis=1)
        self.xmin = np.where(xs_valid)[0][0]
        self.xmax = np.where(xs_valid)[0][-1]
        self.ymin = np.where(ys_valid)[0][0]
        self.ymax = np.where(ys_valid)[0][-1]

        # rs, cs = np.where(self.mask)
        # self.ymax = rs.max()
        # self.ymin = rs.min()
        # self.xmax = cs.max()
        # self.xmin = cs.min()
        self.h = self.ymax-self.ymin+1
        self.w = self.xmax-self.xmin+1

    def set_slice(self, slice_ind):
        assert self.stack is not None and self.resol is not None, 'Stack is not specified'
        self.slice_ind = slice_ind
        self.slice_str = '%04d' % slice_ind
        if self.resol == 'lossless':
            self.image_dir = os.path.join(self.data_dir, self.stack+'_'+self.resol+'_cropped')
            self.image_name = '_'.join([self.stack, self.slice_str, self.resol])
            self.image_path = os.path.join(self.image_dir, self.image_name + '_warped.tif')

        self.image_width, self.image_height = map(int, check_output("identify -format %%Wx%%H %s" % self.image_path, shell=True).split('x'))


        # self.labelings_dir = os.path.join(self.image_dir, 'labelings')
        self.labelings_dir = os.path.join(self.root_labelings_dir, self.stack, self.slice_str)
        
#         self.results_dir = os.path.join(self.image_dir, 'pipelineResults')
        
        self.results_dir = os.path.join(self.root_results_dir, self.stack, self.slice_str)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    # def set_image(self, stack, slice_ind):
    #     self.set_stack(stack)
    #     self.set_slice(slice_ind)
    #     self._load_image()

    def load_multiple_results(self, results):

        from networkx import from_dict_of_lists

        if 'texHist' in results and not hasattr(self, 'texton_hists'):
            self.texton_hists = self.load_pipeline_result('texHist')

        if 'segmentation' in results and  not hasattr(self, 'segmentation'):
            self.segmentation = self.load_pipeline_result('segmentation')
            self.n_superpixels = self.segmentation.max() + 1
        
        if 'texMap' in results and not hasattr(self, 'textonmap'):
            self.textonmap = self.load_pipeline_result('texMap')
            self.n_texton = self.textonmap.max() + 1

        if 'spCentroids' in results and not hasattr(self, 'sp_centroids'):
            self.sp_centroids = self.load_pipeline_result('spCentroids')

        if 'spCoords' in results and not hasattr(self, 'sp_coords'):
            self.sp_coords = self.load_pipeline_result('spCoords')            

        if 'spAreas' in results and not hasattr(self, 'sp_areas'):
            self.sp_areas = self.load_pipeline_result('spAreas')

        if 'edgeCoords' in results and not hasattr(self, 'edge_coords'):
            self.edge_coords = dict(self.load_pipeline_result('edgeCoords'))

        if 'edgeMidpoints' in results and not hasattr(self, 'edge_midpoints'):
            self.edge_midpoints = dict(self.load_pipeline_result('edgeMidpoints'))            

        if 'neighbors' in results and not  hasattr(self, 'neighbors'):
            self.neighbors = self.load_pipeline_result('neighbors')
            self.neighbor_graph = from_dict_of_lists(dict(enumerate(self.neighbors)))
            if not hasattr(self, 'edge_coords'):
                self.edge_coords = dict(self.load_pipeline_result('edgeCoords'))
            self.neighbors_long = dict([(s, set([n for n in nbrs if len(self.edge_coords[frozenset([s,n])]) > 10])) 
                       for s, nbrs in enumerate(self.neighbors)])
            self.neighbor_long_graph = from_dict_of_lists(self.neighbors_long)

        if 'spCentroids' in results and not hasattr(self, 'sp_centroids'):
            self.sp_centroids = self.load_pipeline_result('spCentroids')
        
        if 'edgeNeighbors' in results and not hasattr(self, 'edge_neighbors'):
            self.edge_neighbors = self.load_pipeline_result('edgeNeighbors')

        if 'dedgeNeighbors' in results and not hasattr(self, 'dedge_neighbors'):
            self.dedge_neighbors = self.load_pipeline_result('dedgeNeighbors')
            self.dedge_neighbor_graph = from_dict_of_lists(self.dedge_neighbors)


    def compute_cluster_score(self, cluster, seed=None, seed_weight=0, verbose=False, method='rc-mean', thresh=.2):
        
        cluster_list = list(cluster)
        cluster_avg = self.texton_hists[cluster_list].mean(axis=0)

        surrounds = set([i for i in set.union(*[self.neighbors[c] for c in cluster]) if i not in cluster and i != -1])
        
        if len(surrounds) == 0: # single sp on background
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        surrounds_list = list(surrounds)

        # if verbose:
        #     print 'min', surrounds_list[ds.argmin()]

        ds = np.atleast_1d(np.squeeze(chi2s([cluster_avg], self.texton_hists[surrounds_list])))

        if method == 'min':
            surround_dist = ds.min()
            if verbose:
                print 'min', surrounds_list[ds.argmin()]
            score = surround_dist

        elif method == 'mean':
            surround_dist = ds.mean()
            score = surround_dist

        elif method == 'percentage':
            surround_dist = np.count_nonzero(ds > thresh) / float(len(ds)) # hard
            score = surround_dist

        elif method == 'percentage-soft':        
            sigma = .01
            surround_dist = np.sum(1./(1+np.exp((thresh - ds)/sigma)))/len(ds); #soft        
            if verbose:
                for t in sorted(zip(surrounds_list, ds), key=itemgetter(1)):
                    print t
                plt.hist(ds, bins=np.linspace(0,1,50));
                plt.show();

            score = surround_dist

        elif method == 'rc-min' or method == 'rc-mean':

            sigs_front = []
            if len(cluster) > 1:
                
                # frontiers = set.union(*[set(self.neighbors[s]) for s in surrounds_list]) & set(cluster_list)
                frontiers = cluster
                for f in frontiers:
                    if len(surrounds & set(self.neighbors[f])) > 0:
                        alternative_sps = list((surrounds & set(self.neighbors[f])) - {-1})
                    else:
                        q = list(surrounds-{-1})
                        alternative_sps = [q[np.squeeze(cdist([self.sp_centroids[f]], self.sp_centroids[q])).argmin()]]
                        
                    # alternative_dist = np.atleast_1d(np.squeeze(chi2s([self.texton_hists[f]], 
                    #                                 self.texton_hists[alternative_sps+[f]].mean(axis=0)))).min()
                    # alternative_dist = np.min([chi2(self.texton_hists[f], self.texton_hists[[s,f]].mean(axis=0)) for s in alternative_sps])
                    alternative_dist = np.mean([chi2(self.texton_hists[f], self.texton_hists[[s,f]].mean(axis=0)) for s in alternative_sps])

                    # interior_neighbors = list((set(cluster_list) & set(self.neighbors[f])) - {-1})
                    # interior_avg = self.texton_hists[interior_neighbors + [f]].mean(axis=0)
                    # curr_dist = .5 * chi2(self.texton_hists[f], interior_avg) + .5 * chi2(self.texton_hists[f], cluster_avg)
                    
                    if seed is not None:
                        curr_dist = chi2(self.texton_hists[f], seed_weight*self.texton_hists[seed]+(1.-seed_weight)*self.texton_hists[cluster_list].mean(axis=0))                        
                    else:
                        curr_dist = chi2(self.texton_hists[f], self.texton_hists[cluster_list].mean(axis=0))

                    sig = alternative_dist - curr_dist
                    sigs_front.append(sig)

                if verbose:
                    print 'frontiers advantages'
                    print zip(list(frontiers), sigs_front)

            sigs_sur = []
            for s in surrounds:
                sur_neighbors = self.neighbors[s] - set(cluster)
                alternative_dist = np.mean([chi2(self.texton_hists[s], self.texton_hists[[s,n]].mean(axis=0)) for n in sur_neighbors])

                if seed is not None:
                    curr_dist = chi2(self.texton_hists[s], seed_weight*self.texton_hists[seed]+(1.-seed_weight)*self.texton_hists[cluster_list+[s]].mean(axis=0))
                else:
                    curr_dist = chi2(self.texton_hists[s], self.texton_hists[cluster_list+[s]].mean(axis=0))

                sig = curr_dist - alternative_dist
                sigs_sur.append(sig)

            if verbose:
                print 'surround advantages'
                print zip(list(surrounds), sigs_sur)

            # sigs_sur = np.array(sigs_sur)
            # sigs_front = np.array(sigs_front)

            # thresh = .2
            # # sig = int(sig > thresh)
            # sigma = .025
            # sigs = 1./(1+np.exp((thresh - sigs)/sigma)); #soft

            if method == 'rc-min':
                if len(sigs_front) > 0:
                    score = min(np.min(sigs_sur), np.min(sigs_front))
                    s1_max = np.max(sigs_sur)
                    s1_min = np.min(sigs_sur)
                    s2_max = np.max(sigs_front)
                    s2_min = np.min(sigs_front)
                else:
                    score = np.min(sigs_sur)
                    s1_max = np.max(sigs_sur)
                    s1_min = np.min(sigs_sur)
                    s2_max = np.nan
                    s2_min = np.nan

                # score = .5*np.min(sigs_sur)+.5*np.min(sigs_front) if len(sigs_front) > 0 else 0                
            elif method == 'rc-mean':
                if len(sigs_front) > 0:
                    # print np.mean(sigs_sur), np.mean(sigs_front)
                    score = .5*np.mean(sigs_sur)+.5*np.mean(sigs_front)
                    # score = max(np.mean(sigs_sur), np.mean(sigs_front))
                    s1_max = np.max(sigs_sur)
                    s1_min = np.min(sigs_sur)
                    s2_max = np.max(sigs_front)
                    s2_min = np.min(sigs_front)
                else:
                    score = np.mean(sigs_sur)
                    s1_max = np.max(sigs_sur)
                    s1_min = np.min(sigs_sur)
                    s2_max = np.nan
                    s2_min = np.nan

        else:
            raise 'unrecognized method'
                # print list(frontiers)[np.argmin(sigs)]


        inter_sp_dists = np.squeeze(pdist(self.texton_hists[list(cluster)], chi2))
        inter_sp_dist = inter_sp_dists.mean()

        if seed is not None:
            seed_dist = chi2(cluster_avg, self.texton_hists[seed])
        else:
            seed_dist = np.nan

        if method == 'rc-min' or method == 'rc-mean':
            return score,  np.mean(sigs_sur),  np.mean(sigs_front), inter_sp_dist, seed_dist, s1_max, s1_min, s2_max, s2_min
        else:
            return score,  np.nan, np.nan, inter_sp_dist, seed_dist, np.nan, np.nan, np.nan, np.nan


    # def plot_scores(self, peaks, clusters_allhistory, scores, margin=300, visualize_peaks=False,
    #                 xmin=None, ymin=None, xmax=None, ymax=None, ncol=4, sort_by_score=True):
    def plot_scores(self, peaks1, peaks2, clusters_allhistory, scores, margin=300, visualize_peaks=False,
                    xmin=None, ymin=None, xmax=None, ymax=None, ncol=4, sort_by_score=True):

        fig, axes = plt.subplots(7,1, squeeze=True, sharex=True, figsize=(20,40))

        axes = np.atleast_1d(axes)

        peaks = np.unique(np.r_[peaks1, peaks2])

        scores_to_plot = scores[:,1]
        axes[0].plot(scores_to_plot);
        for p in peaks:
            axes[0].vlines(p, ymin=scores_to_plot.min(), ymax=scores_to_plot.max(), colors='r');
        axes[0].set_xlabel('iteration');
        axes[0].set_ylabel('significance score', fontsize=20);

        s1_mean = scores[:,2]
        axes[1].plot(s1_mean);
        for p in peaks1:
            axes[1].vlines(p, ymin=s1_mean.min(), ymax=s1_mean.max(), colors='r');
        axes[1].set_xlabel('iteration');
        axes[1].set_ylabel('surround keep outside', fontsize=20);

        s2_mean = scores[:,3]
        axes[2].plot(s2_mean);
        for p in peaks2:
            axes[2].vlines(p, ymin=np.nanmin(s2_mean), ymax=np.nanmax(s2_mean), colors='r');
        axes[2].set_xlabel('iteration');
        axes[2].set_ylabel('frontier keep inside', fontsize=20);

        s1_mean = scores[:,2]
        s1_max = scores[:,6]
        s1_min = scores[:,7]
        axes[3].plot(s1_mean);
        axes[3].plot(s1_max, 'g');
        axes[3].plot(s1_min, 'g');
        for p in peaks1:
            axes[3].vlines(p, ymin=s1_mean.min(), ymax=s1_mean.max(), colors='r');
        axes[3].set_xlabel('iteration');
        axes[3].set_ylabel('surround keep outside', fontsize=20);

        s2_mean = scores[:,3]
        s2_max = scores[:,8]
        s2_min = scores[:,9]
        axes[4].plot(s2_mean);
        axes[4].plot(s2_max, 'g');
        axes[4].plot(s2_min, 'g');
        for p in peaks2:
            axes[4].vlines(p, ymin=np.nanmin(s2_mean), ymax=np.nanmax(s2_mean), colors='r');
        axes[4].set_xlabel('iteration');
        axes[4].set_ylabel('frontier keep inside', fontsize=20);

        scores_to_plot = scores[:,4]
        axes[5].plot(scores_to_plot);
        for p in peaks:
            axes[5].vlines(p, ymin=np.nanmin(scores_to_plot), ymax=np.nanmax(scores_to_plot), colors='r');
        axes[5].set_xlabel('iteration');
        axes[5].set_ylabel('mean interior distance', fontsize=20);

        scores_to_plot = scores[:,5]
        axes[6].plot(scores_to_plot);
        for p in peaks:
            axes[6].vlines(p, ymin=scores_to_plot.min(), ymax=scores_to_plot.max(), colors='r');
        axes[6].set_xlabel('iteration');
        axes[6].set_ylabel('seed-avg distance', fontsize=20);

        if not sort_by_score:
            peaks = sorted(peaks)

        if len(peaks) > 0:    
            if visualize_peaks:
                self.visualize_clusters_in_subplots([clusters_allhistory[p] for p in peaks], 
                        ['peak %d, score %.3f'%(pk, sc) for pi, (pk, sc) in enumerate(zip(peaks, scores[peaks,1]))],
                        ncol=ncol)
        

    def visualize_clusters_in_subplots(self, clusters, titles=None, ncol=3, 
                                    xmin=None, ymin=None, xmax=None, ymax=None):

        margin = 300
        ncol = min(ncol, len(clusters))

        if xmin is None:
            self.load_multiple_results(['spCentroids'])

            xmin = np.inf
            ymin = np.inf
            xmax = 0
            ymax = 0

            for cl in clusters:
                centroids = self.sp_centroids[list(cl), ::-1]
                xmin = min(xmin, centroids[:,0].min(axis=0))
                xmax = max(xmax, centroids[:,0].max(axis=0))
                ymin = min(ymin, centroids[:,1].min(axis=0))
                ymax = max(ymax, centroids[:,1].max(axis=0))

            xmin = int(max(0, xmin - margin))
            ymin = int(max(0, ymin - margin))
            xmax = int(min(self.image_width, xmax + margin))
            ymax = int(min(self.image_height, ymax + margin))

        n_cls = len(clusters)
        fig, axes = plt.subplots(int((n_cls-1)/ncol)+1, ncol, figsize=(20,20), squeeze=False)

        for ci, cl in enumerate(clusters):
            viz = self.visualize_cluster(cl, highlight_seed=True, seq_text=True,
                                       ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax)
            ax = axes[ci/ncol, ci%ncol]
            ax.imshow(viz)
            ax.axis('off')
            if titles is not None:
                ax.set_title(titles[ci])
            
        plt.show()

    def grow_cluster(self, seed, seed_weight=.5,
                    verbose=False, all_history=True, 
                     num_sp_percentage_limit=0.05,
                     min_size=1, min_distance=3, thresh=.4,
                     threshold_abs=-0.05, threshold_rel=.4,
                     peakedness_limit=0.001, method='rc-min'):

        from networkx import from_dict_of_lists, Graph, adjacency_matrix, connected_components

        from itertools import chain
        from skimage.feature import peak_local_max
        from scipy.spatial import ConvexHull
        from matplotlib.path import Path

        # self.load_multiple_results(['neighbors', 'texHist', 'segmentation'])

        neighbor_long_graph = from_dict_of_lists(self.neighbors_long)

        visited = set([])
        curr_cluster = set([])

        candidate_scores = [0]
        candidate_sps = [seed]

        score_tuples = []
        added_sps = []
        n_sps = []

        cluster_list = []
        addorder_list = []

        iter_ind = 0

        hull_begin = False

        nearest_surrounds = []
        toadd_list = []

        while len(candidate_sps) > 0:

            if verbose:
                print '\niter', iter_ind

            best_ind = np.argmax(candidate_scores)

            just_added_score = candidate_scores[best_ind]
            sp = candidate_sps[best_ind]

            del candidate_scores[best_ind]
            del candidate_sps[best_ind]

            if sp in curr_cluster:
                continue

            curr_cluster.add(sp)
            added_sps.append(sp)

            extra_sps = []

            sg = self.neighbor_long_graph.subgraph(list(set(range(self.n_superpixels)) - curr_cluster))
            for c in connected_components(sg):
                if len(c) < 10: # holes
                    extra_sps.append(c)

            extra_sps = list(chain(*extra_sps))
            curr_cluster |= set(extra_sps)
            added_sps += extra_sps

            tt = self.compute_cluster_score(curr_cluster, seed=seed, seed_weight=seed_weight, verbose=verbose, thresh=thresh, method=method)

            # nearest_surround = compute_nearest_surround(curr_cluster, neighbors, texton_hists)
            # nearest_surrounds.append(nearest_surround)

            tot, s1, s2, inter_sp_dist, seed_dist, s1_max, s1_min, s2_max, s2_min = tt

            cluster_avg = self.texton_hists[list(curr_cluster)].mean(axis=0)

            if (len(curr_cluster) > 5 and (seed_dist > .2 or inter_sp_dist > .3)) or (len(curr_cluster) > int(self.n_superpixels * num_sp_percentage_limit)):
                # if verbose:
                print 'terminate', seed_dist, inter_sp_dist
                break

            if np.isnan(tot):
                return [seed], -np.inf
            score_tuples.append(np.r_[just_added_score, tt])

            n_sps.append(len(curr_cluster))

            # just_added_score, curr_total_score, exterior_score, interior_score, compactness_score, surround_pval,
            # interior_pval, size_prior

            if verbose:
                print 'add', sp
                print 'extra', extra_sps
                print 'added_sps', added_sps
                print 'curr_cluster', curr_cluster
                print 'n_sps', n_sps
                print 'tt', tot
                if len(curr_cluster) != len(added_sps):
                    print len(curr_cluster), len(added_sps)
                    raise

            cluster_list.append(curr_cluster.copy())
            addorder_list.append(added_sps[:])
            candidate_sps = (set(candidate_sps) | \
                             (set.union(*[self.neighbors_long[i] for i in list(extra_sps)+[sp]]) - {-1})) - curr_cluster
            candidate_sps = list(candidate_sps)

            # for c in candidate_sps:
            #     int_dist = chi2(self.texton_hists[c], self.texton_hists[list(curr_cluster)+[c]].mean(axis=0))
            #     ext_neighbors = self.neighbors[c] - set(curr_cluster)
            #     chi2(self.texton_hists[c], self.texton_hists[s+[c]]) for s in ext_neighbors

            candidate_scores = []
            for c in candidate_sps:
                int_neighbors = list(set(curr_cluster) & self.neighbors[c])
                int_dist = chi2(self.texton_hists[c], self.texton_hists[int_neighbors + [c]].mean(axis=0))
                curr_dist = chi2(self.texton_hists[c], self.texton_hists[list(curr_cluster)+[c]].mean(axis=0))
                seed_dist = chi2(self.texton_hists[c], self.texton_hists[seed])
                sc = .1 * int_dist + .3* curr_dist + .6*seed_dist
                candidate_scores.append(-sc)
                
            # h_avg = self.texton_hists[list(curr_cluster)].mean(axis=0)
            # candidate_scores = -.5*chi2s([h_avg], self.texton_hists[candidate_sps])-\
            #                 .5*chi2s([self.texton_hists[seed]], self.texton_hists[candidate_sps])

            # candidate_scores = candidate_scores.tolist()

            if verbose:
    #                 print 'candidate', candidate_sps
                print 'candidate\n'

                for i,j in sorted(zip(candidate_scores, candidate_sps), reverse=True):
                    print i, j
                print 'best', candidate_sps[np.argmax(candidate_scores)]

            toadd_list.append(candidate_sps[np.argmax(candidate_scores)])

            iter_ind += 1

        score_tuples = np.array(score_tuples)

        # peaks_sorted, peakedness_sorted = find_score_peaks(score_tuples[:,1], min_size=min_size, min_distance=min_distance,
        #                                                     threshold_abs=threshold_abs, threshold_rel=threshold_rel, 
        #                                                     peakedness_lim=peakedness_limit,
        #                                                     verbose=verbose)

        peaks_sorted1, peakedness_sorted1 = find_score_peaks(score_tuples[:,2], min_size=min_size, min_distance=min_distance,
                                                            threshold_abs=threshold_abs, threshold_rel=threshold_rel, 
                                                            peakedness_lim=peakedness_limit,
                                                            verbose=verbose)

        peaks_sorted2, peakedness_sorted2 = find_score_peaks(score_tuples[:,3], min_size=min_size, min_distance=min_distance,
                                                            threshold_abs=threshold_abs, threshold_rel=threshold_rel, 
                                                            peakedness_lim=peakedness_limit,
                                                            verbose=verbose)

        peaks_sorted = np.unique(np.r_[peaks_sorted1, peaks_sorted2])
        peakedness_sorted = np.unique(np.r_[peakedness_sorted1, peakedness_sorted2])

        if all_history:
            # return addorder_list, score_tuples, peaks_sorted, peakedness_sorted, nearest_surrounds, toadd_list
            return addorder_list, score_tuples, peaks_sorted, peakedness_sorted, toadd_list, peaks_sorted1, peaks_sorted2
        else:
            return [addorder_list[i] for i in peaks_sorted], score_tuples[peaks_sorted, 1]



    def convert_cluster_to_descriptor(self, cluster, verbose=False):

        self.load_multiple_results(['neighbors', 'dedgeNeighbors', 'texHist', 'edgeCoords', 'spAreas'])

        dedge_set = self.find_boundary_dedges_ordered(cluster, verbose=verbose)
        
        interior_texture = self.texton_hists[list(cluster)].mean(axis=0)

        surrounds = [e[0] for e in dedge_set]
        exterior_textures = np.array([self.texton_hists[s] if s!=-1 else np.nan * np.ones((self.texton_hists.shape[1],)) 
                                      for s in surrounds])

        points = np.array([self.edge_coords[frozenset(e)].mean(axis=0) for e in dedge_set])
        center = points.mean(axis=0)

        area = self.sp_areas[cluster].sum()
        
        return (dedge_set, interior_texture, exterior_textures, points, center, area)


    def find_boundary_dedges_ordered(self, cluster, verbose=False):

        self.load_multiple_results(['neighbors', 'dedgeNeighbors', 'edgeCoords'])

        surrounds = set([i for i in set.union(*[self.neighbors[c] for c in cluster]) if i not in cluster])
        surrounds = set([i for i in surrounds if any([n not in cluster for n in self.neighbors[i]])])
        
        non_border_dedges = [(s, int_sp) for s in surrounds for int_sp in set.intersection(set(cluster), self.neighbors[s]) 
                             if int_sp != -1 and s != -1]
        border_dedges = [(-1,f) for f in cluster if -1 in self.neighbors[f]] if -1 in surrounds else []

        dedges_cluster = non_border_dedges + border_dedges
        # dedges_cluster_long = [dedge for dedge in dedges_cluster if len(self.edge_coords[frozenset(dedge)]) > 10]

        if verbose:
            print 'surrounds', surrounds
            # print 'non_border_dedges', non_border_dedges
            # print 'border_dedges', border_dedges
            print 'dedges_cluster', dedges_cluster
            # print 'dedges_cluster_long', dedges_cluster_long

        # dedges_cluster_long_sorted = order_nodes(dedges_cluster_long, self.dedge_neighbor_graph, verbose=verbose)    
        dedges_cluster_sorted = order_nodes(dedges_cluster, self.dedge_neighbor_graph, verbose=verbose)    
        
        # missing = set(dedges_cluster_long) - set(dedges_cluster_long_sorted)
        # assert len(missing) == 0, missing
        missing = set(dedges_cluster) - set(dedges_cluster_sorted)
        if len(missing) > 0:
            print 'dedges missing', len(missing), '%d%%'%(100*len(missing)/float(len(dedges_cluster)))
            dedges_cluster_sorted += missing
        
        return dedges_cluster_sorted
        # return dedges_cluster_long_sorted


    def _get_image_filepath(self, stack=None, resol=None, section=None, version='rgb-jpg'):
        if stack is None:
            stack = self.stack
        if resol is None:
            resol = self.resol
        if section is None:
            section = self.slice_ind
            
        slice_str = '%04d' % section

        if version == 'rgb-jpg':
            image_dir = os.path.join(self.data_dir, stack+'_'+resol+'_cropped_downscaled')
            image_name = '_'.join([stack, slice_str, resol, 'warped_downscaled'])
            image_path = os.path.join(image_dir, image_name + '.jpg')
        # elif version == 'gray-jpg':
        #     image_dir = os.path.join(self.data_dir, stack+'_'+resol+'_cropped_grayscale_downscaled')
        #     image_name = '_'.join([stack, slice_str, resol, 'warped'])
        #     image_path = os.path.join(image_dir, image_name + '.jpg')
        elif version == 'gray':
            image_dir = os.path.join(self.data_dir, stack+'_'+resol+'_cropped_grayscale')
            image_name = '_'.join([stack, slice_str, resol, 'warped_grayscale'])
            image_path = os.path.join(image_dir, image_name + '.tif')
        elif version == 'rgb':
            image_dir = os.path.join(self.data_dir, stack+'_'+resol+'_cropped')
            image_name = '_'.join([stack, slice_str, resol, 'warped'])
            image_path = os.path.join(image_dir, image_name + '.tif')
         
        return image_path
    
    def _read_image(self, image_filename):
        if image_filename.endswith('tif') or image_filename.endswith('tiff'):
            from PIL.Image import open
            img = np.array(open(image_filename))/255.
        else:
            img = imread(image_filename)
        return img

    def _load_image(self, versions=['rgb', 'gray', 'rgb-jpg']):
        
        assert self.image_name is not None, 'Image is not specified'

        if 'rgb-jpg' in versions and not hasattr(self, 'image_rgb_jpg'):
            image_filename = self._get_image_filepath(version='rgb-jpg')
            # assert os.path.exists(image_filename), "Image '%s' does not exist" % (self.image_name + '.tif')
            self.image_rgb_jpg = self._read_image(image_filename)
        
        if 'rgb' in versions and not hasattr(self, 'image_rgb'):
            image_filename = self._get_image_filepath(version='rgb')
            # assert os.path.exists(image_filename), "Image '%s' does not exist" % (self.image_name + '.tif')
            self.image_rgb = self._read_image(image_filename)

        if 'gray' in versions and not hasattr(self, 'image'):
            image_filename = self._get_image_filepath(version='gray')
            # assert os.path.exists(image_filename), "Image '%s' does not exist" % (self.image_name + '.tif')
            self.image = self._read_image(image_filename)

        # elif format == 'gray-rgb':
        #     self.image = rgb2gray(self.image_rgb)

        # self.image_rgb = imread(image_filename, as_grey=False)
#         self.image = rgb2gray(self.image_rgb)
        
#         mask_filename = os.path.join(self.image_dir, self.image_name + '_mask.png')
#         self.mask = imread(mask_filename, as_grey=True) > 0
        
    def set_gabor_params(self, gabor_params_id):
        
        self.gabor_params_id = gabor_params_id
        # self._generate_kernels(self.gabor_params)
    
    def _generate_kernels(self, gabor_params_id=None):
        
        from skimage.filter import gabor_kernel

        if gabor_params_id is None:
            assert hasattr(self, 'gabor_params_id')
            gabor_params_id = self.gabor_params_id

        self.gabor_params = json.load(open(os.path.join(self.params_dir, 'gabor', 'gabor_' + gabor_params_id + '.json'), 'r')) if gabor_params_id is not None else None
        
        theta_interval = self.gabor_params['theta_interval']
        self.n_angle = int(180/theta_interval)
        freq_step = self.gabor_params['freq_step']
        freq_max = 1./self.gabor_params['min_wavelen']
        freq_min = 1./self.gabor_params['max_wavelen']
        bandwidth = self.gabor_params['bandwidth']
        self.n_freq = int(np.log(freq_max/freq_min)/np.log(freq_step)) + 1
        self.frequencies = freq_max/freq_step**np.arange(self.n_freq)
        self.angles = np.arange(0, self.n_angle)*np.deg2rad(theta_interval)

        kernels = [gabor_kernel(f, theta=t, bandwidth=bandwidth) for f in self.frequencies for t in self.angles]
        kernels = map(np.real, kernels)

        biases = np.array([k.sum() for k in kernels])
        mean_bias = biases.mean()
        self.kernels = [k/k.sum()*mean_bias for k in kernels] # this enforces all kernel sums to be identical, but non-zero

        # kernels = [k - k.sum()/k.size for k in kernels] # this enforces all kernel sum to be zero

        self.n_kernel = len(kernels)
        self.max_kern_size = np.max([kern.shape[0] for kern in self.kernels])

    def print_gabor_info(self):
        print 'num. of kernels: %d' % (self.n_kernel)
        print 'frequencies:', self.frequencies
        print 'wavelength (pixels):', 1/self.frequencies
        print 'max kernel matrix size:', self.max_kern_size
        
    def set_segmentation_params(self, segm_params_id):
        
        self.segm_params_id = segm_params_id
        if segm_params_id == 'gridsize200':
            self.grid_size = 200
        elif segm_params_id == 'gridsize100':
            self.grid_size = 100
        elif segm_params_id == 'gridsize50':
            self.grid_size = 50
        elif segm_params_id == 'tSLIC200':
            pass
        else:
            self.segm_params = json.load(open(os.path.join(self.params_dir, 'segm', 'segm_' + segm_params_id + '.json'), 'r')) if segm_params_id is not None else None

    def set_vq_params(self, vq_params_id):
        
        self.vq_params_id = vq_params_id
        self.vq_params = json.load(open(os.path.join(self.params_dir, 'vq', 'vq_' + vq_params_id + '.json'), 'r')) if vq_params_id is not None else None
        
    
    def _param_str(self, param_dependencies):
        param_strs = []
        if 'gabor' in param_dependencies:
            param_strs.append('gabor-' + self.gabor_params_id)
        if 'segm' in param_dependencies:
            param_strs.append('segm-' + self.segm_params_id)
        if 'vq' in param_dependencies:
            param_strs.append('vq-' + self.vq_params_id)
        return '-'.join(param_strs)
        
    def _refresh_result_info(self):
        with open(self.repo_dir + '/notebooks/results.csv', 'r') as f:
            f.readline()
            self.result_info = {}
            for row in csv.DictReader(f, delimiter=' '):
                self.result_info[row['name']] = row
        
        
    def _get_result_filename(self, result_name):

        if not hasattr(self, 'result_info'):
            with open(self.repo_dir + '/notebooks/results.csv', 'r') as f:
                f.readline()
                self.result_info = {}
                for row in csv.DictReader(f, delimiter=' '):
                    self.result_info[row['name']] = row
                
        info = self.result_info[result_name]
        
        if info['dir'] == '0':
            result_dir = self.results_dir
            prefix = self.image_name
        elif info['dir'] == '1':
            result_dir = os.path.join(os.environ['GORDON_RESULT_DIR'], self.stack)
            prefix = self.stack + '_' + self.resol

        else:
            raise Exception('unrecognized result dir specification')
            
        if info['param_dep'] == '0':
            param_dep = ['gabor']
        elif info['param_dep'] == '1':
            param_dep = ['segm']
        elif info['param_dep'] == '2':
            param_dep = ['gabor', 'vq']
        elif info['param_dep'] == '3':
            param_dep = ['gabor', 'segm', 'vq']
        else:
            raise Exception('unrecognized result param_dep specification')
            
        result_filename = os.path.join(result_dir, '_'.join([prefix, self._param_str(param_dep), 
                                                             result_name + '.' + info['extension']]))

        return result_filename
            
    def check_pipeline_result(self, result_name):
#         if REGENERATE_ALL_RESULTS:
#             return False
        result_filename = self._get_result_filename(result_name)
        return os.path.exists(result_filename)

    def load_pipeline_result(self, result_name, is_rgb=None, section=None):
        
        result_filename = self._get_result_filename(result_name)
        ext = self.result_info[result_name]['extension']
        
        if ext == 'npy':
            assert os.path.exists(result_filename), "%d: Pipeline result '%s' does not exist, trying to find %s" % (self.slice_ind, result_name + '.' + ext, result_filename)
            data = np.load(result_filename)
        elif ext == 'tif' or ext == 'png' or ext == 'jpg':
            data = imread(result_filename, as_grey=False)
            data = self._regulate_image(data, is_rgb)
        elif ext == 'pkl':
            data = pickle.load(open(result_filename, 'r'))
        elif ext == 'hdf':
            with open_file(result_filename, mode="r") as f:
                data = f.get_node('/data').read()

        # print 'loaded %s' % result_filename

        return data
        
    def save_pipeline_result(self, data, result_name, is_rgb=None, section=None):
        
        result_filename = self._get_result_filename(result_name)
        ext = self.result_info[result_name]['extension']

        if ext == 'npy':
            np.save(result_filename, data)
        elif ext == 'tif' or ext == 'png' or ext == 'jpg':
            data = self._regulate_image(data, is_rgb)
            imsave(result_filename, data)
        elif ext == 'pkl':
            pickle.dump(data, open(result_filename, 'w'))
        elif ext == 'hdf':
            filters = Filters(complevel=9, complib='blosc')
            with open_file(result_filename, mode="w") as f:
                _ = f.create_carray('/', 'data', Atom.from_dtype(data.dtype), filters=filters, obj=data)
            
        print 'saved %s' % result_filename
        

    def load_labeling(self, stack=None, section=None, labeling_name=None):
        labeling_fn = self._load_labeling_path(stack, section, labeling_name)
        labeling = pickle.load(open(labeling_fn, 'r'))
        return labeling

    def _load_labeling_preview_path(self, stack=None, section=None, labeling_name=None):
        if stack is None:
            stack = self.stack
        if section is None:
            section = self.slice_ind

        if labeling_name.endswith('pkl'): # full filename
            return os.path.join(self.labelings_dir, labeling_name[:-4]+'.jpg')
        else:
            return os.path.join(self.labelings_dir, '_'.join([stack, '%04d'%section, labeling_name]) + '.jpg')
        
    def _load_labeling_path(self, stack=None, section=None, labeling_name=None):
        if stack is None:
            stack = self.stack
        if section is None:
            section = self.slice_ind

        if labeling_name.endswith('pkl'): # full filename
            return os.path.join(self.labelings_dir, labeling_name)
        else:
            return os.path.join(self.labelings_dir, '_'.join([stack, '%04d'%section, labeling_name]) + '.pkl')
        

    def load_labeling_preview(self, stack=None, section=None, labeling_name=None):
        return imread(self._load_labeling_preview_path(stack, section, labeling_name))
        
    def save_labeling(self, labeling, new_labeling_name, labelmap_vis):
        
        try:
            os.makedirs(self.labelings_dir)
        except:
            pass

        new_labeling_fn = self._load_labeling_path(labeling_name=new_labeling_name)
        # os.path.join(self.labelings_dir, self.image_name + '_' + new_labeling_name + '.pkl')
        pickle.dump(labeling, open(new_labeling_fn, 'w'))
        print 'Labeling saved to', new_labeling_fn

        # new_preview_fn = self._load_labeling_preview_path(labeling_name=new_labeling_name)

        # # os.path.join(self.labelings_dir, self.image_name + '_' + new_labeling_name + '.tif')
        # data = self._regulate_image(labelmap_vis, is_rgb=True)
        # imsave(new_preview_fn, data)
        # print 'Preview saved to', new_preview_fn

        return new_labeling_fn
        
    def _regulate_image(self, img, is_rgb=None):
        """
        Ensure the image is of type uint8.
        """

        if not np.issubsctype(img, np.uint8):
            try:
                img = img_as_ubyte(img)
            except:
                img_norm = (img-img.min()).astype(np.float)/(img.max() - img.min())    
                img = img_as_ubyte(img_norm)

        if is_rgb is not None:
            if img.ndim == 2 and is_rgb:
                img = gray2rgb(img)
            elif img.ndim == 3 and not is_rgb:
                img = rgb2gray(img)

        return img
    
    
    # def visualize_segmentation(self, bg='rgb-jpg', show_sp_index=True):

    #     if bg == 'originalImage':
    #         if not hasattr(self, 'image'):
    #             self._load_image(format='rgb-jpg')
    #         viz = self.image
    #     elif bg == 'transparent':


    #     img_superpixelized = mark_boundaries(viz, self.segmentation)
    #     img_superpixelized = img_as_ubyte(img_superpixelized)
    #     dm.save_pipeline_result(img_superpixelized, 'segmentationWithoutText')

    #     for s in range(n_superpixels):
    #         cv2.putText(img_superpixelized, str(s), 
    #                     tuple(np.floor(sp_centroids[s][::-1]).astype(np.int) - np.array([10,-10])), 
    #                     cv2.FONT_HERSHEY_DUPLEX, .5, ((255,0,255)), 1)



    def visualize_edge_set(self, edges, bg=None, show_edge_index=False, c=None):
        
        import cv2
        
        if not hasattr(self, 'edge_coords'):
            self.edge_coords = self.load_pipeline_result('edgeCoords')
           
        if not hasattr(self, 'edge_midpoints'):
            self.edge_midpoints = self.load_pipeline_result('edgeMidpoints')
            
        if not hasattr(self, 'dedge_vectors'):
            self.dedge_vectors = self.load_pipeline_result('dedgeVectors')

        if bg == 'originalImage':
            segmentation_viz = self.image_rgb_jpg
        elif bg == 'segmentationWithText':
            if not hasattr(self, 'segmentation_vis'):
                self.segmentation_viz = self.load_pipeline_result('segmentationWithText')
            segmentation_viz = self.segmentation_viz
        elif bg == 'segmentationWithoutText':
            if not hasattr(self, 'segmentation_notext_vis'):
                self.segmentation_notext_viz = self.load_pipeline_result('segmentationWithoutText')
            segmentation_viz = self.segmentation_notext_viz
        else:
            segmentation_viz = bg
            
        vis = img_as_ubyte(segmentation_viz[self.ymin:self.ymax+1, self.xmin:self.xmax+1])
        
        directed = isinstance(list(edges)[0], tuple)
        
        if c is None:
            c = [255,0,0]
        
        for e_ind, edge in enumerate(edges):
                
            if directed:
                e = frozenset(edge)
                midpoint = self.edge_midpoints[e]
                end = midpoint + 10 * self.dedge_vectors[edge]
                cv2.line(vis, tuple((midpoint-(self.xmin, self.ymin)).astype(np.int)), 
                         tuple((end-(self.xmin, self.ymin)).astype(np.int)), 
                         (c[0],c[1],c[2]), 2)

                cv2.circle(vis, tuple((end-(self.xmin, self.ymin)).astype(np.int)), 3,
                         (c[0],c[1],c[2]), -1)

                stroke_pts = self.edge_coords[e]
            else:
                stroke_pts = self.edge_coords[edge]

            for x, y in stroke_pts:
                cv2.circle(vis, (x-self.xmin, y-self.ymin), 5, c, -1)

            if show_edge_index:
                cv2.putText(vis, str(e_ind), 
                            tuple(np.floor(midpoint + [-50, 30] - (self.xmin, self.ymin)).astype(np.int)), 
                            cv2.FONT_HERSHEY_DUPLEX, 1, ((c[0],c[1],c[2])), 3)
        
        return vis
    
    
    def visualize_edge_sets(self, edge_sets, bg='segmentationWithText', show_set_index=0, colors=None, neighbors=None, labels=None):
        '''
        Return a visualization of multiple sets of edgelets
        '''
        
        import cv2
        
        if not hasattr(self, 'edge_coords'):
            self.edge_coords = self.load_pipeline_result('edgeCoords')
           
        if not hasattr(self, 'edge_midpoints'):
            self.edge_midpoints = self.load_pipeline_result('edgeMidpoints')
            
        if not hasattr(self, 'dedge_vectors'):
            self.dedge_vectors = self.load_pipeline_result('dedgeVectors')

        if colors is None:
            colors = np.uint8(np.loadtxt(os.environ['GORDON_REPO_DIR'] + '/visualization/100colors.txt') * 255)
        elif isinstance(colors[0], int):
            colors = [colors] * len(edge_sets)

        
        if bg == 'originalImage':
            if not hasattr(self, 'image_rgb_jpg'):
                self._load_image(versions=['rgb-jpg'])
            segmentation_viz = self.image_rgb_jpg
        elif bg == 'segmentationWithText':
            if not hasattr(self, 'segmentation_vis'):
                self.segmentation_viz = self.load_pipeline_result('segmentationWithText')
            segmentation_viz = self.segmentation_viz
        elif bg == 'segmentationWithoutText':
            if not hasattr(self, 'segmentation_notext_vis'):
                self.segmentation_notext_viz = self.load_pipeline_result('segmentationWithoutText')
            segmentation_viz = self.segmentation_notext_viz
        else:
            segmentation_viz = bg
            
        vis = img_as_ubyte(segmentation_viz[self.ymin:self.ymax+1, self.xmin:self.xmax+1])
            
        # if input are tuples, draw directional sign
        if len(edge_sets) == 0:
            return vis
        else:
            directed = isinstance(list(edge_sets[0])[0], tuple)
            
        for edgeSet_ind, edges in enumerate(edge_sets):
            
#             junction_pts = []
            
            if labels is None:
                s = str(edgeSet_ind)
                c = colors[edgeSet_ind%len(colors)].astype(np.int)
            else:
                s = labels[edgeSet_ind]
                c = colors[int(s)%len(colors)].astype(np.int)
            
            for e_ind, edge in enumerate(edges):
                
                if directed:
                    e = frozenset(edge)
                    midpoint = self.edge_midpoints[e]
                    end = midpoint + 10 * self.dedge_vectors[edge]
                    cv2.line(vis, tuple((midpoint-(self.xmin, self.ymin)).astype(np.int)), 
                             tuple((end-(self.xmin, self.ymin)).astype(np.int)), 
                             (c[0],c[1],c[2]), 2)

                    cv2.circle(vis, tuple((end-(self.xmin, self.ymin)).astype(np.int)), 3,
                             (c[0],c[1],c[2]), -1)
                    stroke_pts = self.edge_coords[e]
                else:
                    stroke_pts = self.edge_coords[edge]
                
                for x, y in stroke_pts:
                    cv2.circle(vis, (x-self.xmin, y-self.ymin), 5, c, -1)

                    # vis[max(0, y-5):min(self.image_height, y+5), 
                    #     max(0, x-5):min(self.image_width, x+5)] = (c[0],c[1],c[2],1) if vis.shape[2] == 4 else c
                                                
#                 if neighbors is not None:
#                     nbrs = neighbors[degde]
#                     for nbr in nbrs:
#                         pts2 = self.edge_coords[frozenset(nbr)]
#                         am = np.unravel_index(np.argmin(cdist(pts[[0,-1]], pts2[[0,-1]]).flat), (2,2))
# #                         print degde, nbr, am
#                         junction_pt = (pts[-1 if am[0]==1 else 0] + pts2[-1 if am[1]==1 else 0])/2
#                         junction_pts.append(junction_pt)
                 
            if show_set_index:

                if directed:
                    centroid = np.mean([self.edge_midpoints[frozenset(e)] for e in edges], axis=0)
                else:
                    centroid = np.mean([self.edge_midpoints[e] for e in edges], axis=0)

                cv2.putText(vis, s, 
                            tuple(np.floor(centroid + [-100, 100] - (self.xmin, self.ymin)).astype(np.int)), 
                            cv2.FONT_HERSHEY_DUPLEX,
                            3, ((c[0],c[1],c[2])), 3)
            
#             for p in junction_pts:
#                 cv2.circle(vis, tuple(np.floor(p).astype(np.int)), 5, (255,0,0), -1)
        
        return vis

    
    def visualize_kernels(self):
        fig, axes = plt.subplots(self.n_freq, self.n_angle, figsize=(20,20))

        for i, kern in enumerate(self.kernels):
            r, c = np.unravel_index(i, (self.n_freq, self.n_angle))
            axes[r,c].matshow(kern, cmap=plt.cm.gray)
        #     axes[r,c].colorbar()
        plt.show()


    def visualize_cluster(self, cluster, bg='segmentationWithText', seq_text=False, highlight_seed=True,
                         ymin=None, xmin=None, ymax=None, xmax=None):

        if ymin is None:
            ymin=self.ymin
        if xmin is None:
            xmin=self.xmin
        if ymax is None:
            ymax=self.ymax
        if xmax is None:
            xmax=self.xmax
        
        if not hasattr(self, 'sp_coords'):
            self.sp_coords = self.load_pipeline_result('spCoords')
                    
        if bg == 'originalImage':
            if not hasattr(self, 'image_rgb_jpg'):
                self._load_image(versions=['rgb-jpg'])
            segmentation_viz = self.image_rgb_jpg
        elif bg == 'segmentationWithText':
            if not hasattr(self, 'segmentation_vis'):
                self.segmentation_viz = self.load_pipeline_result('segmentationWithText')
            segmentation_viz = self.segmentation_viz
        elif bg == 'segmentationWithoutText':
            if not hasattr(self, 'segmentation_notext_vis'):
                self.segmentation_notext_viz = self.load_pipeline_result('segmentationWithoutText')
            segmentation_viz = self.segmentation_notext_viz
        else:
            segmentation_viz = bg

        msk = -1*np.ones((self.image_height, self.image_width), np.int8)

        for i, c in enumerate(cluster):
            rs = self.sp_coords[c][:,0]
            cs = self.sp_coords[c][:,1]
            if highlight_seed and i == 0:
                msk[rs, cs] = 1
            else:
                msk[rs, cs] = 0

        viz_msk = label2rgb(msk[ymin:ymax+1, xmin:xmax+1], image=segmentation_viz[ymin:ymax+1, xmin:xmax+1])

        if seq_text:
            viz_msk = img_as_ubyte(viz_msk[...,::-1])

            if not hasattr(self, 'sp_centroids'):
                self.sp_centroids = self.load_pipeline_result('spCentroids')

            import cv2
            for i, sp in enumerate(cluster):
                cv2.putText(viz_msk, str(i), tuple((self.sp_centroids[sp, ::-1] - (xmin, ymin) - (10,-10)).astype(np.int)), cv2.FONT_HERSHEY_DUPLEX, 1., ((0,255,255)), 1)

        return viz_msk

    
    def visualize_edges_and_superpixels(self, edge_sets, clusters, colors=None):
        if colors is None:
            colors = np.loadtxt(os.environ['GORDON_REPO_DIR'] + '/visualization/100colors.txt')
            
        vis = self.visualize_multiple_clusters(clusters, colors=colors)
        viz = self.visualize_edge_sets(edge_sets, directed=True, img=vis, colors=colors)
        return viz
        
    
    def visualize_multiple_clusters(self, clusters, bg='segmentationWithText', alpha_blend=False, colors=None,
                                    show_cluster_indices=False,
                                    ymin=None, xmin=None, ymax=None, xmax=None):
        
        if ymin is None:
            ymin = self.ymin
        if xmin is None:
            xmin = self.xmin
        if ymax is None:
            ymax = self.ymax
        if xmax is None:
            xmax = self.xmax

        if not hasattr(self, 'segmentation'):
            self.segmentation = self.load_pipeline_result('segmentation')

        if len(clusters) == 0:
            return segmentation_vis
        
        if colors is None:
            colors = np.loadtxt(os.environ['GORDON_REPO_DIR'] + '/visualization/100colors.txt')
            
        n_superpixels = self.segmentation.max() + 1
        

        if bg == 'originalImage':
            if not hasattr(self, 'image_rgb_jpg'):
                self._load_image(versions=['rgb-jpg'])
            segmentation_viz = self.image_rgb_jpg
        elif bg == 'segmentationWithText':
            if not hasattr(self, 'segmentation_vis'):
                self.segmentation_viz = self.load_pipeline_result('segmentationWithText')
            segmentation_viz = self.segmentation_viz
        elif bg == 'segmentationWithoutText':
            if not hasattr(self, 'segmentation_notext_vis'):
                self.segmentation_notext_viz = self.load_pipeline_result('segmentationWithoutText')
            segmentation_viz = self.segmentation_notext_viz
        else:
            segmentation_viz = bg

        mask_alpha = .4
        
        if alpha_blend:
            
            for ci, c in enumerate(clusters):
                m =  np.zeros((n_superpixels,), dtype=np.float)
                m[list(c)] = mask_alpha
                alpha = m[self.segmentation[ymin:ymax+1, xmin:xmax+1]]
                alpha[~self.mask[ymin:ymax+1, xmin:xmax+1]] = 0
                
                mm = np.zeros((n_superpixels,3), dtype=np.float)
                mm[list(c)] = colors[ci]
                blob = mm[self.segmentation[ymin:ymax+1, xmin:xmax+1]]
                
                if ci == 0:
                    vis = alpha_blending(blob, segmentation_viz[ymin:ymax+1, xmin:xmax+1], alpha, 1.)
                else:
                    vis = alpha_blending(blob, vis[..., :-1], alpha, vis[..., -1])

        else:
        
            n_superpixels = self.segmentation.max() + 1

            n = len(clusters)
            m = -1*np.ones((n_superpixels,), dtype=np.int)

            for ci, c in enumerate(clusters):
                m[list(c)] = ci

            a = m[self.segmentation[ymin:ymax+1, xmin:xmax+1]]
            a[~self.mask[ymin:ymax+1, xmin:xmax+1]] = -1
        #     a = -1*np.ones_like(segmentation)
        #     for ci, c in enumerate(clusters):
        #         for i in c:
        #             a[segmentation == i] = ci

            vis = label2rgb(a, image=segmentation_viz[ymin:ymax+1, xmin:xmax+1])

        # vis = img_as_ubyte(vis[..., ::-1])
        vis = img_as_ubyte(vis)

        if show_cluster_indices:
            if not hasattr(self, 'sp_centroids'):
                self.sp_centroids = self.load_pipeline_result('spCentroids')

            for ci, cl in enumerate(clusters):
                cluster_center_yx = self.sp_centroids[cl].mean(axis=0).astype(np.int)
                cv2.putText(vis, str(ci), tuple(cluster_center_yx[::-1] - np.array([10,-10])), 
                            cv2.FONT_HERSHEY_DUPLEX, 1., ((0,255,255)), 1)

                # for i, sp in enumerate(c):
                #     vis = cv2.putText(vis, str(i), 
                #                       tuple(np.floor(sp_properties[sp, [1,0]] - np.array([10,-10])).astype(np.int)), 
                #                       cv2.FONT_HERSHEY_DUPLEX,
                #                       1., ((0,255,255)), 1)
        
        return vis.copy()


def display(vis, filename='tmp.jpg'):
    
    if vis.dtype != np.uint8:
        imsave(filename, img_as_ubyte(vis))
    else:
        imsave(filename, vis)
            
    from IPython.display import FileLink
    return FileLink(filename)

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
    '''
    h1s is n x n_texton
    MUST be float type
    '''    
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
