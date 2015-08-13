import os

import sys
sys.path.append(os.environ['GORDON_REPO_DIR'] + '/pipeline_scripts')
from utilities2014 import *

import time

sys.path.append('/home/yuncong/project/opencv-2.4.9/release/lib/python2.7/site-packages')
import cv2

from scipy.spatial.distance import cdist, pdist, squareform
from scipy.cluster.hierarchy import average, fcluster, leaders, complete, single, dendrogram, ward

from joblib import Parallel, delayed

from skimage.color import gray2rgb
from skimage.util import img_as_float, pad
from skimage.morphology import disk
from skimage.filters.rank import gradient

from collections import defaultdict, Counter
from itertools import combinations, chain, product

import networkx
from networkx import from_dict_of_lists, dfs_postorder_nodes

import matplotlib.pyplot as plt

import warnings
from scipy.interpolate import RectBivariateSpline
from skimage.feature import peak_local_max

section_ind = int(sys.argv[1])

dm = DataManager(generate_hierarchy=False, stack='RS141', resol='x5', section=section_ind)
dm._load_image()

texton_hists = dm.load_pipeline_result('texHist', 'npy')
segmentation = dm.load_pipeline_result('segmentation', 'npy')
n_superpixels = len(np.unique(segmentation)) - 1
textonmap = dm.load_pipeline_result('texMap', 'npy')
n_texton = len(np.unique(textonmap)) - 1

texture_map = dm.load_pipeline_result('textureMap', 'npy')
Gmax = dm.load_pipeline_result('Gmax', 'npy')

thetas = np.linspace(-np.pi/4, np.pi/4, 9)
n_theta = len(thetas)
Rs = [np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]) for theta in thetas]

grid_spacing = (100,100)

n_sample = 5
r = np.linspace(0,1,n_sample) * 20

def evaluate_spline_partial(spline, s):
    return spline(s, range(dm.image_height)) # indirect call is faster than directly put spline() in delayed; don't know why
#         return spline.ev(*np.meshgrid(s, range(dm.image_height), indexing='ij'))


def compute_filter_response_at_points(pts, theta, t2, template_height, template_width, yc, xc):
    
    vs = np.empty((len(pts),), dtype=np.float)

    for i, (x, y) in enumerate(pts):
        yy = (y + lm_edges_tuple_rotated_versions[theta][:,3] - lm_bbox_dims_rotated_versions[theta][9]).astype(np.int)
        xx = (x + lm_edges_tuple_rotated_versions[theta][:,2] - lm_bbox_dims_rotated_versions[theta][8]).astype(np.int)

#         ymax = y + template_height - 1 - yc
#         ymin = y - yc
#         xmax = x + template_width - 1 - xc
#         xmin = x - xc

#         t1 = texture_map[ymin:ymax+1, xmin:xmax+1].reshape((-1,n_texton))    
#         chi2_dists = chi2s(t1, t2)

#         # I expect to see RuntimeWarnings in this block
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore", category=RuntimeWarning)
#             v = np.nanmean(chi2_dists)

# #         valid_ratio = 1 - np.count_nonzero(np.isnan(chi2_dists)) / float(len(chi2_dists))

#         if np.isnan(v):
#             v = 0

#         vs[i] = valid_ratio * np.exp(-v/.5)

        r = Gmax[yy, xx]
    
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            ys = np.maximum(np.minimum(yy[:,None] + int_texture_sampling_positions[:,:,0], dm.image_height-1), 0)
            xs = np.maximum(np.minimum(xx[:,None] + int_texture_sampling_positions[:,:,1], dm.image_width-1), 0)
            avg_int_texture = np.nanmean(texture_map[ys, xs], axis=1)
            cs_int = chi2s(lm_edges_textures[:,:n_texton], avg_int_texture)

            ys = np.maximum(np.minimum(yy[:,None] + ext_texture_sampling_positions[:,:,0], dm.image_height-1), 0)
            xs = np.maximum(np.minimum(xx[:,None] + ext_texture_sampling_positions[:,:,1], dm.image_width-1), 0)
            avg_ext_texture = np.nanmean(texture_map[ys, xs], axis=1)
            cs_ext = chi2s(lm_edges_textures[:,n_texton:], avg_ext_texture)

            vs[i] = np.nansum(r * (np.exp(-cs_int/.5) + np.exp(-cs_ext/.5)))
#         + np.exp(-v/.5)

    return vs


def find_lm(lm_ind):

    print 'landmark', lm_ind
    
    global lm_edges_tuple_rotated_versions
    global lm_bbox_dims_rotated_versions
    global lm_texture_template_rotated_versions
    global lm_edges_textures
    
    lm_texture_template_rotated_versions = [np.load('/home/yuncong/csd395/lm_texture_template_%d_orientation%d.npy'%(lm_ind, theta_i)) 
                                            for theta_i in range(n_theta)]
    lm_bbox_dims_rotated_versions = [np.load('/home/yuncong/csd395/lm_bbox_dims_%d_orientation%d.npy'%(lm_ind, theta_i)) 
                                            for theta_i in range(n_theta)]
    lm_edges_tuple_rotated_versions = [np.load('/home/yuncong/csd395/lm_edge_points_%d_orientation%d.npy'%(lm_ind, theta_i)) 
                                            for theta_i in range(n_theta)]
    lm_edges_textures = np.load('/home/yuncong/csd395/lm_edge_textures_%d.npy'%(lm_ind))
    
    global ext_texture_sampling_positions
    global int_texture_sampling_positions
    
    #=============================================================

    vs_max = np.zeros((dm.image_height, dm.image_width))
    vs_argmax = np.zeros((dm.image_height, dm.image_width), np.uint8)

    vs_max_all_angles  =[]

    for theta in range(n_theta):

        print 'theta', theta

#         b = time.time()
        
        ext_texture_sampling_positions = (lm_edges_tuple_rotated_versions[theta][:,4:][:, None, ::-1] * r[None,:,None]).astype(np.int)
        int_texture_sampling_positions = - ext_texture_sampling_positions
        
        template_width, template_height = lm_bbox_dims_rotated_versions[theta][:2]
        xc, yc = lm_bbox_dims_rotated_versions[theta][8:10]

        ys, xs = np.mgrid[yc : dm.image_height + yc - template_height : grid_spacing[0], 
                          xc : dm.image_width + xc - template_width : grid_spacing[1]]

        t2 = lm_texture_template_rotated_versions[theta].reshape((-1,n_texton))  

#         V =  Parallel(n_jobs=16)(delayed(compute_filter_response_at_points)(s, theta) 
#                                 for s in np.array_split(zip(xs.flat, ys.flat), 16))
        
        
        V =  Parallel(n_jobs=16)(delayed(compute_filter_response_at_points)(s, theta,
                                        t2, template_height, template_width, yc, xc) 
                                for s in np.array_split(zip(xs.flat, ys.flat), 16))
           

    #     V = []
    #     for s in np.array_split(zip(xs.flat, ys.flat), 16):
    #         q = time.time()
    #         V.append(compute_filter_response_at_points(s, theta))
    #         print time.time() - q

        vs = np.concatenate(V)
        vss = np.reshape(vs, xs.shape)

        spline = RectBivariateSpline(range(xc, dm.image_width + xc - template_width, grid_spacing[1]), 
                                     range(yc, dm.image_height + yc - template_height, grid_spacing[0]),
                                     vss.T, bbox=[0, dm.image_width-1, 0, dm.image_height-1])

        xmax = xs.max()
        ymax = ys.max()
        xmin = xs.min()
        ymin = ys.min()

        res = Parallel(n_jobs=16)(delayed(evaluate_spline_partial)(spline, s) 
                                  for s in np.array_split(range(dm.image_width), 16))
        vs_i = np.vstack(res).T
        vs_i[~dm.mask] = 0.
        vs_i[ymax+1:dm.image_height] = 0.
        vs_i[:ymin] = 0.
        vs_i[:, xmax+1:dm.image_width] = 0.
        vs_i[:, :xmin] = 0.

    #         vs_argmax[vs_i > vs_max] = theta

        vs_max_all_angles.append(vs_i)

#         dm.save_pipeline_result(vs_i, 'responseMapLm%dTheta%d'%(lm_ind, theta), 'npy')
    #         vs_max = np.maximum(vs_max, vs_i)

#         print time.time() - b

        #################

    top3_locs = []
    for theta, vs_max in enumerate(vs_max_all_angles):
        vs_max_smooth = gaussian_filter(vs_max, sigma=10)

        peaks = peak_local_max(vs_max_smooth)
        ypeaks = peaks[:,0]
        xpeaks = peaks[:,1]

        order = np.argsort(vs_max_smooth[ypeaks, xpeaks])[::-1]
        ypeaks = ypeaks[order]
        xpeaks = xpeaks[order]

        for y, x in zip(ypeaks, xpeaks)[:3]:
            top3_locs.append((y, x, vs_max[y, x], theta))

    top3_locs = sorted(top3_locs, key=lambda x: x[2], reverse=True)[:3]
    dm.save_pipeline_result(np.array(top3_locs), 'responsePeaksLm%d'%lm_ind, 'npy')
    
    
for lm_ind in range(6):
    find_lm(lm_ind)