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

Gthresh = 1.
beta = 5.
Gmax = 1/(1+np.exp(-beta*(Gmax-Gthresh)))

coherence_map = dm.load_pipeline_result('coherenceMap', 'npy')
eigenvec_map = dm.load_pipeline_result('eigenvecMap', 'npy')


thetas = np.linspace(-np.pi/4, np.pi/4, 9)
n_theta = len(thetas)
Rs = [np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]) for theta in thetas]

grid_spacing = (10,10)

n_sample = 5
sample_portion = np.linspace(0,1,n_sample) * 20

def evaluate_spline_partial(spline, s):
    return spline(s, range(dm.image_height)) # indirect call is faster than directly put spline() in delayed; don't know why


def compute_filter_response_at_points(pts, theta, texture_lm, 
                                      template_height, template_width,
                                     centroid_local_y, centroid_local_x):
    
    n = len(pts)
    
    vs = np.empty((n,), dtype=np.float)
    vs_texture = np.empty((n,), dtype=np.float)
    vs_boundary = np.empty((n,), dtype=np.float)
    vs_striation = np.empty((n,), dtype=np.float)

    for i, (x, y) in enumerate(pts):

        if texture_lm is None:
            v_texture = np.nan
        else:
            bbox_ymax_global = y + template_height - 1 - centroid_local_y
            bbox_ymin_global = y - centroid_local_y
            bbox_xmax_global = x + template_width - 1 - centroid_local_x
            bbox_xmin_global = x - centroid_local_x

            texture_img = texture_map[bbox_ymin_global:bbox_ymax_global+1:10, 
                             bbox_xmin_global:bbox_xmax_global+1:10].reshape((-1,n_texton))

            chi2_dists = chi2s(texture_img, texture_lm)

            # I expect to see RuntimeWarnings in this block
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                v_texture = np.nansum(np.exp(-chi2_dists/.5)) / len(chi2_dists)
                if np.isnan(v_texture):
                    v_texture = 0

                    
        if lm_boundary_textures is None:
            v_boundary = np.nan
        else:
            vertices_local_ys = lm_boundary_vertices_rotated_versions[theta][:,1]
            vertices_local_xs = lm_boundary_vertices_rotated_versions[theta][:,0]

            vertices_global_ys = (vertices_local_ys + y - centroid_local_y).astype(np.int, copy=False)
            vertices_global_xs = (vertices_local_xs + x - centroid_local_x).astype(np.int, copy=False)

            try:
                r = Gmax[vertices_global_ys, vertices_global_xs]
            except Exception as e:
                print x, y, vertices_global_xs.max(axis=0), vertices_global_ys.max(axis=0), Gmax.shape
                raise e

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                
                int_xs = np.maximum(np.minimum(vertices_global_xs[:,None] + int_texture_sampling_positions[:,:,0], 
                                               dm.image_width-1), 0)
                int_ys = np.maximum(np.minimum(vertices_global_ys[:,None] + int_texture_sampling_positions[:,:,1],
                                               dm.image_height-1), 0)
                avg_int_texture = np.nanmean(texture_map[int_ys, int_xs], axis=1)
                int_texture_dists = chi2s(lm_boundary_textures[:,:n_texton], avg_int_texture)

                
                ext_xs = np.maximum(np.minimum(vertices_global_xs[:,None] + ext_texture_sampling_positions[:,:,0],
                                               dm.image_width-1), 0)
                ext_ys = np.maximum(np.minimum(vertices_global_ys[:,None] + ext_texture_sampling_positions[:,:,1], 
                                               dm.image_height-1), 0)
                avg_ext_texture = np.nanmean(texture_map[ext_ys, ext_xs], axis=1)
                ext_texture_dists = chi2s(lm_boundary_textures[:,n_texton:], avg_ext_texture)

                sigma = .5
                v_boundary = np.nansum(r * np.exp(-(int_texture_dists+ext_texture_dists)/(2*sigma)))
 
        if lm_striation_points_rotated_versions[theta] is None:
            v_striation = np.nan
        else:
            striation_points_global_ys = (lm_striation_points_rotated_versions[theta][:,1] + y - centroid_local_y).astype(np.int, copy=False)
            striation_points_global_xs = (lm_striation_points_rotated_versions[theta][:,0] + x - centroid_local_x).astype(np.int, copy=False)
            
            sample_vecs = eigenvec_map[striation_points_global_ys, striation_points_global_xs]
            vec_cosine_sims = np.abs(np.sum(sample_vecs * lm_striation_vecs_rotated_versions[theta], axis=1))
            v_striation = np.mean(vec_cosine_sims)
        
        vs_texture[i] = v_texture
        vs_boundary[i] = v_boundary
        vs_striation[i] = v_striation
        vs[i] = 5. * np.nan_to_num(v_texture) + np.nan_to_num(v_boundary) + np.nan_to_num(v_striation)

    return vs, vs_texture, vs_boundary, vs_striation

import cPickle as pickle

def find_peaks(vs_max_all_angles):
    
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
    
    return top3_locs  

def filter_image_with_landmark_descriptor(lm_ind):

# lm_ind = 5
    
    global lm_texture_template_rotated_versions
    global lm_bbox_rotated_versions
    global lm_boundary_vertices_rotated_versions
    global lm_boundary_textures
    global lm_striation_points_rotated_versions
    global lm_striation_vecs_rotated_versions
    
    global ext_texture_sampling_positions
    global int_texture_sampling_positions

    print 'landmark', lm_ind

    with open('/home/yuncong/csd395/all_landmark_descriptors_%d.pkl'%lm_ind, 'r') as f:
        landmark_descriptor = pickle.load(f)

    lm_texture_template_rotated_versions = map(itemgetter('texture_template'), landmark_descriptor)
    lm_bbox_rotated_versions = map(itemgetter('bbox'), landmark_descriptor)
    lm_boundary_vertices_rotated_versions = map(itemgetter('boundary_vertices'), landmark_descriptor)
    lm_boundary_textures = map(itemgetter('boundary_texture'), landmark_descriptor)[0]
    lm_vertice_normals = map(itemgetter('normal'), landmark_descriptor)
    lm_striations = map(itemgetter('striation'), landmark_descriptor)
    lm_striation_points_rotated_versions = [s[:,:2] if s is not None else None for s in lm_striations]
    lm_striation_vecs_rotated_versions = [s[:,2:] if s is not None else None for s in lm_striations]
        
    vs_max = np.zeros((dm.image_height, dm.image_width))
    vs_argmax = np.zeros((dm.image_height, dm.image_width), np.uint8)

    vs_max_all_angles = []
#     vs_texture_all_angles = []
#     vs_boundary_all_angles = []
#     vs_striation_all_angles = []

    for theta in range(n_theta):
    # theta = 4

        print 'theta', theta

        b = time.time()

        if lm_vertice_normals[theta] is not None:
        
            ext_texture_sampling_positions = (lm_vertice_normals[theta][:, None] * sample_portion[None,:,None]).astype(np.int)
            int_texture_sampling_positions = - ext_texture_sampling_positions

        template_width, template_height = lm_bbox_rotated_versions[theta][:2]
        centroid_local_x, centroid_local_y = lm_bbox_rotated_versions[theta][2:4]

        ys, xs = np.mgrid[centroid_local_y : dm.image_height  - (template_height - centroid_local_y) : grid_spacing[0], 
                          centroid_local_x : dm.image_width - (template_width - centroid_local_x) : grid_spacing[1]]

        #         print dm.image_width, dm.image_height, template_width, template_height, centroid_local_x, centroid_local_y

        if lm_texture_template_rotated_versions[theta] is not None:        
            t2 = lm_texture_template_rotated_versions[theta][::10,::10].reshape((-1,n_texton))  
        else:
            t2 = None
            
        res =  Parallel(n_jobs=16)(delayed(compute_filter_response_at_points)(s, theta, t2, 
                                          template_height, template_width,
                                         lm_bbox_rotated_versions[theta][3],
                                         lm_bbox_rotated_versions[theta][2]) 
                                for s in np.array_split(zip(xs.flat, ys.flat), 16))

        vs, vs_texture, vs_boundary, vs_striation = zip(*res)

        # V = []
        # for s in np.array_split(zip(xs.flat, ys.flat), 16):
        # #         q = time.time()
        #     V.append(compute_filter_response_at_points(s, theta, t2, template_height, template_width, yc, xc))
        # #         print time.time() - q

        vs = np.reshape(np.concatenate(vs), xs.shape)
        vs_texture = np.reshape(np.concatenate(vs_texture), xs.shape)
        vs_boundary = np.reshape(np.concatenate(vs_boundary), xs.shape)
        vs_striation = np.reshape(np.concatenate(vs_striation), xs.shape)

        spline = RectBivariateSpline(range(centroid_local_x, dm.image_width + centroid_local_x - template_width, grid_spacing[1]), 
                                     range(centroid_local_y, dm.image_height + centroid_local_y - template_height, grid_spacing[0]),
                                     vs.T, bbox=[0, dm.image_width-1, 0, dm.image_height-1])

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
                
        vs_i_normalized = vs_i/np.nansum(vs_i)
        
        vs_max_all_angles.append(vs_i_normalized)
#         vs_texture_all_angles.append(vs_texture)
#         vs_boundary_all_angles.append(vs_boundary)
#         vs_striation_all_angles.append(vs_striation)
        
        print time.time() - b

    dm.save_pipeline_result(np.asarray(vs_max_all_angles), 'responseMapLmAllRotations%d'%lm_ind, 'npy')
    
#     return vs_max_all_angles, vs_texture_all_angles, vs_boundary_all_angles, vs_striation_all_angles
    
landmark_indices = [0,5,6,7,8,9,10,11,12,13]
for lm_ind in landmark_indices:
    filter_image_with_landmark_descriptor(lm_ind)