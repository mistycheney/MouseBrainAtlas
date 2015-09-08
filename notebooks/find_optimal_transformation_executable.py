import cPickle as pickle

import os
import sys

from utilities2014 import *

from itertools import product



landmark_indices = range(17)
n_landmark = len(landmark_indices)


atlas_landmark_desc = 'RS141_0001_yuncong_08212015001655'

all_lm_centroid_global = dict([])

for lm_ind in landmark_indices:
    with open('/home/yuncong/csd395/%s_landmark_centroid_global_%d.pkl'%(atlas_landmark_desc, lm_ind), 'r') as f:
        all_lm_centroid_global[lm_ind] = pickle.load(f)
    
dm = DataManager(generate_hierarchy=False, stack='RS141', resol='x5', section=int(sys.argv[1]))
dm._load_image()

response_maps_all_rotations = dict([])
for lm_ind in landmark_indices:
    response_maps_all_rotations[lm_ind] = dm.load_pipeline_result('responseMapLmAllRotations%d'%lm_ind, 'npy')
    

from skimage.feature import peak_local_max

def find_peaks(vs_max, topk=3):
    
    vs_max_smooth = gaussian_filter(vs_max, sigma=10)
#         vs_max_smooth = vs_max

    peaks = peak_local_max(vs_max_smooth)
    ypeaks = peaks[:,0]
    xpeaks = peaks[:,1]

    order = np.argsort(vs_max_smooth[ypeaks, xpeaks])[::-1]
    ypeaks = ypeaks[order]
    xpeaks = xpeaks[order]

    return np.array([(x, y, vs_max[y, x]) for y, x in zip(ypeaks, xpeaks)[:topk]])

def find_peaks_all_rotations(vs_max_all_angles, topk=3):
    
    topk_locs = []
    for theta, vs_max in enumerate(vs_max_all_angles):
        topk_locs += map(lambda t: t+(theta,), find_peaks(vs_max, topk))
        
    topk_locs = sorted(topk_locs, key=lambda x: x[2], reverse=True)[:topk]
    
    return topk_locs

origin_pos_canvas_atlas = DataManager(generate_hierarchy=False, stack='RS141', resol='x5', section=1).load_pipeline_result('originPosOnCanvas', 'npy')
origin_pos_canvas = dm.load_pipeline_result('originPosOnCanvas', 'npy')

image_centroid = np.array((int(dm.image_width/2), int(dm.image_height/2)))
atlas_centroid = np.array((2772, 1385))

xshifts = np.linspace(-400, 400, 100)
yshifts = np.linspace(-400, 400, 100)
angles = np.linspace(-np.pi/4, np.pi/4, 17)


# sample_lms = np.random.choice(range(n_landmark), n_landmark, replace=False)
# sample_lms = [0, 9]
sample_lms = landmark_indices

def rigid_transform_to(pts1, T):
    pts1_trans = np.dot(T, np.column_stack([pts1, np.ones((pts1.shape[0],))]).T).T
    pts1_trans = pts1_trans[:,:2]/pts1_trans[:,-1][:,np.newaxis]
    return pts1_trans

thetas = np.linspace(-np.pi/4, np.pi/4, 9)
n_theta = len(thetas)

X = [all_lm_centroid_global[i] for i in sample_lms]
Xc = X - atlas_centroid
probs = []
for xshift, yshift, angle in product(xshifts, yshifts, angles):
#     print xshift, yshift, angle
    T = np.array([[np.cos(angle), -np.sin(angle), xshift],
                    [np.sin(angle), np.cos(angle), yshift],
                    [0, 0, 1.]])
    TXc = rigid_transform_to(Xc, T).astype(np.int)
    TX = TXc + image_centroid
    
    theta_i = np.argmin(np.abs(thetas - angle))
    
    prob = np.sum(response_maps_all_rotations[lm][theta_i][TX[i,1], TX[i,0]] if TX[i,1] < dm.image_height and TX[i,1] >= 0 \
                                                        and TX[i,0] < dm.image_width and TX[i,0] >= 0\
                  else 0
                  for i, lm in enumerate(sample_lms))
    probs.append(prob)
    
probs_map = np.reshape(probs, (len(xshifts), len(yshifts), len(angles)))


xshift_opt_ind, yshift_opt_ind, rot_opt_ind = np.unravel_index(np.argmax(probs_map), probs_map.shape)

xshift_opt = xshifts[xshift_opt_ind]
yshift_opt = yshifts[yshift_opt_ind]
rot_opt = angles[rot_opt_ind]


theta_i_opt = np.argmin(np.abs(thetas - rot_opt))

R = np.array([[np.cos(rot_opt), -np.sin(rot_opt)],
                [np.sin(rot_opt), np.cos(rot_opt)]])

shift = image_centroid + (xshift_opt, yshift_opt) - np.dot(R, atlas_centroid)
T = np.vstack([np.column_stack([R, shift]), [0,0,1]])

dm.save_pipeline_result(T, 'T', 'npy')

new_centroids_global = [np.squeeze(rigid_transform_to(all_lm_centroid_global[lm_ind][None,:], T).astype(np.int)) 
                        for lm_ind in landmark_indices]

dm.save_pipeline_result(new_centroids_global, 'landmarkPositionsBeforeSnap', 'npy')

def move_landmark(lm_ind):
    
    with open('/home/yuncong/csd395/%s_landmark_boundary_vertices_all_polygons_global_%d.pkl'%(atlas_landmark_desc, lm_ind), 'r') as f:
        all_polygon_boundary_vertices = pickle.load(f)[theta_i_opt]
        
    with open('/home/yuncong/csd395/%s_landmark_textured_area_vertices_all_polygons_global_%d.pkl'%(atlas_landmark_desc, lm_ind), 'r') as f:
        all_polygon_textured_area_vertices = pickle.load(f)[theta_i_opt]

    with open('/home/yuncong/csd395/%s_landmark_striation_points_all_polygons_global_%d.pkl'%(atlas_landmark_desc, lm_ind), 'r') as f:
        all_polygon_striation_vertices = pickle.load(f)[theta_i_opt]

    new_centroid_global = new_centroids_global[lm_ind]

    new_landmark_descriptor = dict([])
    new_landmark_descriptor['bbox'] = np.nan * np.ones((10,))
    new_landmark_descriptor['bbox'][4:6] = new_centroid_global

    peaks = find_peaks(response_maps_all_rotations[lm_ind][theta_i_opt], topk=-1)[:,:2].astype(np.int)
    dists = np.squeeze(cdist([new_centroid_global], peaks))
    peak_indices_in_neighborhood = np.where(dists < 200)[0]
    if len(peak_indices_in_neighborhood) == 0:
        nearest_peak = new_centroid_global
    else:
    
    #     peak_scores = np.exp(-dists[peak_indices_in_neighborhood]**2/10**2) * \
    #             response_maps_all_rotations[lm_ind][theta_i_opt][peaks[peak_indices_in_neighborhood,1],
    #                                                             peaks[peak_indices_in_neighborhood,0]]

        peak_scores = response_maps_all_rotations[lm_ind][theta_i_opt][peaks[peak_indices_in_neighborhood,1],
                                                                peaks[peak_indices_in_neighborhood,0]]

        nearest_peak = peaks[peak_indices_in_neighborhood[np.argmax(peak_scores)]]

    if all_polygon_boundary_vertices is not None:
        new_landmark_descriptor['boundary_vertices_all_polygons_global'] = [rigid_transform_to(v, T).astype(np.int) - new_centroid_global + nearest_peak
                                                                             for v in all_polygon_boundary_vertices]
    else:
        new_landmark_descriptor['boundary_vertices_all_polygons_global'] = None

    if all_polygon_textured_area_vertices is not None:
        new_landmark_descriptor['textured_area_vertices_all_polygons_global'] = [rigid_transform_to(v, T).astype(np.int) - new_centroid_global + nearest_peak
                                                                                  for v in all_polygon_textured_area_vertices]
    else:
        new_landmark_descriptor['textured_area_vertices_all_polygons_global'] = None

    if all_polygon_striation_vertices is not None:
        new_landmark_descriptor['striation_vertices_all_polygons_global'] = [rigid_transform_to(v, T).astype(np.int) - new_centroid_global + nearest_peak
                                                                              for v in all_polygon_striation_vertices]
    else:
        new_landmark_descriptor['striation_vertices_all_polygons_global'] = None

    return new_landmark_descriptor


from joblib import Parallel, delayed
new_landmark_descriptors = Parallel(n_jobs=16)(delayed(move_landmark)(lm_ind) for lm_ind in landmark_indices)
new_landmark_descriptors = dict(zip(landmark_indices, new_landmark_descriptors))


import datetime
timestamp = datetime.datetime.now().strftime("%m%d%Y%H%M%S")

for lm_ind in landmark_indices:
    with open('/home/yuncong/csd395/%s_%s_%s_%s_landmark_descriptors_%d.pkl'%(
            dm.stack, dm.slice_str, 'hector', timestamp, lm_ind), 'w') as f:
        pickle.dump(new_landmark_descriptors[lm_ind], f)
        
        
colors = (np.loadtxt(os.environ['GORDON_REPO_DIR'] + '/visualization/100colors.txt', skiprows=1) * 255).astype(np.int)

from skimage.morphology import disk, binary_dilation
import cv2

viz = img_as_ubyte(dm.image_rgb)

for lm_ind, d in new_landmark_descriptors.iteritems():
        
    if d['boundary_vertices_all_polygons_global'] is not None:
        for vs in d['boundary_vertices_all_polygons_global']:
            for x, y in vs:
                cv2.circle(viz, (x,y), 10, colors[lm_ind], -1)
                        
    if d['textured_area_vertices_all_polygons_global'] is not None:
        for vs in d['textured_area_vertices_all_polygons_global']:
            for x, y in vs:
                cv2.circle(viz, (x,y), 10, colors[lm_ind], -1)
    
    if d['striation_vertices_all_polygons_global'] is not None:
        for vs in d['striation_vertices_all_polygons_global']:
            for x, y in vs:
                cv2.circle(viz, (x,y), 10, colors[lm_ind], -1)

dm.save_pipeline_result(viz, 'landmarkDetections', 'jpg')
        