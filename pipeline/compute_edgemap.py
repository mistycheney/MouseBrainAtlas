#! /usr/bin/env python

import os
import argparse
import sys
import time

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Compute gPb edgemap')

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("slice_ind", type=int, help="slice index")
parser.add_argument("-g", "--gabor_params_id", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNisslWide')
parser.add_argument("-s", "--segm_params_id", type=str, help="segmentation parameters id (default: %(default)s)", default='blueNisslRegular')
parser.add_argument("-v", "--vq_params_id", type=str, help="vector quantization parameters id (default: %(default)s)", default='blueNissl')
args = parser.parse_args()

from joblib import Parallel, delayed

sys.path.append(os.path.join(os.environ['GORDON_REPO_DIR'], 'notebooks'))
from utilities2015 import *

os.environ['GORDON_DATA_DIR'] = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_processed'
os.environ['GORDON_REPO_DIR'] = '/oasis/projects/nsf/csd395/yuncong/Brain'
os.environ['GORDON_RESULT_DIR'] = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_results'

dm = DataManager(data_dir=os.environ['GORDON_DATA_DIR'], repo_dir=os.environ['GORDON_REPO_DIR'], 
    result_dir=os.environ['GORDON_RESULT_DIR'], labeling_dir=os.environ['GORDON_LABELING_DIR'],
    stack=args.stack_name, section=args.slice_ind)

#============================================================

from scipy.signal import convolve2d, fftconvolve
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

dm._load_image()

textonmap = dm.load_pipeline_result('texMap', 'npy')
n_texton = textonmap.max() + 1

mys, mxs = np.where(dm.mask)
mys = mys.astype(np.int16)
mxs = mxs.astype(np.int16)


sys.stderr.write('compute pie histograms ...\n')
t = time.time()

n_theta = 12
theta_binwidth = 2*np.pi/n_theta

radius_list = [50,100,200]
n_radius = len(radius_list)
max_radius = radius_list[-1]

box_indices = np.dstack(np.meshgrid(np.arange(-max_radius, max_radius+1), 
                                    np.arange(-max_radius, max_radius+1))).astype(np.int16)
norms = np.linalg.norm(box_indices.astype(np.float), axis=-1)

pie_indices_r = []
circle_indices_r = []

for ri in range(n_radius):
    circle_indices = box_indices[norms <= radius_list[ri]]
    circle_indices_r.append(circle_indices)
    
    angles = np.arctan2(circle_indices[:,0], circle_indices[:,1])
    angles[angles < 0] += 2*np.pi
    angular_bins = np.int0(angles/theta_binwidth)
    pie_indices = [circle_indices[angular_bins==k] for k in range(n_theta)]
    pie_indices_r.append(pie_indices)
    
pie_mask = [[None for _ in range(n_theta)] for _ in radius_list]
for ri, r in enumerate(radius_list):
    for s in range(n_theta):
        pie_mask[ri][s] = np.zeros((2*r+1, 2*r+1), np.bool).copy()
        pie_mask[ri][s][r+pie_indices_r[ri][s][:,0], r+pie_indices_r[ri][s][:,1]] = 1
        
textonmap_individual_channels = [textonmap == c for c in range(n_texton)]

def compute_pie_histogram(pie_mask, ri, s):
    
    # notice the data type is set to uint16 here!
    h = np.zeros((dm.image_height, dm.image_width, n_texton), np.uint16)
    for c in range(n_texton):
        h[:,:,c] = fftconvolve(textonmap_individual_channels[c], pie_mask, mode='same').astype(np.uint16)
        
    dm.save_pipeline_result(h, 'histRadius%dAngle%d'%(ri,s), 'hdf')
    
for ri in range(n_radius):
    Parallel(n_jobs=16)(delayed(compute_pie_histogram)(pie_mask[ri][s], ri, s) for s in range(n_theta))

sys.stderr.write('done in %f seconds\n' % (time.time() - t))



sys.stderr.write('compute halfdisc histograms ...\n')
t = time.time()
    
H = np.zeros((n_radius, n_theta, dm.image_height, dm.image_width, n_texton), np.uint16)
for ri in range(n_radius):
    for s in range(n_theta):
        H[ri, s] = dm.load_pipeline_result('histRadius%dAngle%d'%(ri,s), 'hdf')
    
def compute_halfdisc_histogram_diff(H, start_bin, mys, mxs, r):
    
    n_theta, height, width = H.shape[1:4]
    
    Gs = np.zeros((height, width), np.float)

    first_half_bins = np.arange(start_bin, start_bin+n_theta/2)%n_theta
    second_half_bins = np.arange(start_bin+n_theta/2, start_bin+n_theta)%n_theta

    H_halfdisk1 = np.sum(H[r, first_half_bins[:,None], mys, mxs], axis=0).astype(np.float)
    H_halfdisk2 = np.sum(H[r, second_half_bins[:,None], mys, mxs], axis=0).astype(np.float)
        
    H_halfdisk1 /= H_halfdisk1.sum(axis=-1)[:,None]
    H_halfdisk2 /= H_halfdisk2.sum(axis=-1)[:,None]
    
    Gs[mys, mxs] = chi2s(H_halfdisk1, H_halfdisk2)
    
    dm.save_pipeline_result(Gs, 'GmaxRadius%dAngle%d'%(r, start_bin), 'npy')
    
    # return or not does not affect performance at all
    return Gs

Gmax = np.zeros((dm.image_height, dm.image_width))
for ri in range(n_radius):
    G = [compute_halfdisc_histogram_diff(H, s, mys, mxs, ri) for s in range(n_theta/2)]
    Gmax_r = np.max(G, axis=0)
    Gmax = np.maximum(Gmax_r, Gmax)
alpha = 1.5
Gmax = 1 - 1/(1+alpha*Gmax)

dm.save_pipeline_result(Gmax, 'Gmax', 'npy')

sys.stderr.write('done in %f seconds\n' % (time.time() - t))


sys.stderr.write('generating visualization ...\n')
t = time.time()

Gmax_viz = plt.cm.jet(Gmax/np.nanmax(Gmax))
dm.save_pipeline_result(Gmax_viz, 'Gmax', 'jpg')

sys.stderr.write('done in %f seconds\n' % (time.time() - t))


sys.stderr.write('non-maximum suppression ...\n')
t = time.time()

dy, dx = np.gradient(Gmax)

grad_norm = np.sqrt(dx**2+dy**2)
grad_unit_vec_y = np.nan_to_num(dy/grad_norm)
grad_unit_vec_x = np.nan_to_num(-dx/grad_norm)

gy = grad_unit_vec_y[mys, mxs]
gx = grad_unit_vec_x[mys, mxs]

a = np.arange(-20, 20)

neighborhood_y = np.outer(gy, a)
neighborhood_x = np.outer(gx, a)

global_neighborhood_y = (mys[:,None] + neighborhood_y).astype(np.int)
global_neighborhood_x = (mxs[:,None] + neighborhood_x).astype(np.int)

b = time.time()
global_neighborhood_y[global_neighborhood_y < 0] = 0
global_neighborhood_y[global_neighborhood_y >= dm.image_height] = dm.image_height-1
global_neighborhood_x[global_neighborhood_x < 0] = 0
global_neighborhood_x[global_neighborhood_x >= dm.image_width] = dm.image_width-1
print time.time()-b

# 60s, slower than above
# b = time.time()
# global_neighborhood_y = np.minimum(np.maximum(global_neighborhood_y.astype(np.int), 0), height)
# global_neighborhood_x = np.minimum(np.maximum(global_neighborhood_x.astype(np.int), 0), width)
# print time.time()-b

global_neighborhood_values = Gmax[global_neighborhood_y, global_neighborhood_x]
global_neighborhood_maximum = global_neighborhood_values.max(axis=1)

values = Gmax[mys, mxs]

is_local_maximum = global_neighborhood_maximum == values

mys_local_max = mys[is_local_maximum]
mxs_local_max = mxs[is_local_maximum]


G_nonmaxsup = np.zeros_like(Gmax)
G_nonmaxsup[mys_local_max, mxs_local_max] = Gmax[mys_local_max, mxs_local_max]
G_nonmaxsup[(grad_unit_vec_y==0)&(grad_unit_vec_x==0)] = 0

dm.save_pipeline_result(G_nonmaxsup, 'Gnonmaxsup', 'npy')

sys.stderr.write('done in %f seconds\n' % (time.time() - t))


sys.stderr.write('generating visualization ...\n')
t = time.time()

G_nonmaxsup_viz = plt.cm.jet(G_nonmaxsup/np.nanmax(G_nonmaxsup));
dm.save_pipeline_result(G_nonmaxsup_viz, 'Gnonmaxsup', 'jpg')

G_nonmaxsup_thresh_viz = dm.image_rgb.copy()

for y,x in zip(*np.where(G_nonmaxsup)):
    if Gmax[y,x] > .6:
        r = int(Gmax[y,x]*5)
        G_nonmaxsup_thresh_viz[y-r:y+r, x-r:x+r] = [Gmax[y,x], 0, 0]
        
dm.save_pipeline_result(G_nonmaxsup_thresh_viz, 'GnonmaxsupThresh', 'jpg')         
# display(G_nonmaxsup_thresh_viz)

G_nonmaxsup_thresh_viz = np.zeros_like(dm.image)

for y,x in zip(*np.where(G_nonmaxsup)):
    if Gmax[y,x] > .6:
        r = int(Gmax[y,x]*5)
#         G_nonmaxsup_thresh_viz[y-r:y+r, x-r:x+r] = Gmax[y,x]
        G_nonmaxsup_thresh_viz[y-r:y+r, x-r:x+r] = 1

G_nonmaxsup_thresh_viz = 1. - G_nonmaxsup_thresh_viz
        
dm.save_pipeline_result(G_nonmaxsup_thresh_viz, 'GnonmaxsupThreshBW', 'jpg')

display(G_nonmaxsup_thresh_viz)

sys.stderr.write('done in %f seconds\n' % (time.time() - t))