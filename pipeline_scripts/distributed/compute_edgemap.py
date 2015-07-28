
import os
import argparse
import sys

sys.path.append(os.path.join(os.environ['GORDON_REPO_DIR'], 'pipeline_scripts'))

if os.environ['DATASET_VERSION'] == '2014':
	from utilities2014 import *
elif os.environ['DATASET_VERSION'] == '2015':
	from utilities import *

parser = argparse.ArgumentParser(
formatter_class=argparse.RawDescriptionHelpFormatter,
description='Compute edgemap')

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("slice_ind", type=int, help="slice index")
parser.add_argument("-g", "--gabor_params_id", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNisslWide')
parser.add_argument("-s", "--segm_params_id", type=str, help="segmentation parameters id (default: %(default)s)", default='blueNisslRegular')
parser.add_argument("-v", "--vq_params_id", type=str, help="vector quantization parameters id (default: %(default)s)", default='blueNissl')
args = parser.parse_args()


dm = DataManager(generate_hierarchy=False, stack=args.stack_name, resol='x5', section=args.slice_ind,
                 gabor_params_id=args.gabor_params_id, 
                 segm_params_id=args.segm_params_id, 
                 vq_params_id=args.vq_params_id)

#============================================================

from scipy.signal import convolve2d, fftconvolve
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

dm._load_image()

textonmap = dm.load_pipeline_result('texMap', 'npy')
textonmap_viz = dm.load_pipeline_result('texMap', 'png')
n_texton = len(np.unique(textonmap)) - 1

mys, mxs = np.where(dm.mask)
mys = mys.astype(np.int16)
mxs = mxs.astype(np.int16)

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
        pie_mask[ri][s] = np.zeros((2*r+1,2*r+1), np.bool).copy()
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
    
dm.save_pipeline_result(Gmax, 'Gmax', 'npy')

Gmax_viz = plt.cm.jet(Gmax/np.nanmax(Gmax));
dm.save_pipeline_result(Gmax_viz, 'Gmax', 'jpg')