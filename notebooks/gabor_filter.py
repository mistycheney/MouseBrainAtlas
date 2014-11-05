# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%load_ext autoreload
%autoreload 2

# <codecell>

from utilities import *

if 'SSH_CONNECTION' in os.environ:
    DATA_DIR = '/home/yuncong/DavidData'
    REPO_DIR = '/home/yuncong/Brain'
else:
    DATA_DIR = '/home/yuncong/BrainLocal/DavidData_v4'
    REPO_DIR = '/home/yuncong/BrainSaliencyDetection'

dm = DataManager(DATA_DIR, REPO_DIR)

# import argparse
# import sys

# parser = argparse.ArgumentParser(
# formatter_class=argparse.RawDescriptionHelpFormatter,
# description='Execute feature extraction pipeline',
# epilog="""
# The following command processes image RS141_x5_0001.tif using blueNissl for both gabor parameters and segmentation parameters.
# python %s RS141 x5 1 -g blueNissl -s blueNissl
# """%(os.path.basename(sys.argv[0]), ))

# parser.add_argument("stack_name", type=str, help="stack name")
# parser.add_argument("resolution", type=str, help="resolution string")
# parser.add_argument("slice_ind", type=int, help="slice index")
# parser.add_argument("-g", "--gabor_params_id", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNissl')
# parser.add_argument("-s", "--segm_params_id", type=str, help="segmentation parameters id (default: %(default)s)", default='blueNissl')
# args = parser.parse_args()

class args:
    stack_name = 'RS141'
    resolution = 'x5'
    slice_ind = 1
    gabor_params_id = 'blueNisslWide'
    segm_params_id = 'blueNissl'

# <codecell>

dm.set_image(args.stack_name, args.resolution, args.slice_ind)
dm.set_gabor_params(gabor_params_id=args.gabor_params_id)
dm.set_segmentation_params(segm_params_id=args.segm_params_id)

# <codecell>

# @timeit
# def build_gabor_kernels(gabor_params):
#     """
#     Generate the Gabor kernels
#     """

from skimage.filter import gabor_kernel

theta_interval = dm.gabor_params['theta_interval']
n_angle = int(180/theta_interval)
freq_step = dm.gabor_params['freq_step']
freq_max = 1./dm.gabor_params['min_wavelen']
freq_min = 1./dm.gabor_params['max_wavelen']
bandwidth = dm.gabor_params['bandwidth']
n_freq = int(np.log(freq_max/freq_min)/np.log(freq_step)) + 1
frequencies = freq_max/freq_step**np.arange(n_freq)
angles = np.arange(0, n_angle)*np.deg2rad(theta_interval)

kernels = [gabor_kernel(f, theta=t, bandwidth=bandwidth) for f in frequencies for t in angles]
kernels = map(np.real, kernels)

n_kernel = len(kernels)

print 'num. of kernels: %d' % (n_kernel)
print 'frequencies:', frequencies
print 'wavelength (pixels):', 1/frequencies

max_kern_size = np.max([kern.shape[0] for kern in kernels])
print 'max kernel matrix size:', max_kern_size

# <codecell>

biases = np.array([k.sum() for k in kernels])
mean_bias = biases.mean()
kernels = [k/k.sum()*mean_bias for k in kernels]

dm.save_pipeline_result(kernels, 'kernels', 'pkl')

# <codecell>

# def gabor_filter():

from joblib import Parallel, delayed
from scipy.signal import fftconvolve

try:
    features = dm.load_pipeline_result('features', 'npy')
    
except Exception as e:

    def convolve_per_proc(i):
        return fftconvolve(dm.image, kernels[i], 'same').astype(np.half)

    filtered = Parallel(n_jobs=16)(delayed(convolve_per_proc)(i) 
                            for i in range(n_kernel))

    
    features = np.empty((dm.image_height, dm.image_width, n_kernel), dtype=np.half)
    for i in range(n_kernel):
        features[...,i] = filtered[i]

    del filtered

    dm.save_pipeline_result(features, 'features', 'npy')

n_feature = features.shape[-1]

def crop_borders(data):
    cropped_data = data[max_kern_size/2:-max_kern_size/2, max_kern_size/2:-max_kern_size/2, ...].copy()
    return cropped_data

# crop borders

try:
    cropped_features = dm.load_pipeline_result('cropFeatures', 'npy')
except:
    cropped_features = crop_borders(features)
    dm.save_pipeline_result(cropped_features, 'cropFeatures', 'npy')

try:
    cropped_img = dm.load_pipeline_result('cropImg', 'tif')    
except:
    cropped_img = crop_borders(dm.image)
    dm.save_pipeline_result(cropped_img, 'cropImg', 'tif')

try:
    cropped_mask = dm.load_pipeline_result('cropMask', 'npy')
except:
    cropped_mask = crop_borders(dm.mask)
    dm.save_pipeline_result(cropped_mask, 'cropMask', 'npy')
    dm.save_pipeline_result(cropped_mask, 'cropMask', 'tif')

cropped_height, cropped_width = cropped_img.shape[:2]
print cropped_height, cropped_width

# <codecell>

# Over-segment the image into superpixels using SLIC (http://ivrg.epfl.ch/research/superpixels)

from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_ubyte

img_rgb = gray2rgb(dm.image)

try:
    segmentation = dm.load_pipeline_result('segmentation', 'npy')
    
except Exception as e:
    segmentation = slic(img_rgb, n_segments=int(dm.segm_params['n_superpixels']), 
                        max_iter=10, 
                        compactness=float(dm.segm_params['slic_compactness']), 
                        sigma=float(dm.segm_params['slic_sigma']), 
                        enforce_connectivity=True)
    print 'segmentation computed'
    
    dm.save_pipeline_result(segmentation.astype(np.int16), 'segmentation', 'npy')

# <codecell>

from skimage.segmentation import relabel_sequential

try:
    cropped_segmentation_relabeled = dm.load_pipeline_result('cropSegmentation', 'npy')
except:
    # segmentation starts from 0
    cropped_segmentation = crop_borders(segmentation)
    n_superpixels = len(np.unique(cropped_segmentation))
    cropped_segmentation[~cropped_mask] = -1
    cropped_segmentation_relabeled, fw, inv = relabel_sequential(cropped_segmentation + 1)

    # make background label -1
    cropped_segmentation_relabeled -= 1
    dm.save_pipeline_result(cropped_segmentation_relabeled, 'cropSegmentation', 'npy')

# <codecell>

sp_props = regionprops(cropped_segmentation_relabeled + 1, intensity_image=cropped_img, cache=True)

def obtain_props_worker(i):
    return sp_props[i].centroid, sp_props[i].area, sp_props[i].mean_intensity, sp_props[i].bbox

r = Parallel(n_jobs=16)(delayed(obtain_props_worker)(i) for i in range(len(sp_props)))
sp_centroids = np.array([s[0] for s in r])
sp_areas = np.array([s[1] for s in r])
sp_mean_intensity = np.array([s[2] for s in r])
sp_bbox = np.array([s[3] for s in r])

sp_properties = np.column_stack([sp_centroids, sp_areas, sp_mean_intensity, sp_bbox])

dm.save_pipeline_result(sp_properties, 'cropSpProps', 'npy')

n_superpixels = len(np.unique(cropped_segmentation_relabeled))

img_superpixelized = mark_boundaries(gray2rgb(cropped_img), cropped_segmentation_relabeled)
img_superpixelized_text = img_as_ubyte(img_superpixelized)

# background label (-1) is not displayed
for s in range(n_superpixels - 1):
    img_superpixelized_text = cv2.putText(img_superpixelized_text, str(s), 
                      tuple(np.floor(sp_centroids[s][::-1]).astype(np.int)), 
                      cv2.FONT_HERSHEY_COMPLEX_SMALL,
                      .5, ((255,0,255)), 1)

dm.save_pipeline_result(img_superpixelized_text, 'cropSegmentation', 'tif')

# <codecell>

# Compute neighbor lists and connectivity matrix

from skimage.morphology import disk
from skimage.filter.rank import gradient
# from scipy.sparse import coo_matrix

try:
    neighbors = dm.load_pipeline_result('neighbors', 'npy')

except:

    edge_map = gradient(cropped_segmentation_relabeled.astype(np.uint8), disk(3))
    neighbors = [set() for i in range(n_superpixels)]

    for y,x in zip(*np.nonzero(edge_map)):
        neighbors[cropped_segmentation_relabeled[y,x]] |= set(cropped_segmentation_relabeled[y-2:y+2,x-2:x+2].ravel())

    for i in range(n_superpixels):
        neighbors[i] -= set([i])

    dm.save_pipeline_result(neighbors, 'neighbors', 'npy')

# <codecell>


