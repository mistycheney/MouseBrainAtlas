# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%load_ext autoreload
%autoreload 2

%autosave 0

# <codecell>

from utilities import *

if 'SSH_CONNECTION' in os.environ:
    DATA_DIR = '/home/yuncong/DavidData'
    REPO_DIR = '/home/yuncong/Brain'
else:
    DATA_DIR = '/home/yuncong/BrainLocal/DavidData_v4'
    REPO_DIR = '/home/yuncong/Brain'

dm = DataManager(DATA_DIR, REPO_DIR)

# import argparse
# import sys

# parser = argparse.ArgumentParser(
# formatter_class=argparse.RawDescriptionHelpFormatter,
# description='Execute feature extraction pipeline',
# epilog="""
# The following command processes image RS141_x5_0001.tif using blueNissl for both gabor parameters and segmentation parameters.
# python %s RS141 x5 1 -g blueNissl -s blueNissl -v blueNissl
# """%(os.path.basename(sys.argv[0]), ))

# parser.add_argument("stack_name", type=str, help="stack name")
# parser.add_argument("resolution", type=str, help="resolution string")
# parser.add_argument("slice_ind", type=int, help="slice index")
# parser.add_argument("-g", "--gabor_params_id", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNissl')
# parser.add_argument("-s", "--segm_params_id", type=str, help="segmentation parameters id (default: %(default)s)", default='blueNissl')
# parser.add_argument("-v", "--vq_params_id", type=str, help="vq parameters id (default: %(default)s)", default='blueNissl')
# args = parser.parse_args()

class args:
    stack_name = 'RS141'
    resolution = 'x5'
    slice_ind = 1
    gabor_params_id = 'blueNisslWide'
    segm_params_id = 'blueNissl'
    vq_params_id = 'blueNissl'

# <codecell>

dm.set_image(args.stack_name, args.resolution, args.slice_ind)
dm.set_gabor_params(gabor_params_id=args.gabor_params_id)
dm.set_segmentation_params(segm_params_id=args.segm_params_id)
dm.set_vq_params(vq_params_id=args.vq_params_id)

from joblib import Parallel, delayed

n_texton = int(dm.vq_params['n_texton'])

theta_interval = dm.gabor_params['theta_interval']
n_angle = int(180/theta_interval)
freq_step = dm.gabor_params['freq_step']
freq_max = 1./dm.gabor_params['min_wavelen']
freq_min = 1./dm.gabor_params['max_wavelen']
n_freq = int(np.log(freq_max/freq_min)/np.log(freq_step)) + 1
frequencies = freq_max/freq_step**np.arange(n_freq)
angles = np.arange(0, n_angle)*np.deg2rad(theta_interval)

kernels = dm.load_pipeline_result('kernels', 'pkl')
n_kernel = len(kernels)

max_kern_size = max([k.shape[0] for k in kernels])

# <codecell>

cropped_image = dm.load_pipeline_result('cropImg', 'tif')

# <codecell>

x = 3855
y = 1005
h = 531
w = 1245
image_patch = cropped_image[y:y+h, x:x+w]

import matplotlib.pyplot as plt
plt.imshow(image_patch, cmap=plt.cm.Greys_r)
plt.show()

# <codecell>

image_patch_gaussian = gaussian_filter(image_patch, sigma=3)
plt.imshow(image_patch_gaussian, cmap=plt.cm.Greys_r)
plt.show()

# <codecell>

from scipy.signal import fftconvolve

features = np.empty((image_patch_gaussian.shape[0], image_patch_gaussian.shape[1], n_kernel), dtype=np.half)
for i in range(n_kernel):
    features[...,i] = fftconvolve(image_patch_gaussian, kernels[i], 'same')

cropped_features_patch = features[max_kern_size/2:-max_kern_size/2, max_kern_size/2:-max_kern_size/2, ...].copy()
cropped_features_patch = cropped_features_patch.astype(np.float)

# <codecell>

# max_responses = np.reshape([cropped_features_patch[:,:,i].max() for i in range(n_kernel)], (n_freq, n_angle))
max_responses = np.reshape([np.sort(cropped_features_patch[:,:,i].flat)[-1000:].mean() for i in range(n_kernel)], (n_freq, n_angle)).astype(np.float)

plt.matshow(max_responses)

plt.xticks(range(n_angle))
xlabels = ['%d'%np.rad2deg(a) for a in angles]
plt.gca().set_xticklabels(xlabels)
plt.xlabel('angle (degrees)')
# 0 degree corresponds to vertical strips

plt.yticks(range(n_freq))
ylabels = ['%.1f'%a for a in 1./frequencies]
plt.gca().set_yticklabels(ylabels)
plt.ylabel('wavelength (pixels)')

plt.title('max responses of all filters')

plt.colorbar()
plt.show()

# <codecell>

import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2,2)

    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask
    detected_peaks = local_max - eroded_background

    return detected_peaks

# <codecell>

rows, cols = np.where(detect_peaks(max_responses))
print zip(1./frequencies[rows], rad2deg(angles[cols]), max_responses[rows, cols])

# <codecell>


