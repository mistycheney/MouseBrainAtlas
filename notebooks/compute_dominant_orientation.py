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
    REPO_DIR = '/home/yuncong/BrainSaliencyDetection'

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

# <codecell>

cropped_segmentation = dm.load_pipeline_result('cropSegmentation', 'npy')
n_superpixels = len(unique(cropped_segmentation)) - 1

cropped_features = dm.load_pipeline_result('cropFeatures', 'npy').astype(np.float)
cropped_height, cropped_width = cropped_features.shape[:2]
cropped_mask = dm.load_pipeline_result('cropMask', 'npy')

cropped_image = dm.load_pipeline_result('cropImg', 'tif')

# <codecell>

cropped_features_tabular = np.reshape(cropped_features, (cropped_height, cropped_width, n_freq, n_angle))

# <codecell>

# x = 885
# y = 1437
# w = 960
# h = 435

# x = 348
# y = 1404
# w = 399
# h = 210

# x = 3591
# y = 419
# w = 327
# h = 150

# x = 3834
# y = 896
# w = 1357
# h = 382

# x = 4900
# y = 1025
# w = 474
# h = 391

x = 4685
y = 965
w = 706
h = 518

image_patch = cropped_image[y:y+h, x:x+w]
cropped_features_patch = cropped_features[y:y+h, x:x+w]

plt.imshow(image_patch, cmap=plt.cm.Greys_r)

# <codecell>

unravel_index(cropped_features_patch[:,:,59].argmax(), cropped_features_patch.shape[:2])

# <codecell>

fig, axes = plt.subplots(n_freq, n_angle, figsize=(100, 100))

for i in range(n_freq):
    for j in range(n_angle):
        axes[i,j].imshow(cropped_features_patch[:, :, i*n_angle + j], cmap=plt.cm.Greys_r)
        axes[i,j].set_xlabel('%d degrees'%np.rad2deg(angles[j]))
        axes[i,j].set_ylabel('%d pixels'%(1./frequencies[i]))
        axes[i,j].set_title('max %.2f, min %.2f '%(cropped_features_patch[:,:,i*n_angle + j].max(), 
                                                   cropped_features_patch[:,:,i*n_angle + j].min()))
tight_layout()

# <codecell>

fig, axes = plt.subplots(n_freq, n_angle, figsize=(100, 100))

for i in range(n_freq):
    for j in range(n_angle):
        axes[i,j].hist(cropped_features_patch[:,:,i*n_angle+j].flat, bins=np.linspace(20,40,100))

plt.show()

# <codecell>

# max_responses = np.reshape([np.sort(cropped_features_patch[:,:,i].flat)[:20].mean() for i in range(n_kernel)], (n_freq, n_angle))
max_responses = np.reshape([cropped_features_patch[:,:,i].max() for i in range(n_kernel)], (n_freq, n_angle))

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

peaks = np.where(detect_peaks(max_responses))
print peaks
max_responses[peaks]

# <codecell>

plt.matshow((max_responses > 20) & (max_freqs == 2) & (max_angles == 4), cmap=plt.cm.Greys_r)
plt.colorbar()

# <codecell>

max_freqs, max_angles = np.unravel_index(cropped_features.argmax(axis=2), (n_freq, n_angle))
max_responses = cropped_features.max(axis=2)
max_mean_ratio = max_responses/cropped_features.mean(axis=2)

# <codecell>

def worker(i):
    chosen = cropped_segmentation == i
    
    max_response_sp = max_responses[chosen].astype(np.float).max()
    max_dir_sp = np.bincount(max_angles[chosen]).argmax()
    max_freq_sp = np.bincount(max_freqs[chosen]).argmax()
    
    all_mmr = max_mean_ratio[chosen].astype(np.float)
    dominant_ratio_sp = np.count_nonzero(all_mmr > 1.02)/float(len(all_mmr))

    return max_dir_sp, max_freq_sp, max_response_sp, dominant_ratio_sp
    
res = Parallel(n_jobs=16)(delayed(worker)(i) for i in range(n_superpixels))
max_dir_sp, max_freq_sp, max_response_sp, dominant_ratio_sp = map(np.array, zip(*res))

# <codecell>

cropped_segmentation_vis = dm.load_pipeline_result('cropSegmentation', 'tif')
cropped_segmentation_vis2 = cropped_segmentation_vis.copy()
cropped_segmentation_vis2[~cropped_mask] = 0

# <codecell>

hc_colors = np.loadtxt('../visualization/high_contrast_colors.txt', skiprows=1)/255.

# <codecell>

# from skimage.color import hsv2rgb
# [hsv2rgb((i/n_freq, )) for i in range(n_freq) for j in range(n_angle)]

# <codecell>

max_response_sp_normalized = (max_response_sp - max_response_sp.min())/(max_response_sp.max() - max_response_sp.min())

from skimage.util import img_as_ubyte

# dirmap_vis2 = gray2rgb(cropped_segmentation_vis2.copy())
dirmap_vis2 = gray2rgb(np.zeros_like(cropped_segmentation, dtype=np.uint8))
dirmap_vis2 = img_as_ubyte(dirmap_vis2)

sp_properties = dm.load_pipeline_result('cropSpProps', 'npy')

for s in range(n_superpixels - 1):
#     if dominant_ratio_sp[s] < 0.2:
#         continue
    
    center = sp_properties[s, [1,0]].astype(np.int)
    angle = angles[max_dir_sp[s]]

    length = max_response_sp_normalized[s]*100
    end = center + np.array([length*np.sin(angle), -length*np.cos(angle)], dtype=np.int)
    cv2.line(dirmap_vis2, tuple(center), tuple(end), (255,0,0), thickness=5, lineType=8, shift=0)
    
#     length = int(1./frequencies[max_freq_sp[s]])
#     end = center + np.array([length*np.cos(angle), length*np.sin(angle)], dtype=np.int)
#     cv2.line(dirmap_vis2, tuple(center), tuple(end), (255,0,0), 
#              thickness=5, lineType=8, shift=0)
    
    
#     cv2.line(dirmap_vis2, tuple(center), tuple(end), tuple(map(int, hc_colors[max_freq_sp[s]+1]*255)), 
#              thickness=5, lineType=8, shift=0)
            

# <codecell>

dm.save_pipeline_result(dirmap_vis2, 'dirMap', 'tif')

# <codecell>

plt.imshow(dirmap_vis2, cmap=plt.cm.Greys_r)

