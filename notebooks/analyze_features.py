# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%load_ext autoreload
%autoreload 2

%autosave 60

# <codecell>

from utilities import *

if 'SSH_CONNECTION' in os.environ:
    DATA_DIR = '/home/yuncong/DavidData'
    REPO_DIR = '/home/yuncong/Brain'
else:
    DATA_DIR = '/home/yuncong/BrainLocal/DavidData_4'
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

raw_kernels = [gabor_kernel(f, theta=t, bandwidth=bandwidth) for f in frequencies for t in angles]
raw_kernels = map(np.real, raw_kernels)

n_kernel = len(raw_kernels)

print 'num. of kernels: %d' % (n_kernel)
print 'frequencies:', frequencies
print 'wavelength (pixels):', 1/frequencies

max_kern_size = np.max([kern.shape[0] for kern in raw_kernels])
print 'max kernel matrix size:', max_kern_size

# <codecell>

# compensate the numerical biases of kernels

biases = np.array([k.sum() for k in raw_kernels])

plt.stem(range(n_kernel), biases)
plt.title('biases before adjustment')
plt.show()

mean_bias = biases.mean()
kernels = [k/k.sum()*mean_bias for k in raw_kernels]

biases_adjusted = np.array([k.sum() for k in kernels])
plt.stem(range(n_kernel), biases_adjusted)
plt.title('biases after adjustment')
plt.show()

# <codecell>

fig, axes = plt.subplots(n_freq, n_angle, figsize=(20,20))

for i in range(n_freq):
    for j in range(n_angle):
        axes[i,j].matshow(kernels[i*n_angle + j])
        if i == n_freq - 1:
            axes[i,j].set_xlabel('%d degrees'%np.rad2deg(angles[j]))
        if j == 0:
            axes[i,j].set_ylabel('%d pixels'%(1./frequencies[i]))
        
#         plt.title('kernel %d'%i)
tight_layout()

# <codecell>

mins = np.empty((n_kernel, ))
maxs = np.empty((n_kernel, ))
means = np.empty((n_kernel, ))
for i in range(n_kernel):
    a = cropped_features[:,:,i].astype(np.float32)
    mins[i] = a.min()
    maxs[i] = a.max()
    means[i] = a.mean()
    
# plt.bar(range(n_kernel), mins, color='r')
# plt.bar(range(n_kernel), maxs, color='b')
# plt.bar(range(n_kernel), means, color='g')

plt.errorbar(range(n_kernel), means, yerr=[abs(mins-means), maxs-means], fmt='--o')

# <codecell>

# cropped_segmentation = dm.load_pipeline_result('cropSegmentation', 'npy')
# n_superpixels = len(unique(cropped_segmentation)) - 1

cropped_features = dm.load_pipeline_result('cropFeatures', 'npy').astype(np.float)
cropped_height, cropped_width = cropped_features.shape[:2]

cropped_mask = dm.load_pipeline_result('cropMask', 'npy')

cropped_features_tabular = np.reshape(cropped_features, (cropped_height, cropped_width, n_freq, n_angle))

# <codecell>

max_freqs, max_angles = np.unravel_index(cropped_features.argmax(axis=2), (n_freq, n_angle))
max_responses = cropped_features.max(axis=2)
max_mean_ratio = max_responses/cropped_features.mean(axis=2)

# <codecell>

plt.hist(max_mean_ratio.flat, bins=np.linspace(1.,2.,100))
plt.xlabel('max-mean ratio')
plt.title('distribution of max-mean ratios')
plt.show()

# <codecell>

plt.matshow(cropped_features_tabular[497, 3651])

plt.xticks(range(n_angle))
xlabels = ['%d'%np.rad2deg(a) for a in angles]
plt.gca().set_xticklabels(xlabels)
plt.xlabel('angle (degrees)')
# 0 degree corresponds to vertical strips

plt.yticks(range(n_freq))
ylabels = ['%.1f'%a for a in 1./frequencies]
plt.gca().set_yticklabels(ylabels)
plt.ylabel('wavelength (pixels)')

plt.colorbar()

plt.show()

# <codecell>

max_row, max_col = np.unravel_index(cropped_features_tabular[493, 3653].argmax(), (n_freq, n_angle))
print 'wavelength =', 1./frequencies[max_row], 'angle =', rad2deg(angles[max_col])

# <codecell>

fig, axes = plt.subplots(n_freq, figsize=(5,10), sharex=True)
plt.xlabel('angle')

for i in range(n_freq):
    axes[i].bar(range(n_angle), cropped_features_tabular[497, 3651, i, :])
    axes[i].set_title('wavelength = %d (pixels)' % (1./frequencies[i]))
    
plt.xticks(range(n_angle))
xlabels = ['%d'%np.rad2deg(a) for a in angles]
plt.gca().set_xticklabels(xlabels)
plt.xlabel('angle (degrees)')
    
tight_layout()

# <codecell>

synthesized_strips = np.ones((100,100))
for i in range(0, 100, 10):
    synthesized_strips[i:i+3] = 0
    
from skimage.transform import rotate
    
synthesized_strips = rotate(synthesized_strips, 45)
    
plt.imshow(synthesized_strips, cmap=plt.cm.Greys_r)

# <codecell>

from scipy.signal import fftconvolve

def convolve_per_proc(i):
    return fftconvolve(synthesized_strips, kernels[i], 'same').astype(np.half)

filtered = Parallel(n_jobs=16)(delayed(convolve_per_proc)(i) 
                        for i in range(n_kernel))

# <codecell>

fig, axes = plt.subplots(n_freq, n_angle, figsize=(20,20))

for i in range(n_freq):
    for j in range(n_angle):
        axes[i,j].imshow(filtered[i*n_angle + j], cmap=plt.cm.Greys_r)
        axes[i,j].set_xlabel('%d degrees'%np.rad2deg(angles[j]))
        axes[i,j].set_ylabel('%d pixels'%(1./frequencies[i]))
        axes[i,j].set_title('max %.2f, min %.2f '%(filtered[i*n_angle + j].max(), filtered[i*n_angle + j].min()))
#         plt.title('kernel %d'%i)
tight_layout()

# <codecell>

max_responses = np.reshape([f.max() for f in filtered], (n_freq, n_angle))

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

synthesized_rods = np.ones((100,100))
for i in range(0, 100, 20):
    synthesized_rods[i:i+5] = 0
for i in range(0, 100, 5):
    synthesized_rods[:, i:i+2] = 1
    
    
# from skimage.transform import rotate
# synthesized_rods = rotate(synthesized_rods, 45)
    
plt.imshow(synthesized_rods, cmap=plt.cm.Greys_r)

# <codecell>

from scipy.signal import fftconvolve

def convolve_per_proc(i):
    return fftconvolve(synthesized_rods, kernels[i], 'same').astype(np.half)

filtered = Parallel(n_jobs=16)(delayed(convolve_per_proc)(i) 
                        for i in range(n_kernel))

# <codecell>

fig, axes = plt.subplots(n_freq, n_angle, figsize=(20,20))

for i in range(n_freq):
    for j in range(n_angle):
        axes[i,j].imshow(filtered[i*n_angle + j], cmap=plt.cm.Greys_r)
        axes[i,j].set_xlabel('%d degrees'%np.rad2deg(angles[j]))
        axes[i,j].set_ylabel('%d pixels'%(1./frequencies[i]))
        axes[i,j].set_title('max %.2f, min %.2f '%(filtered[i*n_angle + j].max(), filtered[i*n_angle + j].min()))
#         plt.title('kernel %d'%i)
tight_layout()

# <codecell>

max_responses = np.reshape([f.max() for f in filtered], (n_freq, n_angle))

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

