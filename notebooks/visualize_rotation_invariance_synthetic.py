# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%load_ext autoreload
%autoreload 2

%autosave 0

from utilities import *

if 'SSH_CONNECTION' in os.environ:
    DATA_DIR = '/home/yuncong/DavidData'
    REPO_DIR = '/home/yuncong/Brain'
else:
    DATA_DIR = '/home/yuncong/BrainLocal/DavidData_4'
    REPO_DIR = '/home/yuncong/BrainSaliencyDetection'

dm = DataManager(DATA_DIR, REPO_DIR)

class args:
    stack_name = 'RS141'
    resolution = 'x5'
    slice_ind = 1
    gabor_params_id = 'blueNisslWide'
    segm_params_id = 'blueNissl'
    vq_params_id = 'blueNissl'
    
    
dm.set_image(args.stack_name, args.resolution, args.slice_ind)
dm.set_gabor_params(gabor_params_id=args.gabor_params_id)
dm.set_segmentation_params(segm_params_id=args.segm_params_id)
dm.set_vq_params(vq_params_id=args.vq_params_id)

from joblib import Parallel, delayed
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

# compensate the numerical biases of kernels

biases = np.array([k.sum() for k in raw_kernels])

mean_bias = biases.mean()
kernels = [k/k.sum()*mean_bias for k in raw_kernels]

# <codecell>

from skimage.transform import rotate

w = 10
h = 10

im_stripes = np.zeros((h, w))
for row in range(0, h, 4):
    im_stripes[row:row+2] = 1
    
ims = []
for theta in range(-45, 360, 45):
    im_stripes_rotated = rotate(im_stripes, -theta) > 0.5
    ims.append(im_stripes_rotated)
    
pattern = np.vstack([np.hstack(ims[:3]), np.hstack([ims[3], np.zeros((h, w)), ims[3]]), np.hstack(ims[2::-1])])

plt.imshow(pattern)

# <codecell>

from joblib import Parallel, delayed
from scipy.signal import fftconvolve

def convolve_per_proc(i):
    return fftconvolve(canvas, kernels[i], 'same').astype(np.half)

filtered = Parallel(n_jobs=16)(delayed(convolve_per_proc)(i) 
                        for i in range(n_kernel))

features = np.empty((canvas.shape[0], canvas.shape[1], n_kernel), dtype=np.half)
for i in range(n_kernel):
    features[...,i] = filtered[i]

del filtered

# <codecell>

features_tabular = features.reshape((features.shape[0]*features.shape[1], n_freq, n_angle))
# max_angle_indices = features_tabular.mean(axis=1).argmax(axis=-1)
max_angle_indices = features_tabular.max(axis=1).argmax(axis=-1)
features_rotated = np.reshape([np.roll(features_tabular[i], -ai, axis=-1) 
                               for i, ai in enumerate(max_angle_indices)], 
                              (features.shape[0], features.shape[1], n_freq * n_angle))

# <codecell>

# Visualize the feature of a specific pixel

x = 6
y = 3
plt.matshow(features[y, x].reshape(n_freq, n_angle))
print 'max angle index is', max_angle_indices[y * canvas.shape[0] + x]
plt.matshow(features_rotated[y, x].reshape(n_freq, n_angle))

# <codecell>

# Visualize features of the entire patch

height, width = pattern.shape

fig, axes = plt.subplots(height, width, figsize=(20,20), facecolor='white')

patch_min = features.min()
patch_max = features.max()

for i in range(height):
    for j in range(width):
        axes[i, j].matshow(features[i, j].reshape(n_freq, n_angle), vmin=patch_min, vmax=patch_max)
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
        
plt.tight_layout()

plt.savefig('patch_features.png', bbox_inches='tight')
plt.close(fig)

# <codecell>

fig, axes = plt.subplots(height, width, figsize=(20,20), facecolor='white')

patch_min = features_rotated.min()
patch_max = features_rotated.max()

for i in range(height):
    for j in range(width):
        axes[i, j].matshow(features_rotated[i, j].reshape(n_freq, n_angle), vmin=patch_min, vmax=patch_max)
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
        
plt.tight_layout()

plt.savefig('patch_features_rotated.png', bbox_inches='tight')
plt.close(fig)

