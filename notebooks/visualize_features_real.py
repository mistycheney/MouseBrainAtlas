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

# compensate the numerical biases of kernels

biases = np.array([k.sum() for k in raw_kernels])

mean_bias = biases.mean()
kernels = [k/k.sum()*mean_bias for k in raw_kernels]

# <codecell>

cropped_features = dm.load_pipeline_result('cropFeatures', 'npy').astype(np.float)
cropped_height, cropped_width = cropped_features.shape[:2]

cropped_mask = dm.load_pipeline_result('cropMask', 'npy')
cropped_image = dm.load_pipeline_result('cropImg', 'tif')

cropped_features_tabular = np.reshape(cropped_features, (cropped_height, cropped_width, n_freq, n_angle))

# <codecell>

# crop from real image

plt.imshow(cropped_image[y0:y0+h, x0:x0+w], cmap=plt.cm.Greys_r)

x0 = 4500
y0 = 2000
w = 20
h = 20

fig, axes = plt.subplots(h, w, figsize=(20,20), facecolor='white')

patch_min = cropped_features[y0:y0+h, x0:x0+w].min()
patch_max = cropped_features[y0:y0+h, x0:x0+w].max()

for i in range(h):
    for j in range(w):
        axes[i, j].matshow(cropped_features[i+y0, j+x0].reshape(n_freq, n_angle))
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
        
plt.tight_layout()

# plt.savefig('patch_features.png', bbox_inches='tight')
plt.show()

