# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%load_ext autoreload
%autoreload 2

%autosave 0

# <codecell>

from utilities import *

from preamble import *

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

x0 = 4500
y0 = 2000
w = 20
h = 20

fig, axes = plt.subplots(h, w, figsize=(20,20), facecolor='white')

patch_min = cropped_features[y0:y0+h, x0:x0+w].min()
patch_max = cropped_features[y0:y0+h, x0:x0+w].max()

for i in range(h):
    for j in range(w):
        axes[i, j].matshow(cropped_features[i+y0, j+x0].reshape(n_freq, n_angle), vmin=patch_min, vmax=patch_max)
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
        
plt.tight_layout()

# cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
# fig.colorbar(im, cax=cax)
# plt.savefig('patch_features.png', bbox_inches='tight')

plt.show()

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
    
    
from skimage.transform import rotate
synthesized_rods = rotate(synthesized_rods, 45)
    
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

