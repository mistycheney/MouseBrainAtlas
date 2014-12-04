# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%load_ext autoreload
%autoreload 2

from preamble import *

# <codecell>

from skimage.util import pad

approx_bg_intensity = dm.image[10:20, 10:20].mean()
# approx_bg_intensity = 0

masked_image = dm.image.copy()
masked_image[~dm.mask] = approx_bg_intensity

# padded_image = pad(masked_image, max_kern_size, 'constant', constant_values=approx_bg_intensity)
padded_image = pad(masked_image, max_kern_size, 'linear_ramp', end_values=approx_bg_intensity)

plt.imshow(padded_image, cm.Greys_r)
plt.show()

# display(padded_image)

# <codecell>

from joblib import Parallel, delayed
from scipy.signal import fftconvolve

# try:
#     features = dm.load_pipeline_result('features', 'npy')
    
# except Exception as e:

def convolve_per_proc(i):
    return fftconvolve(padded_image, kernels[i], 'same').astype(np.half)

padded_filtered = Parallel(n_jobs=16)(delayed(convolve_per_proc)(i) 
                        for i in range(n_kernel))

filtered = [f[max_kern_size:-max_kern_size, max_kern_size:-max_kern_size] for f in padded_filtered]

#     features = np.empty((dm.image_height, dm.image_width, n_kernel), dtype=np.half)
#     for i in range(n_kernel):
#         features[...,i] = filtered[i]

features = np.empty((n_kernel, dm.image_height, dm.image_width), dtype=np.half)
for i in range(n_kernel):
    features[i, ...] = filtered[i]

del filtered

dm.save_pipeline_result(features, 'features', 'npy')

# <codecell>

# visualize a slice of feature responses

cropped_response = features[-1]
plt.matshow(cropped_response, cmap=cm.coolwarm)
plt.colorbar()
plt.show()

# <codecell>

from skimage.exposure import rescale_intensity
cropped_response_vis = rescale_intensity(cropped_response, out_range=(0, 255))
cropped_response_vis[~dm.mask] = 127
display(cropped_response_vis.astype(np.uint8))

