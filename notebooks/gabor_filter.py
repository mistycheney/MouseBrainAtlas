# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from utilities import *

class args:
    stack_name = 'RS141'
    resolution = 'x5'
    slice = 4

# <codecell>

paramset = ParameterSet(1, 2, 4)

# Generate Gabor filter kernels
theta_interval = paramset.gabor_params['theta_interval']
n_angle = int(180/theta_interval)
freq_step = paramset.gabor_params['freq_step']
freq_max = 1./paramset.gabor_params['min_wavelen']
freq_min = 1./paramset.gabor_params['max_wavelen']
bandwidth = paramset.gabor_params['bandwidth']
n_freq = int(np.log(freq_max/freq_min)/np.log(freq_step)) + 1
frequencies = freq_max/freq_step**np.arange(n_freq)

kernels = [gabor_kernel(f, theta=t, bandwidth=bandwidth) for f in frequencies 
          for t in np.arange(0, n_angle)*np.deg2rad(theta_interval)]
kernels = map(np.real, kernels)

n_kernel = len(kernels)

print '=== filter using Gabor filters ==='
print 'num. of kernels: %d' % (n_kernel)
print 'frequencies:', frequencies
print 'wavelength (pixels):', 1/frequencies

max_kern_size = np.max([kern.shape[0] for kern in kernels])
print 'max kernel matrix size:', max_kern_size

# Process the image using Gabor filters
try:
    raise IOError
#     features = load_array('features')
except IOError:
    
    def convolve_per_proc(i):
        return fftconvolve(img, kernels[i], 'same').astype(np.half)
    
    filtered = Parallel(n_jobs=16)(delayed(convolve_per_proc)(i) 
                            for i in range(n_kernel))

    features = np.empty((im_height, im_width, n_kernel), dtype=np.half)
    for i in range(n_kernel):
        features[...,i] = filtered[i]

    del filtered
    
#     save_array(features, 'features')

n_feature = features.shape[-1]

def crop_borders(data):
    return data[max_kern_size/2:-max_kern_size/2, max_kern_size/2:-max_kern_size/2, ...]
    
# crop borders
cropped_features = crop_borders(features)
cropped_img = crop_borders(img)
cropped_mask = crop_borders(mask)
cropped_height, cropped_width = cropped_img.shape[:2]

