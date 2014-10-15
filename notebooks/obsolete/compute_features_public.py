# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import sys, os
from scipy.ndimage import gaussian_filter, measurements

from skimage.filter import threshold_otsu, threshold_adaptive, denoise_bilateral
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

from IPython.display import FileLink

import cv2

import time
import multiprocessing

# <codecell>

# img_name = 'Resized_1_region2'
img_name = 'Resized_6_region'
img = cv2.imread('/home/yuncong/my_csd181_scratch/'+img_name+'.png', 0)

def crop_image(img):
    blurred = gaussian_filter(img, 20)
    thresholded = blurred < threshold_otsu(blurred)
    # plot_figure(thresholded, title='', figsize=(10,10), cbar=False, ticks=None, cmap=cm.Greys_r)
    slc = measurements.find_objects(thresholded)[0]

    margin = 100
    xstart = max(slc[0].start - margin, 0)
    xstop = min(slc[0].stop + margin, img.shape[0])
    ystart = max(slc[1].start - margin, 0)
    ystop = min(slc[1].stop + margin, img.shape[1])

    cutout = img[xstart:xstop, ystart:ystop]
    return cutout
    # plot_figure(cutout, title='', figsize=(10,10), cbar=False, ticks=None, cmap=cm.Greys_r)

img = crop_image(img)

# <codecell>

def plot_figure_small_grey_nocbar(data, title=''):
    plot_figure(data, title, figsize=(3,3), cmap=cm.Greys_r)

def plot_figure(data, title='', figsize=(10,10), cbar=False, cbar_tick=None, cmap=cm.coolwarm):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    p = ax.imshow(data, cmap=cmap)
    if cbar:
        cb = fig.colorbar(p, shrink=.5)
        if cbar_tick:
            if ticks is None:
                ticks = range(np.max(data))
            cb.set_ticks(ticks)
            cb.ax.set_yticklabels([str(i) for i in ticks])
    ax.set_title(title)
    plt.show()

# <codecell>

from skimage.filter import gabor_kernel

kernels = []
kernel_freqs = []

theta_interval = 15

n_cols = img.shape[1]
next_pow2 = 2 ** int(np.ceil(np.log2(n_cols)))
max_freq = next_pow2 / 4
n_freqs = int(np.log2(max_freq))

# note: paper gives frequency in cycles per image width.
# we need cycles per pixel, so divide by image width
n = 4
frequencies =  list((np.sqrt(2) * float(2 ** i)) / n_cols
                    for i in range(max(0, n_freqs - n), n_freqs))

thetas = np.deg2rad(np.arange(0, 180, theta_interval))
for freq in frequencies:
    freq_band=.5
#     angular_band=np.deg2rad(45)
#     sigma_x = np.sqrt(np.log(2)) * (2 ** freq_band + 1) / (np.sqrt(2) * np.pi * frequency * (2 ** freq_band - 1))
#     sigma_y = np.sqrt(np.log(2)) / (np.sqrt(2) * np.pi * frequency * np.tan(angular_band / 2))
    for theta in thetas:
        kernel = gabor_kernel(freq, theta=theta,
                              bandwidth=.5)
#                               sigma_x=sigma_x, sigma_y=sigma_y,
        kernels.append(kernel)
        kernel_freqs.append(freq)

# only real component
kernels = map(np.real, kernels)
# kernels = list(np.real(k) for k in kernels)
# full wave rectification
# kernels = list(np.real(k)+np.imag(k) for k in kernels)

kernel_freqs = np.array(kernel_freqs)
n_kernels = len(kernels)

# <codecell>

def convolve_per_proc(k): return fftconvolve(img, k, 'same')

pool = multiprocessing.Pool()
%time filtered = np.dstack(pool.map(convolve_per_proc, kernels))

np.save('/home/yuncong/my_csd181_scratch/'+img_name+'_filtered', filtered)

# <codecell>

%%time
r2 = 0.95

energies = filtered.sum(axis=0).sum(axis=0)
order = np.argsort(energies)[::-1]
energies_sorted = energies[order]
r2s = np.cumsum(energies_sorted) / energies_sorted.sum()
k = np.searchsorted(r2s, r2)

selected_kernel_indices = order[:k]

filtered_reduced = filtered[..., selected_kernel_indices]

num_feat_reduced = len(selected_kernel_indices)
freqs_reduced = kernel_freqs[selected_kernel_indices]

proportion = 1.5
alpha = 0.25
nonlinear = np.tanh(alpha * filtered_reduced)
# features = nonlinear
sigmas = proportion * (1. / freqs_reduced)
features = np.dstack(gaussian_filter(nonlinear[..., i], sigmas[i]) 
                     for i in range(num_feat_reduced))
# features = np.dstack(gaussian_filter(filtered_reduced[..., i], sigmas[i]) for i in range(num_feat_reduced))
print features.shape

np.save('/home/yuncong/my_csd181_scratch/'+img_name+'_features', features)

