# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import sys, os
from scipy.ndimage import gaussian_filter, measurements
from scipy.signal import fftconvolve
import numpy as np
import matplotlib.pyplot as plt
import cv2
from multiprocessing import Pool

from skimage.filter import gabor_kernel

from utilities import *

global_pool = None

class GaborFeatureComputer(object):    
    def __init__(self):
        self.kernels = None
    
    def _generate_kernels(self, frequencies=None, theta_interval=15):
        self.kernels = []
        self.kernel_freqs = []
        self.kernel_thetas = []
        
        im_width = min(self.img.shape)

        if frequencies is None:
            n_freqs = 6
            cpi_max = im_width/10.
            cpis = np.array([cpi_max/(np.sqrt(2)**i) for i in range(n_freqs)])
            frequencies = cpis/im_width

        for freq in frequencies:
            for theta in np.arange(0, np.pi, np.deg2rad(theta_interval)):
                kernel = gabor_kernel(freq, theta=theta,
                                      bandwidth=1.)
                self.kernels.append(kernel)
                self.kernel_freqs.append(freq)
                self.kernel_thetas.append(theta)

        self.kernels = map(np.real, self.kernels)
        self.kernel_thetas = np.array(self.kernel_thetas)
        self.kernel_freqs = np.array(self.kernel_freqs)
        
        print '# of kernels = %d' % (len(self.kernels))
        print 'frequencies:', frequencies
        print '# of pixels per cycle:', 1/frequencies
        print 'kernel size:', [kern.shape[0] for kern in self.kernels]

        
    def _convolve_per_proc(self, i):
        return fftconvolve(self.img, self.kernels[i], 'same')
    
    def _gaussian_filter_per_proc(self, (sigma, filtered)):
        nonlinear = np.tanh(self.alpha * filtered)
        return nonlinear
#         return gaussian_filter(nonlinear, sigma)
        
    @timeit
    def process_image(self, img, frequencies=None, filtered_output=None, quantile=0.95, 
                        selected_kernel_indices_output=None, 
                        features_output=None):
        self.img = img
                        
        if self.kernels is None:
            self._generate_kernels(frequencies)
                            
        global global_pool
        if global_pool is None:
            global_pool = Pool()
            
        filtered = np.dstack(global_pool.map(self._convolve_per_proc, range(len(self.kernels))))

        if filtered_output is not None:
            np.save(filtered_output, filtered)
            print 'filtered images saved to %s' % filtered_output

        print 'filtering completed.'

#         self.features = filtered
        
        energies = filtered.sum(axis=0).sum(axis=0)
        order = np.argsort(energies)[::-1]
        energies_sorted = energies[order]
        r2s = np.cumsum(energies_sorted) / energies_sorted.sum()
        k = np.searchsorted(r2s, quantile)

        selected_kernel_indices = order[:k]

#         if selected_kernel_indices_output is not None:
#             np.save(selected_kernel_indices_output, selected_kernel_indices)

        num_feat_reduced = len(selected_kernel_indices)
        
        freqs_reduced = self.kernel_freqs[selected_kernel_indices]
        print freqs_reduced
        proportion = 1.5
        sigmas = proportion * (1. / freqs_reduced)
#         print sigmas

        self.alpha = 0.25
        
        filtered_reduced_list = [filtered[:,:,i] for i in selected_kernel_indices]
        self.features = np.dstack(global_pool.map(self._gaussian_filter_per_proc, zip(sigmas, filtered_reduced_list)))

        print '%d features selected' % self.features.shape[-1]

        if features_output is not None:
            np.save(features_output, self.features)
            print 'features saved to %s' % features_output

            
    def visualize_features(self, output=None):
        fig, axes = plt.subplots(ncols=4, nrows=self.features.shape[-1]/4, figsize=(20,10))
        for i, ax in zip(range(self.features.shape[-1]), axes.flat):
            ax.matshow(self.features[...,i])
            ax.set_title('feature %d'%i, fontsize=0.5)
            ax.axis('off')
        plt.close();
        
        if output is not None:
            fig.savefig(output, bbox_inches='tight')
            

