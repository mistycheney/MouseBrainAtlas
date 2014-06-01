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

class GaborFeatureComputer(object):    
    def __init__(self):
        pass
    
    def _generate_kernels(self):
        self.kernels = []
        self.kernel_freqs = []
        self.kernel_thetas = []
        
        im_width = min(self.img.shape)
        theta_interval = 15

        n_freqs = 5
        cpi_max = im_width/10. *np.sqrt(2)
        cpis = np.array([cpi_max/(np.sqrt(2)**i) for i in range(n_freqs)])
        frequencies = cpis/im_width

        for freq in frequencies:
            for theta in np.arange(0, np.pi, np.deg2rad(theta_interval)):
                kernel = gabor_kernel(freq, theta=theta,
                                      bandwidth=.5)
                self.kernels.append(kernel)
                self.kernel_freqs.append(freq)
                self.kernel_thetas.append(theta)

        self.kernels = map(np.real, self.kernels)
        self.kernel_thetas = np.array(self.kernel_thetas)
        self.kernel_freqs = np.array(self.kernel_freqs)
        
        print '# of kernels = %d' % (len(self.kernels))
        print 'frequencies:', frequencies
        print '# of pixels per cycle:', 1/frequencies
        
    def process_image(self, img, filtered_output=None,
                      selected_kernel_indices_output=None, 
                     features_output=None):
        self.img = img
        if self.kernels is None:
            self._generate_kernels()
        self.do_gabor_filter(filtered_output)
        self.compute_features(selected_kernel_indices_output=selected_kernel_indices_output, 
                     features_output=features_output)
        
    def _convolve_per_proc(self, k): return fftconvolve(self.img, k, 'same')
    
    @timeit
    def do_gabor_filter(self, filtered_output=None):
        
        pool = Pool()
        self.filtered = np.dstack(pool.map(self._convolve_per_proc, self.kernels))
        if filtered_output is not None:
            np.save(filtered_output, self.filtered)
            print 'filtered images saved to %s' % filtered_output
        print 'filtering completes'
    
    @timeit
    def compute_features(self, quantile=0.95, selected_kernel_indices_output=None, 
                     features_output=None):

        energies = self.filtered.sum(axis=0).sum(axis=0)
        order = np.argsort(energies)[::-1]
        energies_sorted = energies[order]
        r2s = np.cumsum(energies_sorted) / energies_sorted.sum()
        k = np.searchsorted(r2s, quantile)

        selected_kernel_indices = order[:k]

        if selected_kernel_indices_output is not None:
            np.save(selected_kernel_indices_output, selected_kernel_indices)

        filtered_reduced = self.filtered[..., selected_kernel_indices]

        num_feat_reduced = len(selected_kernel_indices)
        freqs_reduced = self.kernel_freqs[selected_kernel_indices]

        proportion = 1.5
        alpha = 0.25
        nonlinear = np.tanh(alpha * filtered_reduced)
        sigmas = proportion * (1. / freqs_reduced)
        features = np.dstack(gaussian_filter(nonlinear[..., i], sigmas[i]) 
                             for i in range(num_feat_reduced))

        print '%d features selected' % features.shape[-1]

        if features_output is not None:
            np.save(features_output, features)
            print 'features saved to %s' % features_output

