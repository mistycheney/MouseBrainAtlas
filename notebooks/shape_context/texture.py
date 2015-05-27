"""Texture segmentation based on

Jain and Farrokhnia, "Unsupervised texture segmentation using Gabor filters" (1991)

"""

from __future__ import division

import numpy as np
from scipy.signal import fftconvolve

from skimage import img_as_float
from skimage.filter import gabor_kernel
from skimage.filter import gaussian_filter
from skimage.color import hsv2rgb
from skimage.color import gray2rgb

from sklearn.decomposition import PCA


def _compute_sigmas(frequency, freq_band=1, angular_band=np.deg2rad(45)):
    """Taken from "Designing Gabor filters for optimal texture
    separability", (Clausi and Jernigan, 2000).

    """
    sigma_x = np.sqrt(np.log(2)) * (2 ** freq_band + 1) / (np.sqrt(2) * np.pi * frequency * (2 ** freq_band - 1))
    sigma_y = np.sqrt(np.log(2)) / (np.sqrt(2) * np.pi * frequency * np.tan(angular_band / 2))
    return sigma_x, sigma_y


def make_filter_bank(frequencies, thetas, real=True):
    """prepare filter bank of Gabor kernels"""
    # TODO: set MTF of each filter at (u, v) to 0
    kernels = []
    kernel_freqs = []
    for frequency in frequencies:
        sigma_x, sigma_y = _compute_sigmas(frequency)
        for theta in thetas:
            kernel = gabor_kernel(frequency, theta=theta,
                                  bandwidth=1)
            kernels.append(kernel)
            kernel_freqs.append(frequency)
    if real:
        kernels = list(np.real(k) for k in kernels)
    return kernels, np.array(kernel_freqs)


def get_freqs(img, n=4):
    """Compute the appropriate frequencies for an image of the given shape.

    Frequencies are given in cycles/pixel.

    """
    n_cols = img.shape[1]
    next_pow2 = 2 ** int(np.ceil(np.log2(n_cols)))
    max_freq = next_pow2 / 4
    n_freqs = int(np.log2(max_freq))

    # note: paper gives frequency in cycles per image width.
    # we need cycles per pixel, so divide by image width
    frequencies =  list((np.sqrt(2) * float(2 ** i)) / n_cols
                        for i in range(max(0, n_freqs - n), n_freqs))
    return frequencies


def filter_img(img, freqs=None, angle=10, crop=True):
    """create filter bank and filter image"""
    if freqs is None:
        freqs = get_freqs(img)
    thetas = np.deg2rad(np.arange(0, 180, angle))
    kernels, kernel_freqs = make_filter_bank(freqs, thetas)
    filtered = np.dstack(fftconvolve(img, kernel, 'same')
                         for kernel in kernels)
    if crop:
        x = max(k.shape[0] for k in kernels)
        y = max(k.shape[1] for k in kernels)
        x = int(np.ceil(x / 2))
        y = int(np.ceil(y / 2))
        filtered = filtered[x:-x, y:-y]

    return filtered, kernel_freqs


def filter_selection(filtered, kernel_freqs, r2=0.95):
    """Discards some filtered images.

    Returns filtered images with the largest energies so that the
    coefficient of determiniation is >= ``r2``.

    """
    energies = filtered.sum(axis=0).sum(axis=0)

    # sort from largest to smallest energy
    idx = np.argsort(energies)[::-1]
    filtered = filtered[:, :, idx]
    energies = energies[idx]
    total_energy = energies.sum()

    r2s = np.cumsum(energies) / energies.sum()
    k = np.searchsorted(r2s, r2)
    n_start = filtered.shape[2]
    return filtered[:, :, :k], frequencies[idx][:k]



def compute_features(filtered, frequencies,
                     proportion=0.5,
                     alpha=0.25):
    """Compute features for each filtered image.

    ``frequencies[i]`` is the center frequency of the Gabor filter
    that generated ``filtered[i]``.

    """
    # TODO: is this really what the paper means in formula 6?
    nonlinear = np.tanh(alpha * filtered)
    ncols = filtered.shape[1]

    # paper says proportion * n_cols / frequency, but remember that
    # their frequency is in cycles per image width. our frequency is
    # in cycles per pixel, so we just need to take the inverse.
    sigmas = proportion * (1.0 / np.array(frequencies))
    features = np.dstack(gaussian_filter(nonlinear[:, :, i], sigmas[i])
                         for i in range(len(sigmas)))
    return features


def add_coordinates(features, spatial_importance=1.0):
    """Adds coordinates to each feature vector and standardizes each feature."""
    n_rows, n_cols = features.shape[:2]
    coords = np.mgrid[:n_rows, :n_cols].swapaxes(0, 2).swapaxes(0, 1)
    features = np.dstack((features, coords))
    n_feats = features.shape[2]

    means = np.array(list(features[:, :, i].mean() for i in range(n_feats)))
    stds = np.array(list(features[:, :, i].std(ddof=1) for i in range(n_feats)))

    means = means.reshape(1, 1, -1)
    stds = stds.reshape(1, 1, -1)

    features = (features - means) / stds
    features[:, :, -2:] *= spatial_importance
    return features


def segment_textures(img, model, freqs=None, angle=10, select=True, k=4, coord=1):
    """Segments textures using Gabor filters and k-means."""
    filtered, kernel_freqs = filter_img(img, freqs, angle)
    filtered, kernel_freqs = filter_selection(filtered, kernel_freqs)
    features = compute_features(filtered, kernel_freqs)
    features = add_coordinates(features, spatial_importance=coord)
    n_feats = features.shape[-1]
    X = features.reshape(-1, n_feats)
    pca = PCA(k)
    X = pca.fit_transform(X)
    model.fit(X)
    return model.labels_.reshape(img.shape)


def directionality_filter(filtered, angle=10, combine=True):
    """
    Finds the maximum filter response for each pixel.

    Returns the maximum filter response and the angle of maximum response.

    """
    f2 = np.power(filtered, 2)

    n_angles = int(180 / angle)
    n_freqs = int(filtered.shape[2] / n_angles)

    if combine:
        f2_combined = np.dstack(f2[:, :, i::n_angles].sum(axis=2)
                                for i in range(n_angles))
        max_angle_idx = np.argmax(f2_combined, axis=2)
        x, y = np.indices(max_angle_idx.shape)
        magnitude = f2[x, y, max_angle_idx]

        angles = np.arange(0, 180, angle)
        max_angles = angles[max_angle_idx]
    else:
        angles = np.hstack(list(np.arange(0, 180, angle)
                                for f in range(n_freqs)))
        idx = np.argmax(filtered, axis=2)
        x, y = np.indices(idx.shape)
        magnitude = f2[x, y, idx]

        max_angles = angles[idx]

    magnitude = magnitude / np.mean(f2, axis=2)
    return magnitude, max_angles


def scale(arr):
    """scale array to [0, 1]"""
    return (arr - arr.min()) / arr.max()


def crop_to(img, shape):
    """crop ``img`` to ``shape``"""
    x1, y1 = img.shape
    x2, y2 = shape
    if x1 == x2 and y1 == y2:
        return img
    if x2 > x1 or y2 > y1:
        raise Exception('cannot crop image of shape {}'
                        ' to shape {}'.format(img, shape))
    mx = int((x1 - x2) / 2)
    my = int((y1 - y2) / 2)
    return img[mx:-mx, my:-my]


def make_hsv(magnitude, angle, img=None, alpha=0.5):
    """Convert the result of ``directionality_filter`` to an HSV
    image, then convert to RGB."""
    magnitude = scale(magnitude)
    angle = scale(angle)
    h = angle
    s = magnitude
    v = np.ones(h.shape)
    hsv = hsv2rgb(np.dstack([h, s, v]))
    if img is None:
        return hsv
    img = scale(crop_to(img, angle.shape))
    result = hsv + gray2rgb(img)
    return img_as_float(result)
