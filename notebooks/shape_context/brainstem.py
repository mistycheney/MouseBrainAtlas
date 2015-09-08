from __future__ import division

import os
from cPickle import dump, load
import itertools

import glymur

import numpy as np

from scipy.stats import poisson
from scipy import ndimage
from scipy import misc

import sklearn.decomposition as decomp
from sklearn.cluster import KMeans

from skimage.filter import threshold_otsu, threshold_adaptive
from skimage.morphology import binary_dilation, binary_erosion
from skimage.morphology import watershed
from skimage.color import label2rgb
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import clear_border


# list of shape descriptors for clustering
FEATURE_NAMES = (
    'area',
    'convex_area',
    'eccentricity',
    'equivalent_diameter',
    'extent',
    'filled_area',
    'major_axis_length',
    'minor_axis_length',
    'perimeter',
    'solidity',
    )

# a simple cache for grayscale images at different resolutions.
USE_CACHE = True
DATA_DIR = os.path.expanduser("~/devel/data/images/PMD1305_N")
CACHE_DIR = os.path.expanduser("~/devel/data/cache")
CACHE_PATH = os.path.join(CACHE_DIR, 'cache.pkl')

if USE_CACHE:
    if not os.path.exists(CACHE_DIR):
        os.mkdir(CACHE_DIR)
    if not os.path.isdir(CACHE_DIR):
        raise Exception('path is not a directory: {}'.format(CACHE_DIR))
    try:
        CACHE = load(file(CACHE_PATH, 'rb'))
    except IOError:
        CACHE = {}


def get_filenames():
    """returns a list of all jp2 images in the data directory"""
    filenames = os.listdir(DATA_DIR)
    filenames = list(n for n in filenames if os.path.splitext(n)[1] == '.jp2')
    filenames = sorted(filenames, key=lambda n: n.split('_')[-1])
    return filenames


def _read_jp2_img(filename, rlevel):
    """read a jp2 image at the given level of resolution.

    rlevel = 0 is highest resolution.
    rlevel = -1 is a shortcut for the lowest resolution.

    """
    jpimg = glymur.Jp2k(os.path.join(DATA_DIR, filename))
    return jpimg.read(rlevel=rlevel)


def get_img(filename, rlevel):
    """get an image, using the cache if available."""
    try:
        fname = CACHE[(filename, rlevel)]
        return misc.imread(fname)
    except:
        img = _read_jp2_img(filename, rlevel)
        if USE_CACHE:
            cache_file = os.path.join(CACHE_DIR, "{}_rlevel_{}.tif".format(filename, rlevel))
            misc.imsave(cache_file, img)
            CACHE[(filename, rlevel)] = cache_file
            dump(CACHE, open(CACHE_PATH, 'wb'))
        return img


def make_grey(img):
    """convert a color image to grayscale using PCA"""
    pca = decomp.PCA(1)
    img = pca.fit_transform(img.reshape(-1, 3)).reshape(img.shape[:2])
    return (img - img.min()) / (img.max() - img.min())


def get_cutout(filename, rlevel=1, margin=100):
    """read an image, cropping out the background"""
    # find bounding box of brain in a slice
    small_img = get_img(filename, rlevel=4)
    small_img = make_grey(small_img)
    blurred = ndimage.gaussian_filter(small_img, 10)
    slc = ndimage.measurements.find_objects(blurred < threshold_otsu(blurred))[0]
    k = 4 - rlevel

    img = get_img(filename, rlevel=rlevel)
    xstart = max(slc[0].start * 2 ** k - margin, 0)
    xstop = min(slc[0].stop * 2 ** k + margin, img.shape[0])
    ystart = max(slc[1].start * 2 ** k - margin, 0)
    ystop = min(slc[1].stop * 2 ** k + margin, img.shape[1])

    cutout = img[xstart:xstop, ystart:ystop]
    del img
    return cutout


def random_image_sample(img, scale=5):
    """extract a random subset of the image.

    The result is ``scale`` times smaller in each dimension.

    """
    x_shape = int(np.floor(img.shape[0] / scale))
    y_shape = int(np.floor(img.shape[1] / scale))
    x_start = np.random.randint(0, img.shape[0] - x_shape)
    y_start = np.random.randint(0, img.shape[1] - y_shape)

    return img[x_start : x_start + x_shape,
               y_start : y_start + y_shape]


def sample_many(filenames, rlevel=1, scale=5):
    result = []
    for f in filenames:
        img = get_cutout(f, rlevel=rlevel)
        sample = random_image_sample(img, scale).copy()
        del img
        result.append(sample)
    return result


def segment_cells(img):
    """label the cells in an image.

    Returns the labeled image and the number of labels.

    """
    if img.ndim == 3 and img.shape[-1] > 1:
        img = make_grey(img)
    # # global threshold and watershed
    # binary = img < threshold_otsu(img)
    # distance = ndimage.distance_transform_edt(binary)
    # local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=binary)
    # markers = ndimage.label(local_maxi)[0]
    # labels = watershed(-distance, markers, mask=binary)

    # local threshold and erosion / dilation
    t_img = threshold_adaptive(img, 25, offset=.01)
    b_img = binary_erosion(-t_img, np.ones((3, 3)))
    d_img = binary_dilation(b_img, np.ones((3, 3)))
    clear_border(d_img)
    labels, n_labels = ndimage.label(d_img)
    return labels, n_labels


def object_features(img, feature_names=None):
    """returns a feature array for the given features"""
    if feature_names is None:
        feature_names = FEATURE_NAMES
    props = regionprops(img)
    f = lambda x: np.array(x).ravel()
    return np.vstack(np.hstack(np.array(getattr(p, n)).ravel()
                               for n in feature_names)
                    for p in props)


def feature_column_names(feature_names=None):
    """Returns the feature name of each column.

    Some features have more than one value. This function makes it
    easier to figure out the feature corresponding to a particular
    column.

    """
    if feature_names is None:
        feature_names = FEATURE_NAMES
    img = np.zeros((5, 5), dtype=np.bool)
    img[1:4, 1:4] = 1
    props = regionprops(img)
    p = props[0]
    f = lambda x: np.array(x).ravel()
    names = ([n] * len(f(getattr(p, n))) for n in feature_names)
    names = list(itertools.chain(*names))
    return names


def all_object_features(imgs, feature_names=None):
    """a convenience function for extracting features for multiple images."""
    if feature_names is None:
        feature_names=FEATURE_NAMES
    features = list(object_features(img, feature_names)
                    for img in imgs)
    # normalize
    features = np.vstack(features)
    means = features.mean(axis=0)
    stds = features.std(axis=0, ddof=1)
    features = (features - means) / stds
    return features 


def split_labels(imgs, labels, n_objects=None):
    """split a single label vector according to binary images"""
    if n_objects is None:
        n_objects = list(len(set(i.flatten())) - 1 for i in imgs)
    a = np.insert(np.cumsum(n_objects), 0, 0)
    slices = list(slice(a[i], a[i + 1]) for i in range(len(a) - 1))
    labels = list(labels[slc] for slc in slices)
    return labels


def assign_clusters(img, labels, rgb=False, colors=None):
    """label each object with the corresponding cluster from ``labels``."""
    labels = np.insert(labels, 0, np.int32(-1)) + 1
    if rgb:
        return label2rgb(labels[img], bg_label=0, colors=colors)
    return labels[img]

