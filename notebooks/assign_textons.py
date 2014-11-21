# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%load_ext autoreload
%autoreload 2

# <codecell>

from utilities import *

if 'SSH_CONNECTION' in os.environ:
    DATA_DIR = '/home/yuncong/DavidData'
    REPO_DIR = '/home/yuncong/Brain'
else:
    DATA_DIR = '/home/yuncong/BrainLocal/DavidData_v4'
    REPO_DIR = '/home/yuncong/Brain'

dm = DataManager(DATA_DIR, REPO_DIR)


# import argparse
# import sys

# parser = argparse.ArgumentParser(
# formatter_class=argparse.RawDescriptionHelpFormatter,
# description='Execute feature extraction pipeline',
# epilog="""
# The following command processes image RS141_x5_0001.tif using blueNissl for both gabor parameters and segmentation parameters.
# python %s RS141 x5 1 -g blueNissl -v blueNissl
# """%(os.path.basename(sys.argv[0]), ))

# parser.add_argument("stack_name", type=str, help="stack name")
# parser.add_argument("resolution", type=str, help="resolution string")
# parser.add_argument("slice_ind", type=int, help="slice index")
# parser.add_argument("-g", "--gabor_params_id", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNissl')
# parser.add_argument("-v", "--vq_params_id", type=str, help="vq parameters id (default: %(default)s)", default='blueNissl')
# args = parser.parse_args()

class args:
    stack_name = 'RS141'
    resolution = 'x5'
    slice_ind = 1
    gabor_params_id = 'blueNisslWide'
    vq_params_id = 'blueNissl'

# <codecell>

dm.set_image(args.stack_name, args.resolution, args.slice_ind)
dm.set_gabor_params(gabor_params_id=args.gabor_params_id)
dm.set_vq_params(vq_params_id=args.vq_params_id)

n_texton = int(dm.vq_params['n_texton'])

theta_interval = dm.gabor_params['theta_interval']
n_angle = int(180/theta_interval)
freq_step = dm.gabor_params['freq_step']
freq_max = 1./dm.gabor_params['min_wavelen']
freq_min = 1./dm.gabor_params['max_wavelen']
n_freq = int(np.log(freq_max/freq_min)/np.log(freq_step)) + 1

# <codecell>

centroids = dm.load_pipeline_result('textons', 'npy')

# <codecell>

fig, axes = plt.subplots(10, 10, figsize=(20,20), facecolor='white')

for i in range(10):
    for j in range(10):
        axes[i,j].matshow(centroids[i*10+j].reshape((n_freq, n_angle)))
tight_layout()

plt.show()

# <codecell>

import itertools
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist

try:
    textonmap = dm.load_pipeline_result('texMap', 'npy')
except:
    
    centroids = dm.load_pipeline_result('textons', 'npy')

    def compute_dist_per_proc(X_partial, c_all_rot):
        D = cdist(X_partial, c_all_rot, 'sqeuclidean')
        ci, ri = np.unravel_index(D.argmin(axis=1), (n_texton, n_angle))
        return np.c_[ci, ri]

    cropped_features = dm.load_pipeline_result('cropFeatures', 'npy')
    cropped_mask = dm.load_pipeline_result('cropMask', 'npy')

    n_pixels = cropped_features.shape[0]*cropped_features.shape[1]
    n_splits = 1000
    n_sample = min(int(dm.vq_params['n_sample']), n_pixels)

    centroid_all_rotations = np.vstack([np.concatenate(np.roll(np.split(c, n_freq), i)) 
                            for c,i in itertools.product(centroids, range(n_angle))])

    X = cropped_features.reshape(-1, cropped_features.shape[-1])
    r = Parallel(n_jobs=16)(delayed(compute_dist_per_proc)(x,c) 
                            for x, c in zip(np.array_split(X, n_splits, axis=0), 
                                            itertools.repeat(centroid_all_rotations, n_splits)))
    res = np.vstack(r)

    labels = res[:,0]
    #     matched_rotations = res[:,1]

    textonmap = labels.reshape(cropped_features.shape[:2])
    textonmap[~cropped_mask] = -1

    dm.save_pipeline_result(textonmap, 'texMap', 'npy')

    textonmap_rgb = label2rgb(textonmap, image=None, colors=None, alpha=0.3, image_alpha=1)
    dm.save_pipeline_result(textonmap_rgb, 'texMap', 'tif')

# <codecell>

plt.imshow(textonmap)

