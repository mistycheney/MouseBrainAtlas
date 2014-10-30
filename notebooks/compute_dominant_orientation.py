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
    DATA_DIR = '/home/yuncong/BrainLocal/DavidData_4'
    REPO_DIR = '/home/yuncong/BrainSaliencyDetection'

dm = DataManager(DATA_DIR, REPO_DIR)

# import argparse
# import sys

# parser = argparse.ArgumentParser(
# formatter_class=argparse.RawDescriptionHelpFormatter,
# description='Execute feature extraction pipeline',
# epilog="""
# The following command processes image RS141_x5_0001.tif using blueNissl for both gabor parameters and segmentation parameters.
# python %s RS141 x5 1 -g blueNissl -s blueNissl -v blueNissl
# """%(os.path.basename(sys.argv[0]), ))

# parser.add_argument("stack_name", type=str, help="stack name")
# parser.add_argument("resolution", type=str, help="resolution string")
# parser.add_argument("slice_ind", type=int, help="slice index")
# parser.add_argument("-g", "--gabor_params_id", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNissl')
# parser.add_argument("-s", "--segm_params_id", type=str, help="segmentation parameters id (default: %(default)s)", default='blueNissl')
# parser.add_argument("-v", "--vq_params_id", type=str, help="vq parameters id (default: %(default)s)", default='blueNissl')
# args = parser.parse_args()

class args:
    stack_name = 'RS141'
    resolution = 'x5'
    slice_ind = 1
    gabor_params_id = 'blueNisslWide'
    segm_params_id = 'blueNissl'
    vq_params_id = 'blueNissl'

# <codecell>

dm.set_image(args.stack_name, args.resolution, args.slice_ind)
dm.set_gabor_params(gabor_params_id=args.gabor_params_id)
dm.set_segmentation_params(segm_params_id=args.segm_params_id)
dm.set_vq_params(vq_params_id=args.vq_params_id)

from joblib import Parallel, delayed

n_texton = int(dm.vq_params['n_texton'])

theta_interval = dm.gabor_params['theta_interval']
n_angle = int(180/theta_interval)
freq_step = dm.gabor_params['freq_step']
freq_max = 1./dm.gabor_params['min_wavelen']
freq_min = 1./dm.gabor_params['max_wavelen']
n_freq = int(np.log(freq_max/freq_min)/np.log(freq_step)) + 1
frequencies = freq_max/freq_step**np.arange(n_freq)
angles = np.arange(0, n_angle)*np.deg2rad(theta_interval)

# <codecell>

cropped_segmentation = dm.load_pipeline_result('cropSegmentation', 'npy')
n_superpixels = len(unique(cropped_segmentation)) - 1

cropped_features = dm.load_pipeline_result('cropFeatures', 'npy')
cropped_mask = dm.load_pipeline_result('cropMask', 'npy')

# <codecell>

cropped_features_tabular = np.reshape(cropped_features, (cropped_features.shape[0], cropped_features.shape[1], n_freq, n_angle))
directional_max = cropped_features_tabular.max(axis=2)
response_argsorted = directional_max.argsort(axis=2)
response_sorted = np.sort(directional_max, axis=2)
max_second_ratio = response_sorted[:,:,-1]/response_sorted[:,:,-2]
max_response = response_sorted[:,:,-1]
max_dir = response_argsorted[:,:,-1]

# <codecell>

def worker(i):
    return np.bincount(max_dir[cropped_segmentation == i]).argmax()
    
max_dir_sp = Parallel(n_jobs=16)(delayed(worker)(i) for i in range(n_superpixels))
max_dir_sp = np.array(max_dir_sp)

# <codecell>

cropped_image = dm.load_pipeline_result('cropImg', 'tif')

# <codecell>

dirmap = max_dir_sp[cropped_segmentation].copy()
dirmap[max_second_ratio < 1.02] = -1

# <codecell>

hc_colors = np.loadtxt('../visualization/high_contrast_colors.txt', skiprows=1)/255.

# <codecell>

dirmap[(~cropped_mask)]  = -1

dirmap_vis = label2rgb(dirmap, bg_label=-1, image=cropped_image, colors=hc_colors[1:1+n_angle], image_alpha=0.3)
dirmap_vis[~cropped_mask] = 0
dirmap_vis[max_second_ratio < 1.02] = 0
# dm.save_pipeline_result(dirmap_vis, 'dirMap', 'tif')

# <codecell>

from skimage.util import img_as_ubyte

dirmap_vis2 = dirmap_vis.copy()

sp_properties = dm.load_pipeline_result('cropSpProps', 'npy')

dirmap_vis2 = img_as_ubyte(dirmap_vis2)

for s in range(n_superpixels - 1):
    center = sp_properties[s, [1,0]].astype(np.int)
    angle = angles[max_dir_sp[s]]
    end = center + np.array([20*np.sin(angle), -20*np.cos(angle)], dtype=np.int)
    
    cv2.line(dirmap_vis2, tuple(center), tuple(end), (255,255,255), thickness=5, lineType=8, shift=0)
        
dm.save_pipeline_result(dirmap_vis2, 'dirMap', 'tif')
    

