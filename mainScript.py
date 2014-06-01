# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%load_ext autoreload
%autoreload 2

from IPython.display import Image, FileLink

from compute_features_module import GaborFeatureComputer
from kmeans_module import SaliencyDetector

import cv2

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# <codecell>

for img_id in range(241,250):
    img_name = 'PMD1305_%d.reduce2.region1'%img_id
    CACHE_DIR = '/home/yuncong/my_csd181_scratch/'
    IMG_DIR = '/home/yuncong/my_csd181/ParthaData/PMD1305_reduce2/region1/'

    # img_name = 'Resized_6_region'
    # IMG_DIR = '/home/yuncong/my_csd181_scratch/'

    # img = cv2.imread(IMG_DIR + 'PMD1305_%d.reduce2.region.tif' % img_id, 0)
    img = cv2.imread(IMG_DIR + img_name + '.tif', 0)

    gc = GaborFeatureComputer()
    filtered_output = CACHE_DIR + img_name + '_filtered'
    features_output = CACHE_DIR + img_name + '_features'
    gc.process_image(img, filtered_output=filtered_output,
                     features_output=features_output)

# <codecell>

img_id = 244
img_name = 'PMD1305_%d.reduce2.region1'%img_id
CACHE_DIR = '/home/yuncong/my_csd181_scratch/'
IMG_DIR = '/home/yuncong/my_csd181/ParthaData/PMD1305_reduce2/region1/'

sd = SaliencyDetector(img, CACHE_DIR + img_name +'_features.npy')
sd.compute_texton(num_textons=10)
sd.segment_superpixels(compactness=5)
sd.compute_connectivity()
sd.compute_distance_matrix()
sd.compute_saliency_map()
sd.find_salient_clusters(dist_thresh=.5)

# <codecell>

sd.visualize_features()

# <codecell>

sd.visualize_textonmap_superpixels(output='ts1.jpg')
Image('ts1.jpg')

# <codecell>

sd.visualize_saliency_map()

# <codecell>

sd.visualize_salient_clusters(output='sel.jpg')
Image('sel.jpg')

# <codecell>

FileLink('sel.jpg')

