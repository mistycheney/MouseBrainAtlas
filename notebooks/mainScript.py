# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%load_ext autoreload
%autoreload 2

from IPython.display import Image, FileLink

from compute_features_module import GaborFeatureComputer
from kmeans_module import SaliencyDetector
from utilities import *

import cv2
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# <codecell>

CACHE_DIR = '/home/yuncong/my_csd181_scratch/'
# IMG_DIR = '/home/yuncong/my_csd181/ParthaData/PMD1305_reduce2/region3/'
# img_name_fmt = 'PMD1305_%d.reduce2.region3'
IMG_DIR = '/home/yuncong/my_csd181/ParthaData/PMD1305_reduce2/region1/'
img_name_fmt = 'PMD1305_%d.reduce2.region1'

# <codecell>

gc = GaborFeatureComputer()
img_id = 244
# for img_id in range(159,176): # region2
# img_id = 4
print img_id
img_name = img_name_fmt%img_id
img = cv2.imread(IMG_DIR + img_name + '.tif', 0)
gc.process_image(img, 
                 filtered_output = CACHE_DIR + img_name + '_filtered',
                 features_output = CACHE_DIR + img_name + '_features')

# <codecell>

gc.visualize_features(output='output/'+img_name+'_features_%d.png'%img_id)

# <codecell>

FileLink('output/'+img_name+'_features_%d.png'%img_id)

# <codecell>

gc = GaborFeatureComputer()
# for img_id in range(241,250): # region1
for img_id in range(159,176): # region2
    print img_id
    img_name = img_name_fmt%img_id
    img = cv2.imread(IMG_DIR + img_name + '.tif', 0)
    gc.process_image(img, features_output = CACHE_DIR + img_name + '_features')

# <codecell>

compactness = 5
presmooth = 10
num_textons = 10
dist_thresh = .8
neighbor_term_weight = 1.
img_id = 159
img_name = img_name_fmt%img_id
img = cv2.imread(IMG_DIR + img_name + '.tif', 0)
mask = foreground_mask(img)

sd = SaliencyDetector(img, mask=mask, features_input=CACHE_DIR + img_name +'_features.npy')
sd.compute_texton(num_textons=num_textons)
sd.segment_superpixels(compactness=compactness, sigma=presmooth)
sd.visualize_textonmap_superpixels(output='output/'+img_name+'_textonmap_%d.png'%img_id)

# <codecell>

sd.segment_superpixels(compactness=compactness, sigma=presmooth)

# <codecell>

sd.visualize_textonmap_superpixels(output='output/'+img_name+'_textonmap_%d.png'%img_id)
FileLink('output/'+img_name+'_textonmap_%d.png'%img_id)

# <codecell>

np.unique(sd.textonmap)

# <codecell>

compactness = 5
presmooth = 10
num_textons = 10
dist_thresh = .8
neighbor_term_weight = 1.

# for img_id in range(241,250):
# for img_id in range(159,176): # region2

#     print img_id
    
#     img_name = img_name_fmt%img_id
#     img = cv2.imread(IMG_DIR + img_name + '.tif', 0)
#     mask = foreground_mask(img)
    
#     sd = SaliencyDetector(img, mask=mask, features_input=CACHE_DIR + img_name +'_features.npy')
#     sd.compute_texton(num_textons=num_textons)
#     sd.segment_superpixels(compactness=compactness, sigma=presmooth)
#     sd.compute_connectivity()
#     sd.compute_distance_matrix()
#     sd.compute_saliency_map(neighbor_term_weight=neighbor_term_weight)
#     sd.find_salient_clusters(dist_thresh=dist_thresh, n_top_clusters=10)
#     sd.visualize_features(output='output/'+img_name+'_features_%d.png'%img_id)
#     sd.visualize_superpixels(output='output/'+img_name+'_superpixelized_%d.png'%img_id)
#     sd.visualize_textonmap_superpixels(output='output/'+img_name+'_textonmap_%d.png'%img_id)
#     sd.visualize_saliency_map(output='output/'+img_name+'_saliencymap_%d.png'%img_id)
#     sd.visualize_salient_clusters(output='output/'+img_name+'_salientclusters_%d.png'%img_id)
    

# <codecell>

from matplotlib.backends.backend_pdf import PdfPages

with PdfPages('region2_collection.pdf') as pdf:
    
#     for img_id in range(241,250):
    for img_id in range(159,176): # region2
        img_name = img_name_fmt%img_id

        plt.imshow(cv2.imread('output/'+img_name+'_features_%d.png'%img_id))
        plt.title('features, image %d'% img_id, fontsize=5)
        plt.axis('off')
        pdf.savefig(bbox_inches='tight', dpi=300)
        plt.close();
        
        fig, axes = plt.subplots(nrows=2, ncols=2)
        
#         img_name = 'PMD1305_%d.reduce2.region1'%img_id    
#         im = cv2.imread(IMG_DIR + img_name + '.tif')
        
        fig.suptitle('image %d' % img_id, fontsize=5)
    
        axes[0,0].imshow(cv2.imread('output/'+img_name+'_superpixelized_%d.png'%img_id))
        axes[0,0].set_title('superpixelized, compactness=%d, presmooth=%.2f'%(compactness,presmooth), fontsize=5)
        axes[0,0].axis('off')
            
        axes[0,1].imshow(cv2.imread('output/'+img_name+'_textonmap_%d.png'%img_id))
        axes[0,1].set_title('texton map, num_textons=%d'%num_textons, fontsize=5)
        axes[0,1].axis('off')
        
        axes[1,0].imshow(cv2.imread('output/'+img_name+'_saliencymap_%d.png'%img_id))
        axes[1,0].set_title('saliency map, neighbor_term_weight=%.2f'%neighbor_term_weight, fontsize=5)
        axes[1,0].axis('off')
        
        axes[1,1].imshow(cv2.imread('output/'+img_name+'_salientclusters_%d.png'%img_id))
        axes[1,1].set_title('10 most salient clusters, cluster_grow_thresh=%.2f'%dist_thresh, fontsize=5)
        axes[1,1].axis('off')
        
        pdf.savefig(bbox_inches='tight', dpi=300)
        plt.close();

# <codecell>

from matplotlib.backends.backend_pdf import PdfPages

with PdfPages('region1_collection.pdf') as pdf:
    
    for img_id in range(241,250):# region1
#     for img_id in range(159,176): # region2
        img_name = img_name_fmt%img_id
        print img_name

        plt.imshow(cv2.imread('output/'+img_name+'_features_%d.png'%img_id))
        plt.title('features, image %d'% img_id, fontsize=5)
        plt.axis('off')
        pdf.savefig(bbox_inches='tight', dpi=300)
        plt.close();
        
#         fig, axes = plt.subplots(nrows=2, ncols=2)
        
#         img_name = 'PMD1305_%d.reduce2.region1'%img_id    
#         im = cv2.imread(IMG_DIR + img_name + '.tif')
                
        plt.imshow(cv2.imread('output/'+img_name+'_superpixelized_%d.png'%img_id))
        plt.title('image %d, ' % img_id + 'superpixelized, compactness=%d, presmooth=%.2f'%(compactness,presmooth), fontsize=5)
        plt.axis('off')
        pdf.savefig(bbox_inches='tight', dpi=600)
        
        plt.imshow(cv2.imread('output/'+img_name+'_textonmap_%d.png'%img_id))
        plt.title('image %d, ' % img_id + 'texton map, num_textons=%d'%num_textons, fontsize=5)
        plt.axis('off')
        pdf.savefig(bbox_inches='tight', dpi=600)
        
        plt.imshow(cv2.imread('output/'+img_name+'_saliencymap_%d.png'%img_id))
        plt.title('image %d, ' % img_id + 'saliency map, neighbor_term_weight=%.2f'%neighbor_term_weight, fontsize=5)
        plt.axis('off')
        pdf.savefig(bbox_inches='tight', dpi=600)
        
        plt.imshow(cv2.imread('output/'+img_name+'_salientclusters_%d.png'%img_id))
        plt.title('image %d, ' % img_id + '10 most salient clusters, cluster_grow_thresh=%.2f'%dist_thresh, fontsize=5)
        plt.axis('off')
        
        pdf.savefig(bbox_inches='tight', dpi=600)
        plt.close();

