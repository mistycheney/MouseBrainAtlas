# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%load_ext autoreload
%autoreload 2

%autosave 0

# <codecell>

from preamble import *

# <codecell>

segmentation = dm.load_pipeline_result('segmentation', 'npy')
n_superpixels = len(unique(segmentation)) - 1

features = dm.load_pipeline_result('features', 'npy').astype(np.float)

# <codecell>

features = np.rollaxis(features, 0, 3)

# <codecell>

max_freqs, max_angles = np.unravel_index(features.argmax(axis=2), (dm.n_freq, dm.n_angle))
max_responses = features.max(axis=2)
max_mean_ratio = max_responses/features.mean(axis=2)

# <codecell>

def worker(i):
    chosen = segmentation == i
    
    max_response_sp = max_responses[chosen].astype(np.float).max()
    max_dir_sp = np.bincount(max_angles[chosen]).argmax()
    max_freq_sp = np.bincount(max_freqs[chosen]).argmax()
    
    all_mmr = max_mean_ratio[chosen].astype(np.float)
    dominant_ratio_sp = np.count_nonzero(all_mmr > 1.02)/float(len(all_mmr))

    return max_dir_sp, max_freq_sp, max_response_sp, dominant_ratio_sp
    
res = Parallel(n_jobs=16)(delayed(worker)(i) for i in range(n_superpixels))
max_dir_sp, max_freq_sp, max_response_sp, dominant_ratio_sp = map(np.array, zip(*res))

# <codecell>

dm.save_pipeline_result(max_dir_sp, 'spMaxDirInd', 'npy')
dm.save_pipeline_result(dm.angles[max_dir_sp], 'spMaxDirAngle', 'npy')

# <codecell>

segmentation_vis = dm.load_pipeline_result('segmentationWithText', 'jpg')
segmentation_vis2 = segmentation_vis.copy()
segmentation_vis2[~dm.mask] = 0

# <codecell>

hc_colors = np.loadtxt('../visualization/100colors.txt')

# <codecell>

# from skimage.color import hsv2rgb
# [hsv2rgb((i/n_freq, )) for i in range(n_freq) for j in range(n_angle)]

# <codecell>

max_response_sp_normalized = (max_response_sp - max_response_sp.min())/(max_response_sp.max() - max_response_sp.min())

from skimage.util import img_as_ubyte

dirmap_vis2 = gray2rgb(dm.image.copy())
# dirmap_vis2 = gray2rgb(cropped_segmentation_vis2.copy())
# dirmap_vis2 = gray2rgb(np.zeros_like(cropped_segmentation, dtype=np.uint8))
dirmap_vis2 = img_as_ubyte(dirmap_vis2)

sp_properties = dm.load_pipeline_result('spProps', 'npy')

for s in range(n_superpixels - 1):
#     if dominant_ratio_sp[s] < 0.2:
#         continue
    
    center = sp_properties[s, [1,0]].astype(np.int)
    angle = dm.angles[max_dir_sp[s]]

    length = max_response_sp_normalized[s]*100
    end = center + np.array([length*np.sin(angle), -length*np.cos(angle)], dtype=np.int)
    cv2.line(dirmap_vis2, tuple(center), tuple(end), (255,0,0), thickness=5, lineType=8, shift=0)
    
#     length = int(1./frequencies[max_freq_sp[s]])
#     end = center + np.array([length*np.cos(angle), length*np.sin(angle)], dtype=np.int)
#     cv2.line(dirmap_vis2, tuple(center), tuple(end), (255,0,0), 
#              thickness=5, lineType=8, shift=0)
    
    
#     cv2.line(dirmap_vis2, tuple(center), tuple(end), tuple(map(int, hc_colors[max_freq_sp[s]+1]*255)), 
#              thickness=5, lineType=8, shift=0)
            

# <codecell>

dm.save_pipeline_result(dirmap_vis2, 'dirMap', 'jpg')

