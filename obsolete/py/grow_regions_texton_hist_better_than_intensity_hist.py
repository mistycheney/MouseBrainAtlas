# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%load_ext autoreload
%autoreload 2

from preamble import *

# <codecell>

texton_hists = dm.load_pipeline_result('texHist', 'npy')

segmentation = dm.load_pipeline_result('segmentation', 'npy')
n_superpixels = len(unique(segmentation)) - 1

textonmap = dm.load_pipeline_result('texMap', 'npy')
n_texton = len(np.unique(textonmap)) - 1

neighbors = dm.load_pipeline_result('neighbors', 'npy')

sp_properties = dm.load_pipeline_result('spProps', 'npy')

segmentation_vis = dm.load_pipeline_result('segmentationWithText', 'jpg')

# <codecell>

def f(s):
    h, _ = np.histogram(dm.image[segmentation==s], bins=np.linspace(0, 255, 10), density=True)
    return h

n_intensities = 9
intensity_hists = Parallel(n_jobs=16)(delayed(f)(s) for s in range(n_superpixels))
intensity_hists = np.array(intensity_hists)

# <codecell>

# 3223 is texturally closer to 3227 than 3055
# texton hists reflect this, but intensity hists do not

print js(intensity_hists[3227], intensity_hists[3055])
print js(intensity_hists[3227], intensity_hists[3223])

# <codecell>

print js(texton_hists[3227], texton_hists[3055])
print js(texton_hists[3227], texton_hists[3223])

