# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%load_ext autoreload
%autoreload 2

from preamble import *

# <codecell>

segmentation = dm.load_pipeline_result('segmentation', 'npy')
n_superpixels = len(np.unique(segmentation)) - 1

textonmap = dm.load_pipeline_result('texMap', 'npy')

n_texton = len(np.unique(textonmap)) - 1

# try:
#     sp_texton_hist_normalized = dm.load_pipeline_result('texHist', 'npy')
    
# except:
    
def texton_histogram_worker(i):
    return np.bincount(textonmap[(segmentation == i)&(textonmap != -1)], minlength=n_texton)

r = Parallel(n_jobs=16)(delayed(texton_histogram_worker)(i) for i in range(n_superpixels))
sp_texton_hist = np.array(r)
sp_texton_hist_normalized = sp_texton_hist.astype(np.float) / sp_texton_hist.sum(axis=1)[:, np.newaxis] # denom might be invalid

dm.save_pipeline_result(sp_texton_hist_normalized, 'texHist', 'npy')

# compute the null texton histogram
# overall_texton_hist = np.bincount(textonmap[dm.mask].flat)
# overall_texton_hist_normalized = overall_texton_hist.astype(np.float) / overall_texton_hist.sum()

# <codecell>


