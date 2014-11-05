# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from utilities import *

if 'SSH_CONNECTION' in os.environ:
    DATA_DIR = '/home/yuncong/DavidData'
    REPO_DIR = '/home/yuncong/Brain'
else:
    DATA_DIR = '/home/yuncong/BrainLocal/DavidData_v4'
    REPO_DIR = '/home/yuncong/BrainSaliencyDetection'

dm = DataManager(DATA_DIR, REPO_DIR)

class args:
    stack_name = 'RS141'
    resolution = 'x5'
    slice_ind = 2
    gabor_params_id = 'blueNisslWide'
    segm_params_id = 'blueNissl'
    vq_params_id = 'blueNissl'

dm.set_image(args.stack_name, args.resolution, args.slice_ind)
dm.set_gabor_params(gabor_params_id=args.gabor_params_id)
dm.set_segmentation_params(segm_params_id=args.segm_params_id)
dm.set_vq_params(vq_params_id=args.vq_params_id)

from joblib import Parallel, delayed

n_texton = int(dm.vq_params['n_texton'])

texton_hists = dm.load_pipeline_result('texHist', 'npy')

cropped_segmentation = dm.load_pipeline_result('cropSegmentation', 'npy')
n_superpixels = len(unique(cropped_segmentation)) - 1
cropped_mask = dm.load_pipeline_result('cropMask', 'npy')

textonmap = dm.load_pipeline_result('texMap', 'npy')
neighbors = dm.load_pipeline_result('neighbors', 'npy')

cropped_image = dm.load_pipeline_result('cropImg', 'tif')
sp_properties = dm.load_pipeline_result('cropSpProps', 'npy')

# <codecell>

from scipy.spatial.distance import cdist

overall_texton_hist = np.bincount(textonmap[cropped_mask].flat, minlength=n_texton)
overall_texton_hist_normalized = overall_texton_hist.astype(np.float) / overall_texton_hist.sum()
D_sp_null = np.squeeze(cdist(texton_hists, [overall_texton_hist_normalized], chi2))

# <codecell>

def compute_cluster_score(cluster, texton_hists, D_sp_null):
    model = texton_hists[list(cluster)].mean(axis=0)
    D_sp_model = np.squeeze(cdist([model], texton_hists[list(cluster)], chi2))
    model_sum = np.sum(D_sp_model)
    null_sum = np.sum(D_sp_null[list(cluster)])

    # can be made weighted by superpixel size
    
    score = null_sum - model_sum
    return score, null_sum, model_sum

# <codecell>

def grow_cluster(seed, neighbors, texton_hists, D_sp_null):
    
    curr_cluster = set([seed])
    frontier = [seed]
    
    curr_cluster_score, _, curr_model_score = compute_cluster_score(curr_cluster, texton_hists, D_sp_null)
    
    while len(frontier) > 0:
        u = frontier.pop(-1)
        for v in neighbors[u]:
            if v == -1 or v in curr_cluster: 
                continue

            score_new, _, model_sum_new = compute_cluster_score(curr_cluster | set([v]), texton_hists, D_sp_null)
            
            if score_new > curr_cluster_score and model_sum_new - curr_model_score < 0.5 :
                curr_cluster.add(v)
                frontier.append(v)
                curr_cluster_score, _, curr_model_score = compute_cluster_score(curr_cluster, texton_hists, D_sp_null)
            
            if len(curr_cluster) > 50:
                return curr_cluster
            
    return curr_cluster

# <codecell>

cluster_sp = Parallel(n_jobs=16)(delayed(grow_cluster)(s, neighbors, texton_hists, D_sp_null) 
                                 for s in range(n_superpixels))
cluster_sp = map(list, cluster_sp)

dm.save_pipeline_result(cluster_sp, 'clusters', 'pkl')

# <codecell>


