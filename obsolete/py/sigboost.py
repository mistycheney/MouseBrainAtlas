# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%load_ext autoreload
%autoreload 2

%autosave 10

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
    slice_ind = 1
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

# <codecell>

sp_properties = dm.load_pipeline_result('cropSpProps', 'npy')

# <codecell>

from scipy.spatial.distance import cdist

overall_texton_hist = np.bincount(textonmap[cropped_mask].flat, minlength=n_texton)
overall_texton_hist_normalized = overall_texton_hist.astype(np.float) / overall_texton_hist.sum()
D_sp_null = np.squeeze(cdist(texton_hists, [overall_texton_hist_normalized], chi2))

# <codecell>

cluster_sp = dm.load_pipeline_result('clusters', 'pkl')
model_sp = [texton_hists[c].mean(axis=0) for c in cluster_sp]

# <codecell>

n_new_models = 5
new_models = find_new_models(n_new_models)

# <codecell>

def find_new_models(n_new_models):

    weights = np.ones((n_superpixels, ))/n_superpixels

    new_models = []
    
    for t in range(n_new_models):

        print 'model %d' % (t)

        sig_score = np.zeros((n_superpixels, ))
        for i in range(n_superpixels):
            cluster = cluster_sp[i]
            model = model_sp[i]
            D_diff_cluster = D_sp_null[cluster] - np.squeeze(cdist(model[np.newaxis, :], texton_hists[cluster], chi2))
            sig_score[i] = np.mean(weights[cluster] * D_diff_cluster)

        # Pick the most significant superpixel
        q = -1
        most_sig_sp = sig_score.argsort()[q]
        curr_cluster =  cluster_sp[most_sig_sp]
        while len(curr_cluster) < 5:
            q -= 1
            most_sig_sp = sig_score.argsort()[q]
            curr_cluster =  cluster_sp[most_sig_sp]

        print "most significant superpixel", most_sig_sp
        print 'curr_cluster', curr_cluster

        new_models.append(model_sp[i])
        
        D_sp_model = np.squeeze(cdist(model_sp[most_sig_sp][np.newaxis,:], texton_hists, chi2))
        D_sp_diff = D_sp_null - D_sp_model

        # Reduce the weights of superpixels in LR-cluster
        beta = 1.
    #     weights[curr_cluster] = weights[curr_cluster] * np.exp(-5*(D_sp_null[curr_cluster] - D_sp_model[curr_cluster])**beta)
        weights[curr_cluster] = 0

    #     weights = weights/weights.sum()

        weight_vis = weights[cropped_segmentation].copy()
        weight_vis[~cropped_mask] = 0

#         plt.matshow(weight_vis, cmap=plt.cm.Greys_r)
#         plt.colorbar()
#         plt.show()
        
    return new_models
        

# <codecell>

existing_models = dm.load_pipeline_result('models', 'pkl')

# <codecell>

existing_model_hists = np.array([m['texton_hist'] for m in existing_models])
D_sp_existing_model = np.squeeze(cdist(existing_model_hists, texton_hists, chi2))

# lr_decision_thresh = 0.1

def find_best_model(i):
    
    curr_cluster = cluster_sp[i]    
    model_score = np.mean(D_sp_null[curr_cluster][np.newaxis,:] - D_sp_existing_model[:, curr_cluster], axis=1)

    row_min, col_min, row_max, col_max = sp_properties[i, 4:]
    
    for mid, m in enumerate(existing_models):
        x, y, w, h = m['bbox']
        if not (row_min > y - 200 and col_min > x - 200 \
                and row_max < y + h + 200 and col_max < x + w + 200):
            model_score[mid] = -np.inf

    best_sig = model_score.max()
    
    if best_sig > lr_decision_thresh: # sp whose sig is smaller than this is assigned null
        return existing_models[model_score.argmax()]['label'], model_score
    else:        
        return -1, model_score


def assign_existing_models():
    
    res = Parallel(n_jobs=16)(delayed(find_best_model)(i) for i in range(n_superpixels))
    labels, model_scores = map(np.array, zip(*res))
    
    
    return labels, model_scores
    

# <codecell>

assigned_models, model_scores = assign_existing_models()

# <codecell>

hc_colors = np.loadtxt('../visualization/high_contrast_colors.txt', skiprows=1)/255.

ass_map = assigned_models[cropped_segmentation].copy()
ass_map[~cropped_mask] = -1

# <codecell>

ass_vis = label2rgb(ass_map, image=cropped_image, bg_label=-1, colors=hc_colors[1:])
ass_vis[~cropped_mask] = 0
plt.imshow(ass_vis)

# <codecell>

dm.save_pipeline_result(ass_vis, 'tmp', 'tif')

# <codecell>

# plt.bar(np.arange(n_texton), existing_model_hists[0], width=.8, color='k', alpha=.5)
plt.bar(np.arange(n_texton), existing_model_hists[1], width=.8, color='g', alpha=.5)
plt.bar(np.arange(n_texton), existing_model_hists[2], width=.8, color='r', alpha=.5)
plt.bar(np.arange(n_texton), existing_model_hists[3], width=.8, color='b', alpha=.5)

