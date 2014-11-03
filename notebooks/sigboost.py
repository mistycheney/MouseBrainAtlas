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

# <codecell>

from grow_regions import grow_cluster

# <codecell>

from scipy.spatial.distance import cdist

overall_texton_hist = np.bincount(textonmap[cropped_mask].flat)
overall_texton_hist_normalized = overall_texton_hist.astype(np.float) / overall_texton_hist.sum()
D_sp_null = np.squeeze(cdist(texton_hists, [overall_texton_hist_normalized], chi2))

# <codecell>

# def circle_list_to_labeling_field(self, circle_list):
#     label_circles = []
#     for c in circle_list:
#         label = np.where(np.all(self.colors == c.get_facecolor()[:3], axis=1))[0][0] - 1
#         label_circles.append((int(c.center[0]), int(c.center[1]), c.radius, label))
#     return label_circles

def labeling_field_to_labelmap(labeling_field, size):
    
    labelmap = -1*np.ones(size, dtype=np.int)

    for cx,cy,cradius,label in labeling_field:
        for x in np.arange(cx-cradius, cx+cradius):
            for y in np.arange(cy-cradius, cy+cradius):
                if (cx-x)**2+(cy-y)**2 <= cradius**2:
                    labelmap[int(y),int(x)] = label
    return labelmap

@timeit
def label_superpixels(labelmap, segmentation):
    labellist = -1*np.ones((n_superpixels,), dtype=np.int)
    for sp in range(n_superpixels):
        in_sp_labels = labelmap[segmentation==sp]
        
        counts = np.bincount(in_sp_labels+1)
        dominant_label = counts.argmax() - 1
        if dominant_label != -1:
            labellist[sp] = dominant_label
    return labellist
        

@timeit
def generate_models(labellist, sp_texton_hist_normalized):
    
    models = []
    for i in range(np.max(labellist)+1):
        sps = np.where(labellist == i)[0]
        print i, sps
        model = {}
        if len(sps) > 0:
            texton_model = sp_texton_hist_normalized[sps, :].mean(axis=0)
            model['texton_hist'] = texton_model
            models.append(model)

    n_models = len(models)
    print n_models, 'models'
    
    return models


def models_from_labeling(labeling):
    
    labelmap = labeling_field_to_labelmap(labeling['final_label_circles'], size=dm.image.shape)
    
    kernels = dm.load_pipeline_result('kernels', 'pkl')
    max_kern_size = max([k.shape[0] for k in kernels])    
    
    cropped_labelmap = labelmap[max_kern_size/2:-max_kern_size/2, max_kern_size/2:-max_kern_size/2]
    
    labellist = label_superpixels(cropped_labelmap, cropped_segmentation)
    models = generate_models(labellist, texton_hists)
    
    return models

# <codecell>

try:
    models = dm.load_pipeline_result('models', 'pkl')
except:
    labeling = dm.load_labeling('anon_11032014025541')
    models = models_from_labeling(labeling)
    dm.save_pipeline_result(models, 'models', 'pkl')

# <codecell>

weights = np.ones((n_superpixels, ))/n_superpixels

# <codecell>

cluster_sp = Parallel(n_jobs=16)(delayed(grow_cluster)(s, neighbors, texton_hists, D_sp_null) for s in range(n_superpixels))
model_sp = [texton_hists[c].mean(axis=0) for c in clusters]
# clusters = [grow_cluster(s, neighbors, texton_hists, D_sp_null) for s in range(n_superpixels)]

# <codecell>

for t in range(n_models):
    
    print 'model %d' % (t)
 
    sig_score = np.zeros((n_superpixels, ))
    for i in range(n_superpixels):
        cluster = cluster_sp[i]
        model = model_sp[i]
        D_diff_cluster = D_sp_null[cluster] - np.squeeze(cdist([model], texton_hists[cluster], chi2))
        sig_score[i] = np.mean(weights[cluster] * D_diff_cluster)
 
    # Pick the most significant superpixel
    most_sig_sp = sig_score.argsort()[-1]
    print "most significant superpixel", most_sig_sp

    # models are the average of the distributions in the chosen superpixel's RE-cluster
    curr_cluster =  cluster_sp[most_sig_sp]
    D_sp_model = np.squeeze(cdist(model_sp[most_sig_sp], texton_hists, chi2))
    D_sp_diff = D_sp_null - D_sp_model
    
#     matched, _ = grow_cluster_likelihood_ratio(seed_sp, model_texton, model_dir)
#     matched = list(matched)

    # Reduce the weights of superpixels in LR-cluster
    weights[curr_cluster] = weights[curr_cluster] * np.exp(-5*(D_sp_null[curr_cluster] - D_sp_model[curr_cluster])**beta)
    
    weights = weights/weights.sum()

