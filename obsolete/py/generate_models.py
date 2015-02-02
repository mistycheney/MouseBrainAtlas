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
    segm_params_id = 'blueNissl'
    vq_params_id = 'blueNissl'

# <codecell>

dm.set_image(args.stack_name, args.resolution, args.slice_ind)
dm.set_gabor_params(gabor_params_id=args.gabor_params_id)
dm.set_segmentation_params(segm_params_id=args.segm_params_id)
dm.set_vq_params(vq_params_id=args.vq_params_id)

# <codecell>

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

# def circle_list_to_labeling_field(self, circle_list):
#     label_circles = []
#     for c in circle_list:
#         label = np.where(np.all(self.colors == c.get_facecolor()[:3], axis=1))[0][0] - 1
#         label_circles.append((int(c.center[0]), int(c.center[1]), c.radius, label))
#     return label_circles

@timeit
def labeling_field_to_labelmap(labeling_field, size):
    
    labelmap = -1*np.ones(size, dtype=np.int)

    for cx,cy,cradius,label in labeling_field:
        for x in np.arange(cx-cradius, cx+cradius):
            for y in np.arange(cy-cradius, cy+cradius):
                if (cx-x)**2+(cy-y)**2 <= cradius**2:
                    labelmap[int(y),int(x)] = label
    return labelmap


def worker(i, labelmap, segmentation):
    in_sp_labels = labelmap[segmentation==i]

    counts = np.bincount(in_sp_labels+1)
    dominant_label = counts.argmax() - 1

    return dominant_label


@timeit
def label_superpixels(labelmap, segmentation):
    
        
    from joblib import Parallel, delayed

    labellist = np.array(Parallel(n_jobs=16)(delayed(worker)(i, labelmap, segmentation) for i in range(n_superpixels)))
    
#     labellist = -1*np.ones((n_superpixels,), dtype=np.int)

#     for sp in range(n_superpixels):
#         in_sp_labels = labelmap[segmentation==sp]
        
#         counts = np.bincount(in_sp_labels+1)
#         dominant_label = counts.argmax() - 1
#         if dominant_label != -1:
#             labellist[sp] = dominant_label

    return labellist
        

@timeit
def generate_models(labellist, sp_texton_hist_normalized):
    
    models = []
    for i in range(np.max(labellist)+1):
        sps = np.where(labellist == i)[0]
        model = {}
        if len(sps) > 0:
            texton_model = sp_texton_hist_normalized[sps, :].mean(axis=0)
            model['texton_hist'] = texton_model
            
            bboxes = sp_properties[sps, 4:]
            print i, sps
            
            row_min = bboxes[:,0].min()
            col_min = bboxes[:,1].min()
            row_max = bboxes[:,2].max()
            col_max = bboxes[:,3].max()
            
            model['bbox'] = (col_min, row_min, col_max-col_min, row_max-row_min)
            model['label'] = i
            
            models.append(model)

    n_models = len(models)
    print n_models, 'models'
    
    return models

@timeit
def models_from_labeling(labeling, segmentation):
    
    labelmap = labeling_field_to_labelmap(labeling['final_label_circles'], size=dm.image.shape)
    
    kernels = dm.load_pipeline_result('kernels', 'pkl')
    max_kern_size = max([k.shape[0] for k in kernels])
    
    cropped_labelmap = labelmap[max_kern_size/2:-max_kern_size/2, max_kern_size/2:-max_kern_size/2]
    
    labellist = label_superpixels(cropped_labelmap, segmentation)
    models = generate_models(labellist, texton_hists)
    
    return models

# <codecell>

# try:
#     existing_models = dm.load_pipeline_result('models', 'pkl')
# except:
# labeling = dm.load_labeling('anon_11032014025541')
labeling = dm.load_labeling('anon_11042014154647')

existing_models = models_from_labeling(labeling, cropped_segmentation)
n_existing_models = len(existing_models)
dm.save_pipeline_result(existing_models, 'models', 'pkl')

