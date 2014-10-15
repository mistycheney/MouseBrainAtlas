# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import argparse
import cPickle as pickle
import os 
import numpy as np

data_dir = '/home/yuncong/BrainLocal/DavidData'
repo_dir = '/home/yuncong/BrainSaliencyDetection'
params_dir = os.path.join(repo_dir, 'params')

# # parse arguments
# parser = argparse.ArgumentParser(
# formatter_class=argparse.RawDescriptionHelpFormatter,
# description='Supervised learning',
# # epilog="""
# # """%(os.path.basename(sys.argv[0]))
# )

# parser.add_argument("labeling_fn", type=str, help="path to labeling file")
# parser.add_argument("-o", "--output_dir", type=str, help="output directory (default: %(default)s)", default='/oasis/scratch/csd181/yuncong/output')
# args = parser.parse_args()

class args(object):    
    labeling_fn = '/home/yuncong/BrainLocal/DavidData/RS141/x5/0001/redNissl/labelings/RS141_x5_0001_redNissl_anon_10132014165928.pkl'


stack_name, resolution, slice_id, params_name, username, logout_time = os.path.basename(args.labeling_fn)[:-4].split('_')

results_dir = os.path.join(data_dir, stack_name, resolution, slice_id, params_name, 'pipelineResults')
labelings_dir = os.path.join(data_dir, stack_name, resolution, slice_id, params_name, 'labelings')

labeling = pickle.load(open(args.labeling_fn, 'r'))

instance_name = '_'.join([stack_name, resolution, slice_id, params_name])
parent_labeling_name = username + '_' + logout_time
    
def full_object_name(obj_name, ext):
    return os.path.join(data_dir, stack_name, resolution, slice_id, params_name, 'pipelineResults', instance_name + '_' + obj_name + '.' + ext)

sp_texton_hist_normalized = np.load(full_object_name('texHist', 'npy'))
sp_dir_hist_normalized = np.load(full_object_name('dirHist', 'npy'))
    
labellist = labeling['final_labellist']

models = []
for i in range(np.max(labellist)+1):
    sps = np.where(labellist == i)[0]
    model = {}
    if len(sps) > 0:
        texton_model = sp_texton_hist_normalized[sps, :].mean(axis=0)
        model['texton_hist'] = texton_model
        dir_model = sp_dir_hist_normalized[sps, :].mean(axis=0)
        model['dir_hist'] = dir_model
        models.append(model)        
        
n_models = len(models)
print n_models, 'models'

# <codecell>

segmentation = np.load(full_object_name('segmentation', 'npy'))
labelmap = labellist[segmentation]

# <codecell>

for l in range(n_models):
    matched_rows, matched_cols = np.where(labelmap == l)
    ymin = matched_rows.min()
    ymax = matched_rows.max()
    xmin = matched_cols.min()
    xmax = matched_cols.max()
    models[l]['bounding_box'] = (xmin, ymin, xmax-xmin+1, ymax-ymin+1)    

# <codecell>

models_fn = os.path.join(labelings_dir, instance_name+'_models.pkl')
pickle.dump(models, open(models_fn, 'w'))

