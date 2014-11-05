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
    REPO_DIR = '/home/yuncong/Brain'

dm = DataManager(DATA_DIR, REPO_DIR)

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

cropped_features = dm.load_pipeline_result('cropFeatures', 'npy')

# <codecell>

@timeit 
def save_numpy_compressed():
    np.savez_compressed('/tmp/test.npz', cropped_features)
    print os.path.getsize('/tmp/test.npz'), 'bytes'
#     os.remove('/tmp/test.npz')

save_numpy_compressed()

# <codecell>

import tables

@timeit
def save_pytables(complevel):
    filters = tables.Filters(complevel=complevel, complib='blosc')
    f = tables.open_file('/tmp/test', 'w')
    f.create_carray(f.root, 'features', obj=cropped_features, filters=filters)
    print os.path.getsize('/tmp/test'), 'bytes'
#     os.remove('/tmp/test')
    f.close()
    
    
save_pytables(1)

# <codecell>

save_pytables(9)

# <codecell>

del cropped_features

# <codecell>

f = tables.open_file('/tmp/test', 'r')
features = f.root.features
a = features[:]
print a.shape
f.close()

