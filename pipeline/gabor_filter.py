#! /usr/bin/env python

import os
import argparse
import sys
import time

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Execute feature extraction pipeline',
    epilog=""" 
    The following command processes image RS141_x5_0001.tif using the specified parameters.
    python %s RS141 1 -g blueNisslWide -s blueNisslRegular -v blueNissl
    """%(os.path.basename(sys.argv[0]), ))

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("slice_ind", type=int, help="slice index")
parser.add_argument("-g", "--gabor_params_id", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNisslWide')
parser.add_argument("-s", "--segm_params_id", type=str, help="segmentation parameters id (default: %(default)s)", default='blueNisslRegular')
parser.add_argument("-v", "--vq_params_id", type=str, help="vector quantization parameters id (default: %(default)s)", default='blueNissl')
args = parser.parse_args()


from joblib import Parallel, delayed

sys.path.append(os.path.join(os.environ['GORDON_REPO_DIR'], 'notebooks'))
from utilities2015 import *

os.environ['GORDON_DATA_DIR'] = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_processed'
os.environ['GORDON_REPO_DIR'] = '/oasis/projects/nsf/csd395/yuncong/Brain'
os.environ['GORDON_RESULT_DIR'] = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_results'

dm = DataManager(data_dir=os.environ['GORDON_DATA_DIR'], 
  repo_dir=os.environ['GORDON_REPO_DIR'], 
  result_dir=os.environ['GORDON_RESULT_DIR'], labeling_dir=os.environ['GORDON_LABELING_DIR'])

print 'reading image ...',
t = time.time()
dm.set_image(args.stack_name, args.slice_ind)
print 'done in', time.time() - t, 'seconds'

#============================================================

import numpy as np
from joblib import Parallel, delayed
from scipy.signal import fftconvolve

if dm.check_pipeline_result('features_rotated', 'npy'):
    print "features_rotated.npy already exists, skip"
else:
    
    if dm.check_pipeline_result('features', 'npy'):
        print "features.npy already exists, load"
        features = dm.load_pipeline_result('features', 'npy')
    else:

        t = time.time()
        print 'gabor filtering...',

        def convolve_per_proc(i):
            pf = fftconvolve(dm.image[dm.ymin-dm.max_kern_size : dm.ymax+1+dm.max_kern_size, 
                                      dm.xmin-dm.max_kern_size : dm.xmax+1+dm.max_kern_size], 
                             dm.kernels[i], 'same').astype(np.half)
            sys.stderr.write('filtered kernel %d\n'%i)

            return pf[dm.max_kern_size:-dm.max_kern_size, dm.max_kern_size:-dm.max_kern_size]

        filtered = Parallel(n_jobs=4)(delayed(convolve_per_proc)(i) for i in range(dm.n_kernel))
        features = np.asarray(filtered)

        del filtered

        dm.save_pipeline_result(features, 'features', 'npy')

        print 'done in', time.time() - t, 'seconds'


    def rotate_features(fs):
        features_tabular = fs.reshape((fs.shape[0], dm.n_freq, dm.n_angle))
        max_angle_indices = features_tabular.max(axis=1).argmax(axis=-1)
        features_rotated = np.reshape([np.roll(features_tabular[i], -ai, axis=-1) 
                                   for i, ai in enumerate(max_angle_indices)], (fs.shape[0], dm.n_freq * dm.n_angle))

        return features_rotated

    t = time.time()
    print 'rotate features ...',

    n_splits = 1000
    features_rotated = Parallel(n_jobs=16)(delayed(rotate_features)(fs) 
                               for fs in np.array_split(features.reshape((dm.n_kernel,-1)).T, n_splits))
    features_rotated = np.vstack(features_rotated)

    dm.save_pipeline_result(features_rotated, 'featuresRotated', 'npy')

    print 'done in', time.time() - t, 'seconds'


    t = time.time()
    print 'quantize feature vectors ...',

    n_texton = 100
        
    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=n_texton, batch_size=1000)
    kmeans.fit(features_rotated[::10])
    centroids = kmeans.cluster_centers_

    cluster_assignments = fclusterdata(centroids, 1.15, method="complete", criterion="inconsistent")
    # cluster_assignments = fclusterdata(centroids, 80., method="complete", criterion="distance"
    # cluster_assignments = fclusterdata(centroids, 1.1, method="complete", criterion="inconsistent")

    centroids = np.array([centroids[cluster_assignments == i].mean(axis=0) for i in set(cluster_assignments)])

    n_texton = len(centroids)
    print n_texton, 'reduced textons'

    dm.save_pipeline_result(centroids, 'centroids', 'npy')

    print 'done in', time.time() - t, 'seconds'

    del kmeans

        