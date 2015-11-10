#! /usr/bin/env python

import os
import argparse
import sys
import time

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Generate features using Gabor filters')

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("slice_ind", type=int, help="slice index")
parser.add_argument("-g", "--gabor_params_id", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNisslWide')
# parser.add_argument("-s", "--segm_params_id", type=str, help="segmentation parameters id (default: %(default)s)", default='blueNisslRegular')
# parser.add_argument("-v", "--vq_params_id", type=str, help="vector quantization parameters id (default: %(default)s)", default='blueNissl')
args = parser.parse_args()


from joblib import Parallel, delayed

sys.path.append(os.path.join(os.environ['GORDON_REPO_DIR'], 'utilities'))
from utilities2015 import *

# dm = DataManager(gabor_params_id=args.gabor_params_id, 
#                  # segm_params_id=args.segm_params_id, 
#                  # vq_params_id=args.vq_params_id,
#                  stack=args.stack_name, 
#                  section=args.slice_ind)

# print 'reading image ...',
# t = time.time()
# dm._load_image(versions=['gray'])
# dm._generate_kernels()
# print 'done in', time.time() - t, 'seconds'

#============================================================

# import numpy as np
# from joblib import Parallel, delayed
# from scipy.signal import fftconvolve
# import bloscpack as bp

# try:
#     raise
#     features_rotated = dm.load_pipeline_result('featuresRotated')
# # if dm.check_pipeline_result('featuresRotated'):
#     print "features_rotated.npy already exists, skip"
# # else:
# except:
    
    # if dm.check_pipeline_result('features'):
    # # if False:
    #     print "features.npy already exists, load"
    #     features = dm.load_pipeline_result('features')
    # else:

os.system(os.environ['GORDON_REPO_DIR']+'/pipeline/gabor_filter_part.py %s %s 0' % (args.stack_name, args.slice_ind))
os.system(os.environ['GORDON_REPO_DIR']+'/pipeline/gabor_filter_part.py %s %s 1' % (args.stack_name, args.slice_ind))
os.system(os.environ['GORDON_REPO_DIR']+'/pipeline/gabor_filter_part.py %s %s 2' % (args.stack_name, args.slice_ind))
os.system(os.environ['GORDON_REPO_DIR']+'/pipeline/gabor_filter_part.py %s %s 3' % (args.stack_name, args.slice_ind))

    # assert dm.ymin-dm.max_kern_size >= 0 and dm.xmin-dm.max_kern_size >= 0 and \
    #         dm.ymax+1+dm.max_kern_size <= dm.image_height and dm.xmax+1+dm.max_kern_size <= dm.image_width, \
    #         'Not all pixels within the mask have value from the largest kernel'

    # def convolve_per_proc(i):
    #     pf = fftconvolve(dm.image[dm.ymin-dm.max_kern_size : dm.ymax+1+dm.max_kern_size, 
    #                               dm.xmin-dm.max_kern_size : dm.xmax+1+dm.max_kern_size], 
    #                      dm.kernels[i], 'same').astype(np.half)

    #     sys.stderr.write('filtered kernel %d\n'%i)

    #     bp.pack_ndarray_file(pf[dm.max_kern_size:-dm.max_kern_size, dm.max_kern_size:-dm.max_kern_size].copy(), 
    #                          os.environ['GORDON_RESULT_DIR']+'/features_%03d.bp'%i)


    # t = time.time()
    # print 'gabor filtering...',

    # Parallel(n_jobs=4)(delayed(convolve_per_proc)(i) for i in range(dm.n_kernel))

    # sys.stderr.write('done in %f seconds\n' % (time.time() - t))


# if not dm.check_pipeline_result('centroids'):

#     t = time.time()
#     sys.stderr.write('quantize feature vectors ...\n')

#     n_texton = 100

#     from sklearn.cluster import MiniBatchKMeans
#     kmeans = MiniBatchKMeans(n_clusters=n_texton, batch_size=1000)
#     kmeans.fit(features_rotated[::10])
#     centroids = kmeans.cluster_centers_

#     from scipy.cluster.hierarchy import fclusterdata

#     cluster_assignments = fclusterdata(centroids, 1.15, method="complete", criterion="inconsistent")
#     # cluster_assignments = fclusterdata(centroids, 80., method="complete", criterion="distance"
#     # cluster_assignments = fclusterdata(centroids, 1.1, method="complete", criterion="inconsistent")

#     centroids = np.array([centroids[cluster_assignments == i].mean(axis=0) for i in set(cluster_assignments)])

#     n_texton = len(centroids)
#     print n_texton, 'reduced textons'

#     dm.save_pipeline_result(centroids, 'centroids')

#     sys.stderr.write('done in %.2f seconds\n' % (time.time() - t))

#     del kmeans