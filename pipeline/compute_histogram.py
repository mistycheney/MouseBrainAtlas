#! /usr/bin/env python

import os
import argparse
import sys
import time

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Compute texton histograms')

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("slice_ind", type=int, help="slice index")
parser.add_argument("texton_path", type=str, help="path to textons.npy")
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

dm = DataManager(data_dir=os.environ['GORDON_DATA_DIR'], repo_dir=os.environ['GORDON_REPO_DIR'], 
    result_dir=os.environ['GORDON_RESULT_DIR'], labeling_dir=os.environ['GORDON_LABELING_DIR'],
    stack=args.stack_name, section=args.slice_ind)

#==================================================

if dm.check_pipeline_result('texMap', 'npy') and dm.check_pipeline_result('texMap', 'jpg'):
    print "texMap.npy already exists, skip"

    textonmap = dm.load_pipeline_result('texMap', 'npy')
    n_texton = textonmap.max() + 1

else:

    print 'loading centroids and features ...',
    t = time.time()
    # 	centroids = dm.load_pipeline_result('textons', 'npy')
    centroids = np.load(args.texton_path)
    features_rotated = dm.load_pipeline_result('featuresRotated', 'npy')
    print 'done in', time.time() - t, 'seconds'

    n_texton = len(centroids)

    print 'assign textons ...',
    t = time.time()

    # from sklearn.cluster import MiniBatchKMeans
    from scipy.spatial.distance import cdist

    # kmeans = MiniBatchKMeans(n_clusters=n_texton, batch_size=1000, init=centroids, max_iter=1)
    # kmeans.fit(features_rotated)
    # # final_centroids = kmeans.cluster_centers_
    # labels = kmeans.labels_

    label_list = []
    for fs in np.array_split(features_rotated, 3):
        D = cdist(fs, centroids)
        labels = np.argmin(D, axis=1)
        label_list.append(labels)
    labels = np.concatenate(label_list)

    print 'done in', time.time() - t, 'seconds'

    # dm._load_image()
    textonmap = -1 * np.ones((dm.image_height, dm.image_width), dtype=np.int)
    textonmap[dm.mask] = labels

    dm.save_pipeline_result(textonmap, 'texMap', 'npy')

    colors = np.loadtxt(dm.repo_dir + '/visualization/100colors.txt')
    
    textonmap_viz = np.zeros((dm.image_height, dm.image_width, 3), np.uint8)
    textonmap_viz[dm.mask] = colors[textonmap[dm.mask]]
    dm.save_pipeline_result(textonmap_viz, 'texMap', 'jpg')


if dm.check_pipeline_result('texHist', 'npy'):
	print "texHist.npy already exists, skip"

else:
	
    print 'computing histograms ...',
    t = time.time()

    segmentation = dm.load_pipeline_result('segmentation', 'npy')
    n_superpixels = segmentation.max() + 1

    def texton_histogram_worker(i):
        # return np.bincount(textonmap[(segmentation == i)&(textonmap != -1)], minlength=n_texton)
        return np.bincount(textonmap[segmentation == i], minlength=n_texton)

    r = Parallel(n_jobs=8)(delayed(texton_histogram_worker)(i) for i in range(n_superpixels))
    sp_texton_hist = np.array(r)
    sp_texton_hist_normalized = sp_texton_hist.astype(np.float) / sp_texton_hist.sum(axis=1)[:, np.newaxis] # denom might be invalid

    dm.save_pipeline_result(sp_texton_hist_normalized, 'texHist', 'npy')

    print 'done in', time.time() - t, 'seconds'

