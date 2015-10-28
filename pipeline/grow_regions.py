#! /usr/bin/env python

import os
import argparse
import sys
import time

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Grow regions',
    epilog="")

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("slice_ind", type=int, help="slice index")
parser.add_argument("-g", "--gabor_params_id", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNisslWide')
parser.add_argument("-s", "--segm_params_id", type=str, help="segmentation parameters id (default: %(default)s)", default='blueNisslRegular')
parser.add_argument("-v", "--vq_params_id", type=str, help="vector quantization parameters id (default: %(default)s)", default='blueNissl')
args = parser.parse_args()

from joblib import Parallel, delayed

sys.path.append(os.path.join(os.environ['GORDON_REPO_DIR'], 'utilities'))
from utilities2015 import *

dm = DataManager(data_dir=os.environ['GORDON_DATA_DIR'], 
                 repo_dir=os.environ['GORDON_REPO_DIR'], 
                 result_dir=os.environ['GORDON_RESULT_DIR'], 
                 labeling_dir=os.environ['GORDON_LABELING_DIR'],
                 gabor_params_id=args.gabor_params_id, 
                 segm_params_id=args.segm_params_id, 
                 vq_params_id=args.vq_params_id,
                 stack=args.stack_name, 
                 section=args.slice_ind)

#======================================================

# seed_weight=0
# coherence_limit=0.5
# num_sp_percentage_limit=0.05
# min_size=3 min_distance=2
# threshold_abs=-0.1
# threshold_rel=0.06
# peakedness_limit=.002
# method='rc-mean'

dm.load_multiple_results(results=['texHist', 'segmentation', 'texMap', 'neighbors', 
                                  'edgeCoords', 'spCentroids', 'edgeNeighbors', 'dedgeNeighbors',
                                  'spCoords', 'spCentroids'])

def grow_cluster(*args, **kwargs):
    return dm.grow_cluster(*args, **kwargs)

def find_boundary_dedges_ordered(*args, **kwargs):
    return dm.find_boundary_dedges_ordered(*args, **kwargs)

try:
    raise
    all_seed_cluster_score_tuples = dm.load_pipeline_result('allSeedClusterScoreTuples')
    sys.stderr.write('allSeedClusterScoreTuples exists, skip ...\n')
    all_seeds, all_clusters, all_cluster_scores = zip(*all_seed_cluster_score_tuples)
except:
    sys.stderr.write('growing regions ...\n')
    t = time.time()
    expansion_clusters_tuples = Parallel(n_jobs=16)(delayed(grow_cluster)(s, verbose=False, all_history=False, 
                                                                         seed_weight=0,
                                                                        num_sp_percentage_limit=0.2,
                                                                     min_size=1, min_distance=5,
                                                                        threshold_abs=-0.1,
                                                                        threshold_rel=0.02,
                                                                       peakedness_limit=.002,
                                                                       method='rc-mean')
                                    for s in range(dm.n_superpixels))
    sys.stderr.write('done in %f seconds\n' % (time.time() - t))

    expansion_clusters, expansion_cluster_scores = zip(*expansion_clusters_tuples)

    all_seed_cluster_tuples = [(s,c) for s in range(dm.n_superpixels)  for c in expansion_clusters[s] ]
    all_cluster_scores = [c for s in range(dm.n_superpixels)  for c in expansion_cluster_scores[s]]

    all_seeds, all_clusters = zip(*all_seed_cluster_tuples)

    dm.save_pipeline_result(zip(all_seeds, all_clusters, all_cluster_scores), 'allSeedClusterScoreTuples')


sys.stderr.write('find boundary edges ...\n')
t = time.time()
all_cluster_dedges = Parallel(n_jobs=16)(delayed(find_boundary_dedges_ordered)(c) for c in all_clusters)
sys.stderr.write('done in %f seconds\n' % (time.time() - t))

dm.save_pipeline_result(zip(all_seeds, all_clusters, all_cluster_scores, all_cluster_dedges), 'allSeedClusterScoreDedgeTuples')