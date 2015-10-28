#! /usr/bin/env python

import os
import argparse
import sys
import time

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Find growed cluster consensus',
    epilog="")

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("slice_ind", type=int, help="slice index")
parser.add_argument("-g", "--gabor_params_id", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNisslWide')
parser.add_argument("-s", "--segm_params_id", type=str, help="segmentation parameters id (default: %(default)s)", default='blueNisslRegular')
parser.add_argument("-v", "--vq_params_id", type=str, help="vector quantization parameters id (default: %(default)s)", default='blueNissl')
args = parser.parse_args()


from joblib import Parallel, delayed

sys.path.append(os.path.join(os.environ['GORDON_REPO_DIR'], 'notebooks'))
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


def compute_cluster_coherence_score(cluster, verbose=False):
    
    if len(cluster) > 1:
        cluster_avg = dm.texton_hists[cluster].mean(axis=0)
        ds = np.squeeze(chi2s([cluster_avg], dm.texton_hists[list(cluster)]))
        var = ds.mean()
    else:
        var = 0
    
    return var

def compute_cluster_significance_score(*args, **kwargs):
    return dm.compute_cluster_score(*args, **kwargs)[0]


from scipy.spatial.distance import cdist, pdist, squareform
from scipy.cluster.hierarchy import average, fcluster, leaders, complete, single, dendrogram, ward

from collections import defaultdict, Counter
from itertools import combinations, chain, product

import networkx

from clustering import *


def scores_to_vote(scores):
    vals = np.unique(scores)
    d = dict(zip(vals, np.linspace(0, 1, len(vals))))
    votes = np.array([d[s] for s in scores])
    votes = votes/votes.sum()
    return votes

surround_high_contrast_thresh = .1
coherence_limit = .25
# significance_limit = .8
area_limit = 60000
nonoverlapping_area_limit = 2.
bg_texton = 3
bg_texton_percentage = .4
# significance_limit = -0.81
significance_limit = 0
consensus_limit = -20

dm.load_multiple_results(results=['texHist', 'segmentation', 'texMap', 'neighbors', 
                                  'edgeCoords', 'spCentroids', 'edgeNeighbors', 'dedgeNeighbors',
                                  'spCoords', 'spAreas', 'edgeMidpoints'])

try:
    raise
    good_clusters = dm.load_pipeline_result('goodClusters')
    good_dedges = dm.load_pipeline_result('goodDedges')

except:

    all_seed_cluster_score_dedge_tuples = dm.load_pipeline_result('allSeedClusterScoreDedgeTuples')
    all_seed, all_clusters, all_cluster_scores, all_cluster_dedges = zip(*all_seed_cluster_score_dedge_tuples)


    d = defaultdict(list)
    for se, cl in zip(all_seed, all_clusters):
        d[se].append(cl)
        
    all_cluster_consensus = []
    for se, cl in zip(all_seed, all_clusters):
        total_diff = 0
        c = 0
        for s in cl:
            if s != se:
                if len(d[s]) > 0:
                    diff_size = np.min([len((set(cl2)|set(cl))-(set(cl2)&set(cl))) for cl2 in d[s]])
                    total_diff += diff_size
                    c += 1
        if c > 0:
            mean_diff = total_diff/float(c)
            all_cluster_consensus.append(-mean_diff)
        else:
            all_cluster_consensus.append(-np.inf)

    all_cluster_consensus = np.array(all_cluster_consensus)


    all_cluster_sigs = np.array(all_cluster_scores)
    all_cluster_sigs_perc = np.array([compute_cluster_significance_score(cl, method='percentage-soft',
                                                               thresh=surround_high_contrast_thresh) 
                             for cl in all_clusters])
    all_cluster_coherences = np.array([compute_cluster_coherence_score(cl) for cl in all_clusters])
    all_cluster_hists = [dm.texton_hists[cl].mean(axis=0) for cl in all_clusters]
    all_cluster_entropy = np.nan_to_num([-np.sum(hist[hist!=0]*np.log(hist[hist!=0])) for hist in all_cluster_hists])

    all_cluster_centroids = np.array([dm.sp_centroids[cl, ::-1].mean(axis=0) for cl in all_clusters])

    all_cluster_compactness = np.array([len(eds)**2/float(len(cl)) for cl, eds in zip(all_clusters, all_cluster_dedges)])
    all_cluster_compactness = .001 * np.maximum(all_cluster_compactness-40,0)**2
    
    all_cluster_area = np.array([dm.sp_areas[cl].sum() for cl in all_clusters])

    remaining_cluster_indices = [i for i, (cl, coh, sig, perc, ent, cent, comp, area, cons, hist) in enumerate(zip(all_clusters, 
                                                                                      all_cluster_coherences, 
                                                                                      all_cluster_sigs,
                                                                                    all_cluster_sigs_perc,
                                                                                      all_cluster_entropy,
                                                                                      all_cluster_centroids,
                                                                                      all_cluster_compactness,
                                                                                      all_cluster_area,
                                                                                    all_cluster_consensus,
                                                                                    all_cluster_hists)) 
            if coh < coherence_limit and sig > significance_limit and area > area_limit \
                             and cons > consensus_limit and \
#                  comp < 50 and \
             ((ent > 1.5 and hist[bg_texton] < bg_texton_percentage) or \
              (cent[0] - dm.xmin > 800 and \
               dm.xmax - cent[0] > 800 and \
               cent[1] - dm.ymin > 800 and \
               dm.ymax - cent[1] > 800)
             )]

    sys.stderr.write('remaining_cluster_indices = %d\n'%len(remaining_cluster_indices))

    all_remaining_seed = [all_seed[i] for i in remaining_cluster_indices]
    all_remaining_clusters = [all_clusters[i] for i in remaining_cluster_indices]
    all_remaining_cluster_sigs = [all_cluster_sigs[i] for i in remaining_cluster_indices]
    all_remaining_cluster_sigpercs = [all_cluster_sigs_perc[i] for i in remaining_cluster_indices]
    all_remaining_cluster_coherences = [all_cluster_coherences[i] for i in remaining_cluster_indices]
    all_remaining_cluster_dedges = [all_cluster_dedges[i] for i in remaining_cluster_indices]
    all_remaining_cluster_consensus = [all_cluster_consensus[i] for i in remaining_cluster_indices]
    all_remaining_seed_cluster_score_dedge_tuples = [all_seed_cluster_score_dedge_tuples[i] for i in remaining_cluster_indices]

    sys.stderr.write('group clusters ...\n')
    t = time.time()

    all_remaining_seed_cluster_sig_coh_dedge_tuples = zip(all_remaining_seed, all_remaining_clusters, 
                                                all_remaining_cluster_sigs, 
                                              all_remaining_cluster_sigpercs,
                                              all_remaining_cluster_coherences,
                                             all_remaining_cluster_dedges,
                                                         all_remaining_cluster_consensus)

    # tuple_indices_grouped, tuples_grouped, _ = group_tuples(all_seed_cluster_sig_coh_dedge_tuples, 
    #                                                          val_ind = 1,
    #                                                          metric='jaccard',
    #                                                          dist_thresh=.2)

    # merge if area difference is less than 1% of entire frame
    tuple_indices_grouped, tuples_grouped, _ = group_tuples(all_remaining_seed_cluster_sig_coh_dedge_tuples, 
                                                             val_ind = 1,
                                                             metric='nonoverlap-area',
                                                             dist_thresh=nonoverlapping_area_limit, 
                                                            sp_areas=dm.sp_areas)

    n_group = len(tuple_indices_grouped)
    sys.stderr.write('%d groups\n'%n_group)

    sys.stderr.write('done in %.2f seconds ...\n' % (time.time() - t))


    all_seeds_grouped, all_clusters_grouped, \
    all_sigs_grouped, all_sigpercs_grouped, all_cohs_grouped, all_dedges_grouped,\
    all_consensus_grouped = [list(map(list, lst)) for lst in zip(*[zip(*g) for g in tuples_grouped])]

    all_cluster_grouped_union = [set.union(*map(set, cls)) for cls in all_clusters_grouped]
    all_scores_grouped = np.array([-dm.sp_areas[list(set(union_cl)-set(seeds))].sum()
                                   for seeds, union_cl in zip(all_seeds_grouped, all_cluster_grouped_union)])


    group_rep_indices = map(np.argmax, all_sigs_grouped)

    group_rep_clusters = [cls[rep] for cls, rep in zip(all_clusters_grouped, group_rep_indices)]
    group_contrasts = [compute_cluster_significance_score(cl, method='rc-mean') for cl in group_rep_clusters]
    
    # group_inbreed = all_scores_grouped
    group_consensus = np.array([vs[rep] for vs, rep in zip(all_consensus_grouped, group_rep_indices)])
    group_size = [len(g) for g in all_clusters_grouped]


    d1 = scores_to_vote(group_contrasts)
    # d2 = scores_to_vote(group_inbreed)
    d3 = scores_to_vote(group_size)
    d4 = scores_to_vote(group_consensus)
    group_indices_ranked = np.argsort(.5*d1 + .4*d4 + .1*d3)[::-1]

    # rep_inbreed_ranked = [group_inbreed[i] for i in group_indices_ranked]
    # rep_contrast_ranked = [group_contrasts[i] for i in group_indices_ranked]
    rep_clusters_ranked = [all_clusters_grouped[i][np.argmax(all_sigpercs_grouped[i])] for i in group_indices_ranked]
    rep_dedges_ranked = [dm.find_boundary_dedges_ordered(cl) for cl in rep_clusters_ranked]

    good_clusters = rep_clusters_ranked
    good_dedges = rep_dedges_ranked

    dm.save_pipeline_result(good_clusters, 'goodClusters')
    dm.save_pipeline_result(good_dedges, 'goodDedges')


os.system('rm %s/*%s*landmarks*Viz.jpg'%(dm.results_dir, dm.segm_params_id))


# for i in range(0, len(good_dedges), 10):
for i in range(0, 100, 10):
    viz = dm.visualize_edge_sets(good_dedges[i:i+10], show_set_index=True)
    try:
        dm.save_pipeline_result(viz, 'landmarks%dViz'%(i+10))
    except:
        pass


def fit_ellipse_to_points(pts):

    pts = np.array(list(pts) if isinstance(pts, set) else pts)
    c0 = pts.mean(axis=0)
    coords0 = pts - c0

    U,S,V = np.linalg.svd(np.dot(coords0.T, coords0)/coords0.shape[0])
    v1 = U[:,0]
    v2 = U[:,1]
    s1 = np.sqrt(S[0])
    s2 = np.sqrt(S[1])

    return v1, v2, s1, s2, c0


boundary_models = []

for i, (cl, dedges) in enumerate(zip(good_clusters, good_dedges)[:100]):

    dedge_list = list(dedges)

    interior_texture = dm.texton_hists[list(cl)].mean(axis=0)
    exterior_textures = []
    
    cluster_coords = np.vstack([dm.sp_coords[s] for s in cl])
    ell = fit_ellipse_to_points(cluster_coords)
    
    edge_points = []
    
    for e in dedge_list:
        pts_e = dm.edge_coords[frozenset(e)]
        sample_indices = np.arange(20, len(pts_e)-20, 200)

        if len(sample_indices) > 0:
            sample_pts_e = pts_e[sample_indices]
            edge_points.append(sample_pts_e)
            surr = e[0]
            ext_tex = dm.texton_hists[surr] if surr != -1 else np.nan * np.ones((dm.n_texton,))
            exterior_textures.append([ext_tex for _ in sample_indices])
    
    edge_points = np.vstack(edge_points)
    exterior_textures = np.vstack(exterior_textures)

    center = np.mean([dm.edge_midpoints[frozenset(e)] for e in dedge_list], axis=0)

    boundary_models.append((dedge_list, interior_texture, exterior_textures, edge_points, center) +\
                           ell)

dm.save_pipeline_result(boundary_models, 'boundaryModels')