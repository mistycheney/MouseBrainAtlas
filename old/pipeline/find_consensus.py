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

from clustering import *


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

def compute_cluster_consensus_score(se, cl):
    diff_sizes = [np.min([len((set(cl2)|set(cl))-(set(cl2)&set(cl))) for cl2 in growed_from[s]]) 
                  for s in cl if s != se and len(growed_from[s]) > 0]
    if len(diff_sizes) > 0:
        mean_diff = np.mean(diff_sizes)
        return -mean_diff
    else:
        return -np.inf

def scores_to_vote(scores):
    vals = np.unique(scores)
    d = dict(zip(vals, np.linspace(0, 1, len(vals))))
    votes = np.array([d[s] for s in scores])
    votes = votes/votes.sum()
    return votes

coherence_limit = .25
area_limit = 60000
nonoverlapping_area_limit = 2.
bg_texton = 3
bg_texton_percentage = .2
significance_limit = 0.05
consensus_limit = -20

dm.load_multiple_results(results=['texHist', 'segmentation', 'texMap', 'neighbors', 
                                  'edgeCoords', 'spCentroids', 'edgeNeighbors', 'dedgeNeighbors',
                                  'spCoords', 'edgeMidpoints', 'spAreas'])

try:
    raise
    proposal_tuples = dm.load_pipeline_result('proposals')
    rep_clusters_ranked, rep_dedges_ranked, rep_sigs_ranked = zip(*proposal_tuples)

except:

    all_seed_cluster_score_dedge_tuples = dm.load_pipeline_result('allSeedClusterScoreDedgeTuples')
    all_seeds, all_clusters, all_cluster_scores, all_cluster_dedges = zip(*all_seed_cluster_score_dedge_tuples)
    sys.stderr.write('%d proposals\n'%len(all_clusters))

    all_clusters_unique_dict = {}
    for i, cl in enumerate(all_clusters):
      all_clusters_unique_dict[frozenset(cl)] = i

    all_unique_cluster_indices = all_clusters_unique_dict.values()
    all_unique_clusters = [all_clusters[i] for i in all_unique_cluster_indices]
    all_unique_dedges = [all_cluster_dedges[i] for i in all_unique_cluster_indices]

    all_unique_cluster_scores = [all_cluster_scores[i] for i in all_unique_cluster_indices]
    all_unique_seeds = [all_seeds[i] for i in all_unique_cluster_indices]

    sys.stderr.write('%d unique proposals\n'%len(all_unique_clusters))

    growed_from = defaultdict(list)
    for se, cl in zip(all_seeds, all_clusters):
        growed_from[se].append(cl)
        
    all_cluster_consensus = Parallel(n_jobs=16)(delayed(compute_cluster_consensus_score)(se, cl) 
                                                    for se, cl in zip(all_unique_seeds, all_unique_clusters))
    all_cluster_consensus = np.array(all_cluster_consensus)

    all_cluster_sigs = np.array(all_unique_cluster_scores)
    all_cluster_coherences = np.array([compute_cluster_coherence_score(cl) for cl in all_unique_clusters])

    all_cluster_hists = [dm.texton_hists[cl].mean(axis=0) for cl in all_unique_clusters]
    all_cluster_entropy = np.nan_to_num([-np.sum(hist[hist!=0]*np.log(hist[hist!=0])) for hist in all_cluster_hists])

    all_cluster_centroids = np.array([dm.sp_centroids[cl, ::-1].mean(axis=0) for cl in all_unique_clusters])

    all_cluster_area = np.array([dm.sp_areas[cl].sum() for cl in all_unique_clusters])


    remaining_cluster_indices = [i for i, (cl, coh, sig, ent, cent, area, cons, hist) in enumerate(zip(all_clusters, 
                                                                                          all_cluster_coherences, 
                                                                                          all_cluster_sigs,
    #                                                                                     all_cluster_sigs_perc,
                                                                                          all_cluster_entropy,
                                                                                          all_cluster_centroids,
    #                                                                                       all_cluster_compactness,
                                                                                          all_cluster_area,
                                                                                        all_cluster_consensus,
                                                                                        all_cluster_hists)) 
                if coh < coherence_limit and sig > significance_limit and \
                    area > area_limit and cons > consensus_limit and \
    #                  comp < 50 and \
                 ((ent > 1.5 and hist[bg_texton] < bg_texton_percentage) or \
                  (cent[0] - dm.xmin > 800 and \
                   dm.xmax - cent[0] > 800 and \
                   cent[1] - dm.ymin > 800 and \
                   dm.ymax - cent[1] > 800)
                 )]


    sys.stderr.write('remaining_cluster_indices = %d\n'%len(remaining_cluster_indices))

    all_remaining_seeds = [all_unique_seeds[i] for i in remaining_cluster_indices]
    all_remaining_clusters = [all_unique_clusters[i] for i in remaining_cluster_indices]
    all_remaining_cluster_dedges = [all_unique_dedges[i] for i in remaining_cluster_indices]
    all_remaining_cluster_sigs = [all_cluster_sigs[i] for i in remaining_cluster_indices]
    all_remaining_cluster_hists = [all_cluster_hists[i] for i in remaining_cluster_indices]
    
    all_remaining_overlap_mat = compute_pairwise_distances(all_remaining_clusters, metric='overlap-size')
    all_remaining_histdist_mat = compute_pairwise_distances(all_remaining_cluster_hists, chi2)
    all_remaining_distmat = ~((all_remaining_overlap_mat > 0) & (all_remaining_histdist_mat < .2))

    t = time.time()

    all_remaining_cluster_tuples = zip(all_remaining_seeds, all_remaining_clusters, all_remaining_cluster_dedges,
                                       all_remaining_cluster_hists, all_remaining_cluster_sigs)

    cluster_indices_grouped, tuples_grouped, _ = group_tuples(all_remaining_cluster_tuples, 
                                                            val_ind = 1,
                                                            distance_matrix=all_remaining_distmat,
                                                           dist_thresh=.00001,
                                                           linkage='complete')

    print time.time() - t

    n_group = len(cluster_indices_grouped)
    sys.stderr.write('%d groups\n'%n_group)


    group_internal_order = [np.argsort(map(itemgetter(4), tuple_group))[::-1] for tuple_group in tuples_grouped]
    cluster_indices_grouped = [[ci_group[i] for i in order] for order, ci_group in zip(group_internal_order, cluster_indices_grouped)]
    tuples_grouped = [[tuple_group[i] for i in order] for order, tuple_group in zip(group_internal_order, tuples_grouped)]

    all_seeds_grouped, all_clusters_grouped, all_dedges_grouped, \
    all_hists_grouped, all_sigs_grouped = [list(map(list, lst)) for lst in zip(*[zip(*g) for g in tuples_grouped])]


    group_rep_indices = map(np.argmax, all_sigs_grouped)

    group_rep_clusters = [cls[rep] for cls, rep in zip(all_clusters_grouped, group_rep_indices)]

    group_contrasts = [compute_cluster_significance_score(cl, method='rc-mean') for cl in group_rep_clusters]

    group_size = [len(g) for g in all_clusters_grouped]

    d1 = scores_to_vote(group_contrasts)
    d3 = scores_to_vote(group_size)
    group_indices_ranked = np.argsort(.5*d1 + 0*d3)[::-1]

    rep_sigs_ranked = [group_contrasts[i] for i in group_indices_ranked]
    rep_clusters_ranked = [all_clusters_grouped[i][group_rep_indices[i]] for i in group_indices_ranked]
    rep_dedges_ranked = [dm.find_boundary_dedges_ordered(cl) for cl in rep_clusters_ranked]

    dm.save_pipeline_result(zip(rep_clusters_ranked, rep_dedges_ranked, rep_sigs_ranked), 'proposals')

# os.system('rm %s/*%s*landmarks*Viz.jpg'%(dm.results_dir, dm.segm_params_id))

# # for i in range(0, len(good_dedges), 10):
# for i in range(0, 100, 10):
#     viz = dm.visualize_edge_sets(good_dedges[i:i+10], show_set_index=True)
#     try:
#         dm.save_pipeline_result(viz, 'landmarks%dViz'%(i+10))
#     except:
#         pass

sp_covered_by = defaultdict(set)
for i, cl in enumerate(rep_clusters_ranked):
    for s in cl:
        sp_covered_by[s].add(i)

dm.save_pipeline_result(sp_covered_by, 'spCoveredByProposals')

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

for i, (cl, dedges, sig) in enumerate(zip(rep_clusters_ranked, rep_dedges_ranked, rep_sigs_ranked)[:100]):

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

    boundary_models.append((cl, dedge_list, sig, interior_texture, exterior_textures, edge_points, center) +\
                           ell)

dm.save_pipeline_result(boundary_models, 'boundaryModels')