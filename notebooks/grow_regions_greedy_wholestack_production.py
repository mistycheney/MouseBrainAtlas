
# In[14]:

def alpha_blending(src_rgb, dst_rgb, src_alpha, dst_alpha):
    
    out_alpha = src_alpha + dst_alpha * (1. - src_alpha)
    out_rgb = (src_rgb * src_alpha[..., None] +
               dst_rgb * dst_alpha[..., None] * (1. - src_alpha[..., None])) / out_alpha[..., None]
    
    out = np.zeros((src_rgb.shape[0], src_rgb.shape[1], 4))
        
    out[..., :3] = out_rgb
    out[..., 3] = out_alpha
    
    return out


def compute_cluster_score(cluster, texton_hists, neighbors, output=False):
    
    cluster_list = list(cluster)    
    cluster_avg = texton_hists[cluster_list].mean(axis=0)
    
    surrounds = set([i for i in set.union(*[neighbors[c] for c in cluster]) if i not in cluster and i != -1])
    surrounds_list = list(surrounds)

#     f_avg = texton_freqs[cluster_list].sum(axis=0)
    
#     interior_pvals = [chi2pval(f_avg, texton_freqs[i])[0] for i in cluster_list]
#     interior_pval = np.mean(interior_pvals)
    interior_pval = 0

#     surround_pvals = [chi2pval(f_avg, texton_freqs[i])[0] for i in surrounds_list]
#     surround_pval = np.max(surround_pvals)
    surround_pval = 0
#     assert not np.isnan(surround_pval), (cluster_list, surrounds_list[where(np.isnan(surround_pvals))[0]])

    interior_dist = np.squeeze(cdist([cluster_avg], texton_hists[cluster_list], chi2)).mean()
    surround_dist = np.squeeze(cdist([cluster_avg], texton_hists[surrounds_list], chi2)).min()

#     surround_stats = [chi2pval(f_avg, texton_freqs[i])[1] for i in surrounds_list]
#     surround_stat = np.min(surround_stats) 
    
#     compactness = 0
    compactness = len(find_boundaries([cluster], neighbors=neighbors)[0])**2/float(len(cluster))
    compactness = .001 * np.maximum(compactness-40,0)**2
    
    size_prior = .1 * (1-np.exp(-.8*len(cluster)))
    
#     score = 0. * surround_dist - interior_dist -  .0001 * np.maximum(compactness-50,0)**2
#     score = - surround_pval + interior_pval
#     score = - interior_dist
#     score = surround_dist - 0 * interior_dist - compactness
#     score = - surround_pval - 0 * interior_dist - compactness + size_prior
    score = surround_dist - 0 * interior_dist - compactness + size_prior
    
#     return score, surround_pval, interior_pval, compactness
    return score, surround_dist, interior_dist, compactness, surround_pval, interior_pval, size_prior
#     return score, surround_stat, interior_dist, compactness


# In[19]:

def grow_cluster3(seed, neighbors, texton_hists, output=False, all_history=False):
            
    visited = set([])
    curr_cluster = set([])
        
    candidate_scores = [0]
    candidate_sps = [seed]

    score_tuples = []
    added_sps = []
    
    iter_ind = 0
        
    while len(candidate_sps) > 0:

        best_ind = np.argmax(candidate_scores)
        
        heuristic = candidate_scores[best_ind]
        sp = candidate_sps[best_ind]
        
        del candidate_scores[best_ind]
        del candidate_sps[best_ind]
        
        if sp in curr_cluster:
            continue
                
        iter_ind += 1
        curr_cluster.add(sp)
        added_sps.append(sp)
        
        tt = compute_cluster_score(curr_cluster, texton_hists=texton_hists, neighbors=neighbors)
        tot, exterior, interior, compactness, surround_pval, interior_pval, size_prior = tt
        score_tuples.append(np.r_[heuristic, tt])
        
        if output:
            print 'iter', iter_ind, 'add', sp

        visited.add(sp)
        
        candidate_sps = (set(candidate_sps) | (neighbors[sp] - set([-1])) | (visited - curr_cluster)) - curr_cluster
        candidate_sps = list(candidate_sps)
        
#         f_avg = texton_freqs[list(curr_cluster)].sum(axis=0)
#         candidate_scores = [chi2pval(f_avg, texton_freqs[i])[0] for i in candidate_sps]

        h_avg = texton_hists[list(curr_cluster)].mean(axis=0)
        candidate_scores = [-chi2(h_avg, texton_hists[i]) for i in candidate_sps]

#         candidate_scores = [compute_cluster_score(curr_cluster | set([s])) for s in candidate_sps]
                
        if len(visited) > int(n_superpixels * 0.03):
            break

    score_tuples = np.array(score_tuples)
    
    min_size = 2
    scores = score_tuples[:,1]
    cutoff = np.argmax(scores[min_size:]) + min_size
    
    if output:
        print 'cutoff', cutoff

    final_cluster = added_sps[:cutoff]
    final_score = scores[cutoff]
    
    if all_history:
        return list(final_cluster), final_score, added_sps, score_tuples
    else:
        return list(final_cluster), final_score


# In[20]:

def visualize_cluster(cluster, segmentation, segmentation_vis, text=False, highlight_seed=False):

    a = -1*np.ones_like(segmentation)        

    for i, c in enumerate(cluster):
        if highlight_seed:
            if i == 0:
                a[segmentation == c] = 1       
            else:
                a[segmentation == c] = 0
        else:
            a[segmentation == c] = 0

    vis = label2rgb(a, image=segmentation_vis)

    vis = img_as_ubyte(vis[...,::-1])

    if text:

        import cv2

        for i, sp in enumerate(cluster):
            vis = cv2.putText(vis, str(i), 
                              tuple(np.floor(sp_properties[sp, [1,0]] - np.array([10,-10])).astype(np.int)), 
                              cv2.FONT_HERSHEY_DUPLEX,
                              1., ((0,255,255)), 1)

    return vis.copy()

def visualize_multiple_clusters(clusters, segmentation, segmentation_vis, alpha_blend=True):
    
    colors = np.loadtxt('../visualization/100colors.txt')
    n_superpixels = segmentation.max() + 1
    
    mask_alpha = .4
    
    if alpha_blend:
        
        for ci, c in enumerate(clusters):
            m =  np.zeros((n_superpixels,), dtype=np.float)
            m[list(c)] = mask_alpha
            alpha = m[segmentation]
            alpha[~dm.mask] = 0
            
            mm = np.zeros((n_superpixels,3), dtype=np.float)
            mm[list(c)] = colors[ci]
            blob = mm[segmentation]
            
            if ci == 0:
                vis = alpha_blending(blob, gray2rgb(dm.image), alpha, 1.*np.ones((dm.image_height, dm.image_width)))
            else:
                vis = alpha_blending(blob, vis[..., :-1], alpha, vis[..., -1])
                
    else:
    
        n_superpixels = segmentation.max() + 1

        n = len(clusters)
        m = -1*np.ones((n_superpixels,), dtype=np.int)

        for ci, c in enumerate(clusters):
            m[list(c)] = ci

        a = m[segmentation]
        a[~dm.mask] = -1
    #     a = -1*np.ones_like(segmentation)
    #     for ci, c in enumerate(clusters):
    #         for i in c:
    #             a[segmentation == i] = ci

        vis = label2rgb(a, image=segmentation_vis)

        vis = img_as_ubyte(vis[...,::-1])

    #     for ci, c in enumerate(clusters):
    #         for i, sp in enumerate(c):
    #             vis = cv2.putText(vis, str(i), 
    #                               tuple(np.floor(sp_properties[sp, [1,0]] - np.array([10,-10])).astype(np.int)), 
    #                               cv2.FONT_HERSHEY_DUPLEX,
    #                               1., ((0,255,255)), 1)
    
    return vis.copy()


# In[25]:

def find_boundaries(clusters, neighbors, neighbor_graph=None):
        
    n_superpixels = len(clusters)
    
    surrounds_sps = []
    frontiers_sps = []
    
    for cluster_ind, cluster in enumerate(clusters):
        
        surrounds = set([i for i in set.union(*[neighbors[c] for c in cluster]) if i not in cluster and i != -1])
#         surrounds = set([i for i in surrounds if any([(n not in cluster) and (n not in surrounds) for n in neighbors[i]])])
        surrounds = set([i for i in surrounds if any([n not in cluster for n in neighbors[i]])])

        if len(surrounds) == 0:
            continue

        frontiers = set.union(*[neighbors[c] for c in surrounds]) & set(cluster)

        if neighbor_graph is not None:
        
            surrounds_subgraph = neighbor_graph.subgraph(surrounds)
            surrounds_traversal = list(dfs_postorder_nodes(surrounds_subgraph))

            frontiers_subgraph = neighbor_graph.subgraph(frontiers)
            frontiers_traversal = list(dfs_postorder_nodes(frontiers_subgraph))

            surrounds_sps.append(surrounds_traversal)
            frontiers_sps.append(frontiers_traversal)
        
        else:
            surrounds_sps.append(list(surrounds))
            frontiers_sps.append(list(frontiers))
        
    return surrounds_sps, frontiers_sps


def chi2pval(O1, O2):
    n1 = O1.sum()
    n2 = O2.sum()
    n = n1 + n2
    nc = O1 + O2
    E1 = n1/n*nc
    E2 = n2/n*nc
    v = np.nonzero((O1 > 0) &  (O2 > 0) & (vars > 0))[0]
    if len(v) == 0:
        return 1e-6, np.nan
    dof = max(len(v)-1, 1)
    q = np.sum((O1[v]-E1[v])**2/vars[v]+(O2[v]-E2[v])**2/vars[v])
    return chisqprob(q, dof), q


# In[22]:

def compute_overlap(c1, c2):
#     return float(len(c1 & c2))/min(len(c1),len(c2))
    return float(len(c1 & c2))/len(c1 | c2)

def compute_overlap_partial(indices, sets):
    n_sets = len(sets)
    
    overlap_matrix = np.zeros((len(indices), n_sets))
        
    for ii, i in enumerate(indices):
        for j in range(n_sets):
            c1 = set(sets[i])
            c2 = set(sets[j])
            overlap_matrix[ii, j] = compute_overlap(c1, c2)
            
    return overlap_matrix

def set_pairwise_distances(sets):

    partial_overlap_mat = Parallel(n_jobs=16, max_nbytes=1e6)(delayed(compute_overlap_partial)(s, sets) 
                                        for s in np.array_split(range(len(sets)), 16))
    overlap_matrix = np.vstack(partial_overlap_mat)
    distance_matrix = 1 - overlap_matrix
    
    return distance_matrix


# In[23]:

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import average, fcluster, leaders, complete, single, dendrogram

def group_clusters(clusters=None, dist_thresh = 0.01, distance_matrix=None):

    if distance_matrix is None:
        assert clusters is not None
        distance_matrix = set_pairwise_distances(clusters)

    lk = complete(squareform(distance_matrix))
#     lk = average(squareform(distance_matrix))
#     lk = single(squareform(distance_matrix))

    # T = fcluster(lk, 1.15, criterion='inconsistent')
    T = fcluster(lk, dist_thresh, criterion='distance')

    n_groups = len(set(T))    
    groups = [None] * n_groups

    for group_id in range(n_groups):
        groups[group_id] = np.where(T == group_id)[0]
        
    return [g for g in groups if len(g) > 0]



# In[ ]:
import sys
sys.path.append('/home/yuncong/Brain/pipeline_scripts')
from utilities2014 import *
import os

from scipy.spatial.distance import cdist, pdist, squareform
from joblib import Parallel, delayed
from skimage.color import gray2rgb

from skimage.measure import find_contours
import cv2
from skimage.util import img_as_float

from networkx import from_dict_of_lists, Graph, adjacency_matrix, dfs_postorder_nodes
from networkx.algorithms import node_connected_component

from scipy.stats import chisquare, chisqprob

os.environ['GORDON_DATA_DIR'] = '/home/yuncong/project/DavidData2014tif/'
os.environ['GORDON_REPO_DIR'] = '/home/yuncong/Brain'
os.environ['GORDON_RESULT_DIR'] = '/home/yuncong/project/DavidData2014results/'
os.environ['GORDON_LABELING_DIR'] = '/home/yuncong/project/DavidData2014labelings/'

dm = DataManager(data_dir=os.environ['GORDON_DATA_DIR'], 
  repo_dir=os.environ['GORDON_REPO_DIR'], 
  result_dir=os.environ['GORDON_RESULT_DIR'], 
  labeling_dir=os.environ['GORDON_LABELING_DIR'])

dm.set_stack('RS141')
dm.set_resol('x5')
dm.set_gabor_params(gabor_params_id='blueNisslWide')
dm.set_segmentation_params(segm_params_id='blueNisslRegular')
dm.set_vq_params(vq_params_id='blueNissl')

for section_id in range(int(sys.argv[1]), int(sys.argv[2])):

    dm.set_slice(section_id)
    dm._load_image()

    texton_hists = dm.load_pipeline_result('texHist', 'npy')
    segmentation = dm.load_pipeline_result('segmentation', 'npy')
    n_superpixels = len(np.unique(segmentation)) - 1
    textonmap = dm.load_pipeline_result('texMap', 'npy')
    n_texton = len(np.unique(textonmap)) - 1
    neighbors = dm.load_pipeline_result('neighbors', 'npy')
    sp_properties = dm.load_pipeline_result('spProps', 'npy')
    segmentation_vis = dm.load_pipeline_result('segmentationWithoutText', 'jpg')
    texton_freqs = texton_hists * sp_properties[:,2][:, np.newaxis]
    vars = np.var(texton_freqs, axis=0)

    try:
        sp_sp_dists = dm.load_pipeline_result('texHistPairwiseDist', 'npy')
        raise
    except:
        def f(a):
            sp_dists = cdist(a, texton_hists, metric=chi2)
    #         sp_dists = cdist(a, texton_hists, metric=js)
            return sp_dists

        sp_dists = Parallel(n_jobs=16)(delayed(f)(s) for s in np.array_split(texton_hists, 16))
        sp_sp_dists = np.vstack(sp_dists)

        dm.save_pipeline_result(sp_sp_dists, 'texHistPairwiseDist', 'npy')

    center_dists = pdist(sp_properties[:, :2])
    center_dist_matrix = squareform(center_dists)

    neighbors_dict = dict(zip(np.arange(n_superpixels), [list(i) for i in neighbors]))
    neighbor_graph = from_dict_of_lists(neighbors_dict)


    try:
        expansion_clusters_tuples = dm.load_pipeline_result('clusters', 'pkl')
        raise
    except Exception as e:

        import time
        b = time.time()

        expansion_clusters_tuples = Parallel(n_jobs=16)(delayed(grow_cluster3)(s, neighbors, texton_hists) for s in range(n_superpixels))

        print time.time() - b

        dm.save_pipeline_result(expansion_clusters_tuples, 'clusters', 'pkl')

    # expansion_clusters_tuples = dm.load_pipeline_result('clusters', 'pkl')
    expansion_clusters, expansion_cluster_scores = zip(*expansion_clusters_tuples)
    expansion_cluster_scores = np.array(expansion_cluster_scores)


    try:
        D = dm.load_pipeline_result('clusterPairwiseDist', 'npy')
        raise
    except:
        D = set_pairwise_distances(expansion_clusters)
        dm.save_pipeline_result(D, 'clusterPairwiseDist', 'npy')

    try:
        expansion_cluster_groups = dm.load_pipeline_result('clusterGroups', 'pkl')
        raise
    except:
        import time
        t = time.time()
        expansion_cluster_groups = group_clusters(expansion_clusters, dist_thresh=.8, distance_matrix=D)
        dm.save_pipeline_result(expansion_cluster_groups, 'clusterGroups', 'pkl')

        print time.time() - t

    print len(expansion_cluster_groups), 'expansion cluster groups'
    expansion_cluster_group_sizes = np.array(map(len, expansion_cluster_groups))


    big_group_indices = np.where(expansion_cluster_group_sizes > 5)[0]
    n_big_groups = len(big_group_indices)
    print n_big_groups, 'big cluster groups'
    big_groups = [expansion_cluster_groups[i] for i in big_group_indices]

    from collections import Counter

    representative_clusters = []
    representative_cluster_scores = []
    representative_cluster_indices = []

    big_groups_valid = []

    for g in big_groups:
        for i in np.argsort(expansion_cluster_scores[g])[::-1]:
            c = expansion_clusters[g[i]]
            sc = expansion_cluster_scores[g[i]]
            if len(c) > n_superpixels * .004:
                representative_clusters.append(c)
                representative_cluster_indices.append(g[i])
                representative_cluster_scores.append(sc)
                big_groups_valid.append(g)
                break

    print len(representative_clusters), 'representative clusters'

    representative_cluster_scores_sorted, representative_clusters_sorted_by_score,     representative_cluster_indices_sorted_by_score,     big_groups_sorted_by_score = map(list, zip(*sorted(zip(representative_cluster_scores, 
                                                            representative_clusters,
                                                            representative_cluster_indices,
                                                            big_groups_valid), reverse=True)))


    final_clusters_sorted_by_score = representative_clusters_sorted_by_score[:50]
    final_cluster_scores_sorted = representative_cluster_scores_sorted[:50]
    final_cluster_indices_sorted_by_score = representative_cluster_indices_sorted_by_score[:50]


    ###################


    vis = visualize_multiple_clusters(final_clusters_sorted_by_score[:10], segmentation=segmentation, segmentation_vis=dm.image)
    dm.save_pipeline_result( vis, 'regionsTop10' , 'jpg')

    vis = visualize_multiple_clusters(final_clusters_sorted_by_score[:20], segmentation=segmentation, segmentation_vis=dm.image)
    dm.save_pipeline_result( vis, 'regionsTop20' , 'jpg')

    vis = visualize_multiple_clusters(final_clusters_sorted_by_score[:30], segmentation=segmentation, segmentation_vis=dm.image)
    dm.save_pipeline_result( vis, 'regionsTop30' , 'jpg')

    vis = visualize_multiple_clusters(final_clusters_sorted_by_score[:40], segmentation=segmentation, segmentation_vis=dm.image)
    dm.save_pipeline_result( vis, 'regionsTop40' , 'jpg')

    vis = visualize_multiple_clusters(final_clusters_sorted_by_score[10:20], segmentation=segmentation, segmentation_vis=dm.image)
    dm.save_pipeline_result( vis, 'regionsTop10to20' , 'jpg')

    vis = visualize_multiple_clusters(final_clusters_sorted_by_score[20:30], segmentation=segmentation, segmentation_vis=dm.image)
    dm.save_pipeline_result( vis, 'regionsTop20to30' , 'jpg')

    vis = visualize_multiple_clusters(final_clusters_sorted_by_score[30:40], segmentation=segmentation, segmentation_vis=dm.image)
    dm.save_pipeline_result( vis, 'regionsTop30to40' , 'jpg')

    vis = visualize_multiple_clusters(final_clusters_sorted_by_score[40:], segmentation=segmentation, segmentation_vis=dm.image)
    dm.save_pipeline_result( vis, 'regionsTop40toX' , 'jpg')


    #     fig = plt.figure(figsize=(10,10))
    #     plt.imshow(vis)
    #     plt.title('sorted group ' + str(i) + ', score ' + str(s))
    #     plt.axis('off')
    #     plt.show()


    # colors = (np.loadtxt('../visualization/100colors.txt') * 255).astype(np.int)

    # def visualize_contours(clusters):

    #     vis = dm.image_rgb.copy()

    #     for ci, c in enumerate(clusters):
    #         q = np.zeros((n_superpixels,))
    #         q[c] = 1.
    #         v = q[segmentation]
    #         contours = find_contours(img_as_float(v), 0.8)
    #         contour = contours[np.argmax(map(len, contours))]
    #         contour = np.round(contour[:,::-1].reshape((-1,1,2))).astype(np.int)
    #         cv2.polylines(vis, [contour], isClosed=True, color=colors[ci%len(colors)], thickness=10) 
    #     #     cv2.polylines(vis, [contour], isClosed=True, color=[237,194,136], thickness=5) 

    #     #     ax.plot(contour[:,1], contour[:,0])

    #     # fig.savefig('tmp.png', bbox_inches='tight', pad_inches=0)
    #     # fig.savefig('tmp.png', pad_inches=0)
    #     # SaveFigureAsImage('tmp.png', fig, orig_size=dm.image.shape[:2])
    #     # FileLink('tmp.png')
    #     return vis

    # # dm.save_pipeline_result(representative_clusters_sorted_by_score, 'goodRegions', 'pkl')

    # vis = visualize_contours(final_clusters_sorted_by_score[:10])
    # dm.save_pipeline_result(np.uint8(vis), 'contoursTop10', 'jpg')

    # vis = visualize_contours(final_clusters_sorted_by_score[10:20])
    # dm.save_pipeline_result(np.uint8(vis), 'contoursTop10to20', 'jpg')

    # vis = visualize_contours(final_clusters_sorted_by_score[20:30])
    # dm.save_pipeline_result(np.uint8(vis), 'contoursTop20to30', 'jpg')

    # vis = visualize_contours(final_clusters_sorted_by_score[30:])
    # dm.save_pipeline_result(np.uint8(vis), 'contoursTop30to40', 'jpg')

    # vis = visualize_contours(final_clusters_sorted_by_score)
    # dm.save_pipeline_result(np.uint8(vis), 'contoursTopAll', 'jpg')

    # vis = visualize_contours(final_clusters_sorted_by_score[:20])
    # dm.save_pipeline_result(np.uint8(vis), 'contoursTop20', 'jpg')

    # vis = visualize_contours(final_clusters_sorted_by_score[:30])
    # dm.save_pipeline_result(np.uint8(vis), 'contoursTop30', 'jpg')

    # vis = visualize_contours(final_clusters_sorted_by_score[:40])
    # dm.save_pipeline_result(np.uint8(vis), 'contoursTop40', 'jpg')

    representative_clusters = zip(representative_cluster_scores_sorted, representative_clusters_sorted_by_score, 
                       representative_cluster_indices_sorted_by_score, 
                       big_groups_sorted_by_score)

    dm.save_pipeline_result(representative_clusters, 'representativeClusters', 'pkl')

