#! /usr/bin/env python

from flask import Flask, jsonify

app = Flask(__name__)

import os
import argparse
import sys
import time
import cv2

from joblib import Parallel, delayed
from collections import defaultdict
from matplotlib.path import Path
from shapely.geometry import Polygon

# parser = argparse.ArgumentParser(
#     formatter_class=argparse.RawDescriptionHelpFormatter,
#     description='Top down detection of specified landmarks')

# parser.add_argument("stack_name", type=str, help="stack name")
# parser.add_argument("slice_ind", type=int, help="slice index")
# parser.add_argument("-l", "--labels", type=str, help="labels", nargs='+', default=[])
# parser.add_argument("-g", "--gabor_params_id", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNisslWide')
# parser.add_argument("-s", "--segm_params_id", type=str, help="segmentation parameters id (default: %(default)s)", default='blueNisslRegular')
# parser.add_argument("-v", "--vq_params_id", type=str, help="vector quantization parameters id (default: %(default)s)", default='blueNissl')
# args = parser.parse_args()

# print args.labels

sys.path.append(os.path.join(os.environ['GORDON_REPO_DIR'], 'utilities'))
from utilities2015 import *

from enum import Enum

class ProposalType(Enum):
    GLOBAL = 'global'
    LOCAL = 'local'
    FREEFORM = 'freeform'
    
class PolygonType(Enum):
    CLOSED = 'closed'
    OPEN = 'open'
    TEXTURE = 'textured'
    TEXTURE_WITH_CONTOUR = 'texture with contour'
    DIRECTION = 'directionality'

dms = dict([(sc, DataManager(stack='MD593', section=sc, segm_params_id='tSLIC200', load_mask=False)) 
            for sc in range(60, 151)])

for dm in dms.itervalues():
    dm.load_multiple_results(['texHist', 'spCentroids', 'edgeMidpoints', 'edgeEndpoints'])

label_examples = pickle.load(open(os.environ['GORDON_RESULT_DIR']+'/database/label_examples.pkl', 'r'))
label_position = pickle.load(open(os.environ['GORDON_RESULT_DIR']+'/database/label_position.pkl', 'r'))
label_polygon = pickle.load(open(os.environ['GORDON_RESULT_DIR']+'/database/label_polygon.pkl', 'r'))
label_texture = pickle.load(open(os.environ['GORDON_RESULT_DIR']+'/database/label_texture.pkl', 'r'))

def grow_cluster_section(sec, *args, **kwargs):
    return dms[sec].grow_cluster(*args, **kwargs)

def grow_clusters_from_sps(sec, sps):
    
    expansion_clusters_tuples = Parallel(n_jobs=16)(delayed(grow_cluster_section)(sec, s, verbose=False, all_history=False, 
                                                                         seed_weight=0,
                                                                        num_sp_percentage_limit=0.05,
                                                                     min_size=1, min_distance=2,
                                                                        threshold_abs=-0.1,
                                                                        threshold_rel=0.02,
                                                                       peakedness_limit=.002,
                                                                       method='rc-mean')
                                    for s in sps)

    all_seed_cluster_score_tuples = [(seed, cl, sig) for seed, peaks in enumerate(expansion_clusters_tuples) 
                                     for cl, sig in zip(*peaks)]
    all_seeds, all_clusters, all_scores = zip(*all_seed_cluster_score_tuples)

    all_clusters_unique_dict = {}
    for i, cl in enumerate(all_clusters):
        all_clusters_unique_dict[frozenset(cl)] = i

    all_unique_cluster_indices = all_clusters_unique_dict.values()
    all_unique_cluster_scores = [all_scores[i] for i in all_unique_cluster_indices]
    all_unique_cluster_indices_sorted = [all_unique_cluster_indices[i] for i in np.argsort(all_unique_cluster_scores)[::-1]]

    all_unique_tuples = [all_seed_cluster_score_tuples[i] for i in all_unique_cluster_indices_sorted]

    return all_unique_tuples

def compute_cluster_coherence_score(sec, cluster, verbose=False):
    
    if len(cluster) > 1:
        cluster_avg = dms[sec].texton_hists[cluster].mean(axis=0)
        ds = np.squeeze(chi2s([cluster_avg], dms[sec].texton_hists[list(cluster)]))
        var = ds.mean()
    else:
        var = 0
    
    return var


coherence_limit = .25
area_limit = 60000
nonoverlapping_area_limit = 2.
bg_texton = 3
bg_texton_percentage = .2
significance_limit = 0.05
consensus_limit = -20

def scores_to_vote(scores):
    vals = np.unique(scores)
    d = dict(zip(vals, np.linspace(0, 1, len(vals))))
    votes = np.array([d[s] for s in scores])
    votes = votes/votes.sum()
    return votes


def filter_clusters(sec, all_unique_tuples, label, sec2):
    
    dm = dms[sec]
    
    all_unique_seeds, all_unique_clusters, all_unique_cluster_scores = zip(*all_unique_tuples)
    
    all_cluster_sigs = np.array(all_unique_cluster_scores)
    all_cluster_coherences = np.array([compute_cluster_coherence_score(sec, cl) for cl in all_unique_clusters])
    all_cluster_hists = [dm.texton_hists[cl].mean(axis=0) for cl in all_unique_clusters]
    all_cluster_entropy = np.nan_to_num([-np.sum(hist[hist!=0]*np.log(hist[hist!=0])) for hist in all_cluster_hists])

    all_cluster_centroids = np.array([dm.sp_centroids[cl, ::-1].mean(axis=0) for cl in all_unique_clusters])

    dm.load_multiple_results(['spAreas'])
    all_cluster_area = np.array([dm.sp_areas[cl].sum() for cl in all_unique_clusters])
    
    remaining_cluster_indices = [i for i, (cl, coh, sig, ent, cent, area, hist) in enumerate(zip(all_unique_clusters, 
                                                                                      all_cluster_coherences, 
                                                                                      all_cluster_sigs,
                                                                                      all_cluster_entropy,
                                                                                      all_cluster_centroids,
                                                                                      all_cluster_area,
                                                                                    all_cluster_hists)) 
            if coh < coherence_limit and sig > significance_limit and \
                area > area_limit and \
             ((ent > 1.5 and hist[bg_texton] < bg_texton_percentage) or \
              (cent[0] - dm.xmin > 800 and \
               dm.xmax - cent[0] > 800 and \
               cent[1] - dm.ymin > 800 and \
               dm.ymax - cent[1] > 800)
             )]
    
    print '%d unique clusters, %d remaining clusters' % (len(all_unique_clusters), len(remaining_cluster_indices))
    
    all_remaining_clusters = [all_unique_clusters[i] for i in remaining_cluster_indices]

    tex_dists = cdist([label_texture[label]], [all_cluster_hists[i] for i in remaining_cluster_indices], chi2)[0]
    
#     remaining_cluster_indices_sortedByTexture = [remaining_cluster_indices[j] for j in np.argsort(tex_dists)]
      
    polygons = [Polygon(vertices_from_dedges(sec, dm.find_boundary_dedges_ordered(cl))) for cl in all_remaining_clusters]

    polygon_overlaps = []
    for p in polygons:
        try:
            polygon_overlaps.append(label_polygon[label][sec2].intersection(p).area)
        except:
            polygon_overlaps.append(0)
    
#     rank = np.argsort(.3*scores_to_vote(polygon_overlaps) + .7*scores_to_vote(-tex_dists))[::-1]
    rank = np.argsort(.1*scores_to_vote(polygon_overlaps) + .9*scores_to_vote(-tex_dists))[::-1]

    all_remaining_clusters_sorted = [all_unique_clusters[i] for i in rank]

#     remaining_cluster_indices_sortedByOverlap = [remaining_cluster_indices[j] for j in np.argsort(polygon_overlaps)[::-1]]
    
#     all_remaining_clusters_sortedByTexture = [all_unique_clusters[i] for i in remaining_cluster_indices_sortedByTexture]

#     all_remaining_clusters_sortedByOverlap = [all_unique_clusters[i] for i in remaining_cluster_indices_sortedByOverlap]
    
#     return all_remaining_clusters_sortedByTexture
#     return all_remaining_clusters_sortedByOverlap
    return all_remaining_clusters_sorted


def vertices_from_dedges(sec, dedges):

    vertices = []
    for de_ind, de in enumerate(dedges):
        midpt = dms[sec].edge_midpoints[frozenset(de)]
        endpts = dms[sec].edge_endpoints[frozenset(de)]
        endpts_next_dedge = dms[sec].edge_endpoints[frozenset(dedges[(de_ind+1)%len(dedges)])]

        dij = cdist([endpts[0], endpts[-1]], [endpts_next_dedge[0], endpts_next_dedge[-1]])
        i,j = np.unravel_index(np.argmin(dij), (2,2))
        if i == 0:
            vertices += [endpts[-1], midpt, endpts[0]]
        else:
            vertices += [endpts[0], midpt, endpts[-1]]
        
    return vertices

@app.route('/')
def index():
    return "Brainstem"

from flask import request

# @app.route('/top_down_detect/<labels>')
@app.route('/top_down_detect')
def top_down_detect():

	labels = request.args.getlist('labels')
	section = request.args.get('section', type=int)

	print section, labels

	d = {}

	for label in labels:

		ks = np.array(label_position[label].keys())

		closest_labeled_section = ks[np.argmin(np.abs(ks-section))]
		print 'closest_labeled_section %d' % closest_labeled_section

		v1,v2,s1,s2,c0 = label_position[label][closest_labeled_section]

		# v1,v2,s1,s2,c0 = label_position['pontine']

		angle = np.rad2deg(np.arctan2(v1[1], v1[0]))
		ell_vertices = cv2.ellipse2Poly(tuple(c0.astype(np.int)), (int(2*1.5*s1), int(2*1.5*s2)), int(angle), 0, 360, 10)

		sps = np.where([Path(ell_vertices).contains_point(s) for s in dms[section].sp_centroids[:,::-1]])[0]

		print '%d sps to look at\n' % len(sps)

		all_unique_tuples = grow_clusters_from_sps(section, sps)
		all_remaining_clusters_sorted = filter_clusters(section, all_unique_tuples, label, closest_labeled_section)

		best_sps = all_remaining_clusters_sorted[0]
		best_dedges = dms[section].find_boundary_dedges_ordered(best_sps)
		best_sig = np.nan

		d[label] = (best_sps, best_dedges, best_sig)

	return jsonify(**d)	

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', use_reloader=False)
