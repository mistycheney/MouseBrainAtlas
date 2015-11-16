#! /usr/bin/env python

from flask import Flask, jsonify, request

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
	ALGORITHM = 'algorithm'
	
class PolygonType(Enum):
	CLOSED = 'closed'
	OPEN = 'open'
	TEXTURE = 'textured'
	TEXTURE_WITH_CONTOUR = 'texture with contour'
	DIRECTION = 'directionality'

label_examples = None
label_polygon = None
label_position = None
label_texture = None

dms = dict([(sc, DataManager(stack='MD594', section=sc, segm_params_id='tSLIC200', load_mask=False)) 
			for sc in range(47, 185)])

for dm in dms.itervalues():
	dm.load_multiple_results(['texHist', 'spCentroids', 'edgeMidpoints', 'edgeEndpoints'])

def grow_cluster_section(sec, *args, **kwargs):
    return dms[sec].grow_cluster(*args, **kwargs)

def grow_clusters_from_sps(sec, sps):
    
    expansion_clusters_tuples = Parallel(n_jobs=16)\
    (delayed(grow_cluster_section)(sec, s, verbose=False, all_history=False, seed_weight=0,\
                                   num_sp_percentage_limit=0.05, min_size=1, min_distance=2, threshold_abs=-0.1,
                                   threshold_rel=0.02, peakedness_limit=.002, method='rc-mean',
                                   seed_dist_lim = 0.2, inter_sp_dist_lim=1.) for s in sps)

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


def sort_clusters(sec, all_unique_tuples, label, sec2):
    
    dm = dms[sec]
    
    seeds, clusters, sigs = zip(*all_unique_tuples)
    
#     props = compute_proposal_properties(all_unique_tuples, sec)  
#     cluster_areas = map(attrgetter('area'), props)
#     cluster_centers = map(attrgetter('centroid'), props)
    
    tex_dists = chi2s(label_texture[label], [dm.texton_hists[sps].mean(axis=0) for sps in clusters])
    
#     remaining_cluster_indices_sortedByTexture = [remaining_cluster_indices[j] for j in np.argsort(tex_dists)]
      
    polygons = [Polygon(dm.vertices_from_dedges(dm.find_boundary_dedges_ordered(cl))) 
                for cl in clusters]

    polygon_overlaps = []
    for p in polygons:
        try:
#             polygon_overlaps.append(label_polygon[label][sec2].intersection(p).area)
            polygon_overlaps.append(float(label_polygon[label][sec2].intersection(p).area)/label_polygon[label][sec2].union(p).area)
        except:
#             print list(p.exterior.coords)
            polygon_overlaps.append(0)
    
#     rank = np.argsort(.3*scores_to_vote(polygon_overlaps) + .7*scores_to_vote(-tex_dists))[::-1]
    rank = np.argsort(.5*scores_to_vote(polygon_overlaps) + .5*scores_to_vote(-tex_dists))[::-1]

    all_remaining_clusters_sorted = [clusters[i] for i in rank]

#     remaining_cluster_indices_sortedByOverlap = [remaining_cluster_indices[j] for j in np.argsort(polygon_overlaps)[::-1]]
    
#     all_remaining_clusters_sortedByTexture = [all_unique_clusters[i] for i in remaining_cluster_indices_sortedByTexture]

#     all_remaining_clusters_sortedByOverlap = [all_unique_clusters[i] for i in remaining_cluster_indices_sortedByOverlap]
    
#     return all_remaining_clusters_sortedByTexture
#     return all_remaining_clusters_sortedByOverlap
    return all_remaining_clusters_sorted

@app.route('/')
def index():
	return "Brainstem"


@app.route('/update_db')
def update_db():

	label_examples = defaultdict(list) # cluster, dedges, sig, stack, sec, proposal_type, labeling username, timestamp

	for sec, dm in dms.iteritems():
		
		dm.reload_labelings()
		res = dm.load_proposal_review_result('yuncong', 'latest', 'consolidated')
		
		if res is None:
			continue
			
		usr, ts, suffix, result = res
		
		for props in result:
			if props['type'] == ProposalType.FREEFORM:
				pp = Path(props['vertices'])
				cl = np.where([pp.contains_point(s) for s in dm.sp_centroids[:, ::-1]])[0]
				de = dm.find_boundary_dedges_ordered(cl)
				sig = dm.compute_cluster_score(cl, method='rc-mean')[0]
				label_examples[props['label']].append((cl, de, sig, dm.stack, dm.slice_ind, ProposalType.FREEFORM, usr, ts))            
			else:
				label_examples[props['label']].append((props['sps'], props['dedges'], props['sig'], 
										  dm.stack, dm.slice_ind, props['type'], usr, ts))

	label_examples.default_factory = None

	label_texture = {}

	for label, proposals in label_examples.iteritems():
		w = []
		for prop in proposals:
			cluster, dedges, sig, stack, sec, proposal_type, username, timestamp = prop
			w.append(dms[sec].texton_hists[cluster].mean(axis=0))
			
		label_texture[label] = np.mean(w, axis=0)


	label_position = defaultdict(lambda: {})

	for label, proposals in label_examples.iteritems():

		for prop in proposals:
			cluster, dedges, sig, stack, sec, proposal_type, username, timestamp = prop
			
			dms[sec].load_multiple_results(['spCoords'])
			
			pts = np.vstack([ dms[sec].sp_coords[sp][:, ::-1] for sp in cluster])
			ell = fit_ellipse_to_points(pts)
			label_position[label][sec] = ell
			
	label_position.default_factory=None

	from shapely.geometry import Polygon

	label_polygon = defaultdict(lambda: {})

	for label, proposals in label_examples.iteritems():

		for prop in proposals:
			cluster, dedges, sig, stack, sec, proposal_type, username, timestamp = prop
			vs = vertices_from_dedges(sec, dedges)
			polygon = Polygon(vs)
			label_polygon[label][sec] = polygon
			
	#         pts = np.vstack([ dms[sec].sp_coords[sp][:, ::-1] for sp in cluster])
	#         label_coords[label][sec] = pts

	label_polygon.default_factory=None

	pickle.dump(label_examples, open(os.environ['GORDON_RESULT_DIR']+'/database/label_examples.pkl', 'w'))
	pickle.dump(label_position, open(os.environ['GORDON_RESULT_DIR']+'/database/label_position.pkl', 'w'))
	pickle.dump(label_polygon, open(os.environ['GORDON_RESULT_DIR']+'/database/label_polygon.pkl', 'w'))
	pickle.dump(label_texture, open(os.environ['GORDON_RESULT_DIR']+'/database/label_texture.pkl', 'w'))

	d = {'result': 0}
	return jsonify(**d)

# @app.route('/top_down_detect/<labels>')
@app.route('/top_down_detect')
def top_down_detect():

	global label_examples
	global label_position
	global label_polygon
	global label_texture

	label_examples = pickle.load(open(os.environ['GORDON_RESULT_DIR']+'/database/label_examples.pkl', 'r'))
	label_position = pickle.load(open(os.environ['GORDON_RESULT_DIR']+'/database/label_position.pkl', 'r'))
	label_polygon = pickle.load(open(os.environ['GORDON_RESULT_DIR']+'/database/label_polygon.pkl', 'r'))
	label_texture = pickle.load(open(os.environ['GORDON_RESULT_DIR']+'/database/label_texture.pkl', 'r'))

	labels = request.args.getlist('labels')
	section = request.args.get('section', type=int)

	print section, labels

	d = {}

	for label in labels:

		ks = np.array(label_position[label].keys())
		ds = ks - section

		next_labeled_section = ks[ds >= 0][0]
		prev_labeled_section = ks[ds <= 0][0]

		print 'prev_labeled_section', prev_labeled_section, 'next_labeled_section', next_labeled_section

		if abs(section - prev_labeled_section) < abs(section - next_labeled_section):
		    v1,v2,s1,s2,c0 = label_position[label][prev_labeled_section]
		else:
		    v1,v2,s1,s2,c0 = label_position[label][next_labeled_section]

		# v1,v2,s1,s2,c0 = label_position['pontine']

		angle = np.rad2deg(np.arctan2(v1[1], v1[0]))
		ell_vertices = cv2.ellipse2Poly(tuple(c0.astype(np.int)), (int(2*1.5*s1), int(2*1.5*s2)), int(angle), 0, 360, 10)

		sps = np.where([Path(ell_vertices).contains_point(s) for s in dms[section].sp_centroids[:,::-1]])[0]

		print '%d sps to look at\n' % len(sps)

		all_unique_tuples = grow_clusters_from_sps(section, sps)

		if abs(section - prev_labeled_section) < abs(section - next_labeled_section):
		    clusters_sorted = sort_clusters(section, all_unique_tuples, label, prev_labeled_section)
		else:
		    clusters_sorted = sort_clusters(section, all_unique_tuples, label, next_labeled_section)

		best_sps = clusters_sorted[0]
		best_dedges = dms[section].find_boundary_dedges_ordered(best_sps)
		best_sig = np.nan

		d[label] = (best_sps, best_dedges, best_sig)

	return jsonify(**d)	

if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0', use_reloader=False)
