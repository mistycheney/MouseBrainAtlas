#! /usr/bin/env python

import os
import argparse
import sys
import time

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Compute texton map')

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("slice_ind", type=int, help="slice index")
parser.add_argument("-t", "--texton_path", type=str, help="path to textons.npy", default='')
parser.add_argument("-g", "--gabor_params_id", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNisslWide')
parser.add_argument("-v", "--vq_params_id", type=str, help="vector quantization parameters id (default: %(default)s)", default='blueNissl')
args = parser.parse_args()

from joblib import Parallel, delayed

sys.path.append(os.path.join(os.environ['GORDON_REPO_DIR'], 'utilities'))
from utilities2015 import *

from scipy.spatial.distance import cdist

dm = DataManager(gabor_params_id=args.gabor_params_id, 
                 stack=args.stack_name, 
                 section=args.slice_ind,
                 result_dir='/scratch/yuncong/CSHL_data_results')

#==================================================

# if dm.check_pipeline_result('texMap') and dm.check_pipeline_result('texMapViz'):
# # if False:
#     print "texMap.npy already exists, skip"

#     textonmap = dm.load_pipeline_result('texMap')
#     n_texton = textonmap.max() + 1

# else:

	# print 'loading centroids ...',
	# t = time.time()

if args.texton_path == '':
    centroids = dm.load_pipeline_result('textons')
else:
    centroids = np.load(args.texton_path)

n_texton = len(centroids)

# features_rotated = np.c_[np.r_[dm.load_pipeline_result('featuresRotated0'), dm.load_pipeline_result('featuresRotated1')],
#                         np.r_[dm.load_pipeline_result('featuresRotated2'), dm.load_pipeline_result('featuresRotated3')]]

# features_rotated = dm.load_pipeline_result('featuresRotated')
# print 'done in', time.time() - t, 'seconds'

# print 'assign textons ...',
# t = time.time()

textonmap = -1 * np.ones((dm.image_height, dm.image_width), dtype=np.int8)

block_size = 7000

for col, xmin in enumerate(range(dm.xmin, dm.xmax, block_size)):
    for row, ymin in enumerate(range(dm.ymin, dm.ymax, block_size)):

		xmax = xmin + block_size - 1
		ymax = ymin + block_size - 1

		t = time.time()
		sys.stderr.write('load featuresRotated ...')

		if not dm.check_pipeline_result('featuresMaskedRotatedRow%dCol%d'%(row, col)):
			continue
		
		features_rotated = dm.load_pipeline_result('featuresMaskedRotatedRow%dCol%d'%(row, col))

		sys.stderr.write('done in %f seconds\n' % (time.time() - t))

		t = time.time()
		sys.stderr.write('assign labels ...')

		n = len(features_rotated)

		labels = np.empty(n, np.int8)

		# label_list = []
		c = 0
		for i in range(0, n, 5000000):
		    Ds = Parallel(n_jobs=16)(delayed(cdist)(fs, centroids) for fs in np.array_split(features_rotated[i:i+5000000], 16))
		    for D in Ds:
				labels[c:c+D.shape[0]] = np.argmin(D, axis=1)
				c = c + D.shape[0]

			# labels[i:i+5000000] = np.concatenate([np.argmin(D, axis=1) for D in Ds])

		    # for D in Ds:
		    #     label_list.append(np.argmin(D, axis=1))

		sys.stderr.write('done in %f seconds\n' % (time.time() - t))


	# labels = np.concatenate(label_list)

		mask = dm.mask[ymin:ymax+1, xmin:xmax+1]
		textonmap[ymin:ymax+1, xmin:xmax+1][mask] = labels
	
	# sys.stderr.write('done in %f seconds\n' % (time.time() - t))

# print 'done in', time.time() - t, 'seconds'

t = time.time()
sys.stderr.write('dumping texmap ...')

dm.save_pipeline_result(textonmap, 'texMap')

sys.stderr.write('done in %f seconds\n' % (time.time() - t))


t = time.time()
sys.stderr.write('dumping texmap visualization ...')

colors = (np.loadtxt(dm.repo_dir + '/visualization/100colors.txt') * 255).astype(np.uint8)

textonmap_viz = np.zeros((dm.image_height, dm.image_width, 3), np.uint8)
textonmap_viz[dm.mask] = colors[textonmap[dm.mask]]
dm.save_pipeline_result(textonmap_viz, 'texMapViz')

sys.stderr.write('done in %f seconds\n' % (time.time() - t))
