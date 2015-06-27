
import os
import argparse
import sys
from joblib import Parallel, delayed

sys.path.append(os.path.join(os.environ['GORDON_REPO_DIR'], 'pipeline_scripts'))

if os.environ['DATASET_VERSION'] == '2014':
	from utilities2014 import *
elif os.environ['DATASET_VERSION'] == '2015':
	from utilities import *

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

dm = DataManager(data_dir=os.environ['GORDON_DATA_DIR'], 
  repo_dir=os.environ['GORDON_REPO_DIR'], 
  result_dir=os.environ['GORDON_RESULT_DIR'], labeling_dir=os.environ['GORDON_LABELING_DIR'])

dm.set_gabor_params(gabor_params_id='blueNisslWide')
dm.set_segmentation_params(segm_params_id='blueNisslRegular')
dm.set_vq_params(vq_params_id='blueNissl')

dm.set_image(args.stack_name, 'x5', args.slice_ind)

#============================================================

if dm.check_pipeline_result('features_rotated', 'npy'):

	print "features_rotated.npy already exists, skip"
else:

	features = dm.load_pipeline_result('features', 'npy')

	valid_features = features[:, dm.mask].T
	n_valid = len(valid_features)

	del features

	def rotate_features(fs):
	    features_tabular = fs.reshape((fs.shape[0], dm.n_freq, dm.n_angle))
	    max_angle_indices = features_tabular.max(axis=1).argmax(axis=-1)
	    features_rotated = np.reshape([np.roll(features_tabular[i], -ai, axis=-1) 
	                               for i, ai in enumerate(max_angle_indices)], (fs.shape[0], dm.n_freq * dm.n_angle))

	    return features_rotated

	from joblib import Parallel, delayed

	n_splits = 1000
	features_rotated_list = Parallel(n_jobs=16)(delayed(rotate_features)(fs) for fs in np.array_split(valid_features, n_splits))
	features_rotated = np.vstack(features_rotated_list)

	del valid_features

	dm.save_pipeline_result(features_rotated, 'features_rotated', 'npy')

