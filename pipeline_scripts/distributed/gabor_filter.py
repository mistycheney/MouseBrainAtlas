
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

if dm.check_pipeline_result('features', 'npy'):
	print "features.npy already exists, skip"

else:

	from skimage.util import pad

	approx_bg_intensity = dm.image[10:20, 10:20].mean()

	masked_image = dm.image.copy()
	masked_image[~dm.mask] = approx_bg_intensity

	padded_image = pad(masked_image, dm.max_kern_size, 'linear_ramp', end_values=approx_bg_intensity)

	from joblib import Parallel, delayed
	from scipy.signal import fftconvolve

	def convolve_per_proc(i):
	    return fftconvolve(padded_image, dm.kernels[i], 'same').astype(np.half)

	padded_filtered = Parallel(n_jobs=16)(delayed(convolve_per_proc)(i) 
	                        for i in range(dm.n_kernel))

	filtered = [f[dm.max_kern_size:-dm.max_kern_size, dm.max_kern_size:-dm.max_kern_size] for f in padded_filtered]

	features = np.empty((dm.n_kernel, dm.image_height, dm.image_width), dtype=np.half)
	for i in range(dm.n_kernel):
	    features[i, ...] = filtered[i]

	del filtered

	dm.save_pipeline_result(features, 'features', 'npy')