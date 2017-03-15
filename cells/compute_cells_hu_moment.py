#! /usr/bin/env python

import os
import sys
import time

from multiprocess import Pool
from skimage.measure import moments_central, moments_hu, moments_normalized, moments

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from data_manager import *
from metadata import *
from cell_utilities import *

cell_size_thresh_um2 = 30
cell_size_thresh = cell_size_thresh_um2 / XY_PIXEL_DISTANCE_LOSSLESS**2
print 'Cell size threshold = %.2f' % cell_size_thresh

import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='')
parser.add_argument("stack", type=str, help="stack name")
args = parser.parse_args()

stack = args.stack

first_section, last_section = metadata_cache['section_limits'][stack]

for sec in range(first_section, last_section+1):

    if is_invalid(stack=stack, sec=sec):
        continue

    cells_aligned_padded = load_cell_data(what='cells_aligned_mirrored_padded', stack=stack, sec=sec, ext='bp')

    n_cells = cells_aligned_padded.shape[0]
    print 'Extract %d cells from section %d.' % (n_cells, sec)

    cell_sizes = np.reshape(cells_aligned_padded, (n_cells, -1)).sum(axis=1)

    cell_sizes_fp = get_cell_data_filepath('cellSizes', stack=stack, sec=sec, ext='bp')
    bp.pack_ndarray_file(cell_sizes, cell_sizes_fp)

    large_cell_indices = np.where(cell_sizes > cell_size_thresh)[0]
    print 'Got %s large cells.' % len(large_cell_indices)

    large_cell_indices_fp = get_cell_data_filepath('largeCellIndices', stack=stack, sec=sec, ext='bp')
    bp.pack_ndarray_file(large_cell_indices, large_cell_indices_fp)

    def compute_hu_moments(i):
        b = cells_aligned_padded[i].astype(np.uint8)
        m = moments(b, order=1)
        hu = moments_hu(moments_normalized(moments_central(b, cc=m[0,1]/m[0,0], cr=m[1,0]/m[0,0])))
        return hu

    t = time.time()

# Using parallel sometimes causes stall. Not much faster than sequential anyway.
#     pool = Pool(8)
#     large_cell_hu_moments = np.array(pool.map(compute_hu_moments, large_cell_indices))
#     pool.close()
#     pool.join()

    large_cell_hu_moments = np.array([compute_hu_moments(i) for i in large_cell_indices])

    sys.stderr.write('Compute hu moments: %.2f seconds.\n' % (time.time()-t)) # 3-7s

    cell_orientations = load_cell_data(what='blobOrientations', stack=stack, sec=sec, ext='bp')
    cell_mirrors = load_cell_data(what='cells_aligned_mirrorDirections', stack=stack, sec=sec, ext='bp')

    large_cell_features = np.c_[cell_orientations[large_cell_indices],
                                     cell_mirrors[large_cell_indices],
                                     cell_sizes[large_cell_indices],
                                     large_cell_hu_moments]

    large_cell_features_fp = get_cell_data_filepath('largeCellFeatures', stack=stack, sec=sec, ext='bp')
    bp.pack_ndarray_file(large_cell_features, large_cell_features_fp)
