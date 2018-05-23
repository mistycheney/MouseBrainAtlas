#! /usr/bin/env python

import sys
import os
import time

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg') # https://stackoverflow.com/a/3054314
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from metadata import *
from data_manager import *

import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='')

parser.add_argument("brain_name", type=str, help="Brain name")
# parser.add_argument("--out_dir", type=str, help="Output directory")
args = parser.parse_args()

stack = args.brain_name

# for section in set(metadata_cache['valid_sections_all'][stack]) - set(metadata_cache['valid_sections'][stack]):
for section in metadata_cache['valid_sections'][stack]:
    
    print "Section", section
    
    t = time.time()
    
    img = DataManager.load_image_v2(stack=stack, prep_id=None, section=section, version='Ntb', resol='raw')

    sys.stderr.write('Load image: %.2f seconds.\n' % (time.time() - t))

    t = time.time()
    tb_mask = DataManager.load_thumbnail_mask_v3(stack=stack, prep_id=None, section=section)
    raw_mask = resize(tb_mask, img.shape) > .5
    
    save_data(raw_mask, 
          DataManager.get_image_filepath_v2(stack=stack, prep_id=None, section=section, version='mask', resol='raw', ext='bp'), 
          upload_s3=False)
    
    sys.stderr.write('Rescale mask: %.2f seconds.\n' % (time.time() - t))

    t = time.time()
    
    mean_std_all_regions = []
    cx_cy_all_regions = []
    region_size = 5000
    region_spacing = 3000

    for cx in range(0, img.shape[1], region_spacing):
        for cy in range(0, img.shape[0], region_spacing):
            region = img[max(cy-region_size/2, 0):min(cy+region_size/2+1, img.shape[0]-1), 
                         max(cx-region_size/2, 0):min(cx+region_size/2+1, img.shape[1]-1)]
            region_mask = raw_mask[max(cy-region_size/2, 0):min(cy+region_size/2+1, img.shape[0]-1), 
                                   max(cx-region_size/2, 0):min(cx+region_size/2+1, img.shape[1]-1)]
            if np.count_nonzero(region_mask) == 0:
                continue
            mean_std_all_regions.append((region[region_mask].mean(), region[region_mask].std()))
            cx_cy_all_regions.append((cx, cy))
            
    sys.stderr.write('Compute mean/std for sample regions: %.2f seconds.\n' % (time.time() - t))
    
    t = time.time()
    mean_map = resample_scoremap(sparse_scores=np.array(mean_std_all_regions)[:,0], 
                             sample_locations=cx_cy_all_regions,
                             gridspec=(region_size, region_spacing, img.shape[1], img.shape[0], (0,0)),
                            downscale=4, 
                                 interpolation_order=2)

    sys.stderr.write('Interpolate mean map: %.2f seconds.\n' % (time.time() - t)) #10s

    t = time.time()
    mean_map = rescale_by_resampling(mean_map, new_shape=(img.shape[1], img.shape[0]))
    sys.stderr.write('Scale up mean map: %.2f seconds.\n' % (time.time() - t)) #30s

    t = time.time()
    std_map = resample_scoremap(sparse_scores=np.array(mean_std_all_regions)[:,1], 
                             sample_locations=cx_cy_all_regions,
                             gridspec=(region_size, region_spacing, img.shape[1], img.shape[0], (0,0)),
                            downscale=4,
                               interpolation_order=2)
    sys.stderr.write('Interpolate std map: %.2f seconds.\n' % (time.time() - t)) #10s

    t = time.time()
    std_map = rescale_by_resampling(std_map, new_shape=(img.shape[1], img.shape[0]))
    sys.stderr.write('Scale up std map: %.2f seconds.\n' % (time.time() - t)) #30s
    
    # Save mean/std results.
    
    fp = DataManager.get_intensity_normalization_result_filepath(what='region_centers', stack=stack, section=section)
    create_parent_dir_if_not_exists(fp)    
    np.savetxt(fp, cx_cy_all_regions)
    
    fp = DataManager.get_intensity_normalization_result_filepath(what='mean_std_all_regions', stack=stack, section=section)
    create_parent_dir_if_not_exists(fp)
    np.savetxt(fp, mean_std_all_regions)
    
    fp = DataManager.get_intensity_normalization_result_filepath(what='mean_map', stack=stack, section=section)
    create_parent_dir_if_not_exists(fp)
    bp.pack_ndarray_file(mean_map.astype(np.float16), fp)
    
    fp = DataManager.get_intensity_normalization_result_filepath(what='std_map', stack=stack, section=section)
    create_parent_dir_if_not_exists(fp)
    bp.pack_ndarray_file(std_map.astype(np.float16), fp)

    # Export normalized image.
    
    t = time.time()
    raw_mask = raw_mask & (std_map > 0)
    img_normalized = np.zeros(img.shape, np.float32)
    img_normalized[raw_mask] = (img[raw_mask] - mean_map[raw_mask]) / std_map[raw_mask]
    sys.stderr.write('Normalize: %.2f seconds.\n' % (time.time() - t)) #30s

    t = time.time()
    save_data(img_normalized.astype(np.float16), 
              DataManager.get_intensity_normalization_result_filepath(what='normalized_float_map', stack=stack, section=section),
             upload_s3=False)
    sys.stderr.write('Save float version: %.2f seconds.\n' % (time.time() - t)) #30s
           
    # Export histogram.
    
    plt.hist(img_normalized[raw_mask].flatten(), bins=100, log=True);
    fp = DataManager.get_intensity_normalization_result_filepath(what='float_histogram_png', stack=stack, section=section)
    create_parent_dir_if_not_exists(fp)
    plt.savefig(fp)
    plt.close();
    
    # Compute normalized float-point value percentiles.
    
#     raw_mask = load_data(DataManager.get_image_filepath_v2(stack=stack, prep_id=None, section=section, version='mask', resol='raw', ext='bp'),
#                         download_s3=False)
    
#     img_normalized = load_data(
#               DataManager.get_intensity_normalization_result_filepath(what='normalized_float_map', stack=stack, section=section),
#              download_s3=False)
        
#     q = img_normalized[raw_mask]
    
#     save_data(np.percentile(q, range(101)), 
#               DataManager.get_intensity_normalization_result_filepath(what='float_percentiles', stack=stack, section=section),
#             upload_s3=False)
    
    # Clip normalized float-point values.
    
    # img_normalized = load_data(DataManager.get_intensity_normalization_result_filepath(what='normalized_float_map', stack=stack, section=section), download_s3=False)
        
    t = time.time()
    img_normalized_uint8 = rescale_intensity_v2(img_normalized, -2., 50.)
    sys.stderr.write('Rescale to uint8: %.2f seconds.\n' % (time.time() - t))

    t = time.time()
    raw_mask = load_data(DataManager.get_image_filepath_v2(stack=stack, prep_id=None, section=section, version='mask', resol='raw', ext='bp'),
                        download_s3=False)
    img_normalized_uint8[~raw_mask] = 0
    sys.stderr.write('Load mask: %.2f seconds.\n' % (time.time() - t))
    
    t = time.time()
    save_data(img_normalized_uint8, DataManager.get_image_filepath_v2(stack=stack, prep_id=None, section=section, version='NtbNormalizedAdaptive', resol='raw'),
             upload_s3=False)
    sys.stderr.write('Save uint8 version: %.2f seconds.\n' % (time.time() - t))
    
    # Invert and Gamma Correction.
    
    gamma_map = img_as_ubyte(adjust_gamma(np.arange(0, 256, 1) / 255., 8.))
        
    # img_normalized_uint8 = \
    # DataManager.load_image_v2(stack=stack, prep_id=None, section=section, version='NtbNormalizedAdaptive', resol='raw')
    
    img = 255 - img_normalized_uint8
    
    save_data(gamma_map[img], 
              DataManager.get_image_filepath_v2(stack=stack, prep_id=None, section=section, version='NtbNormalizedAdaptiveInvertedGamma', resol='raw'),
             upload_s3=False)
    
    
    