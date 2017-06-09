#! /usr/bin/env python

import os, sys

from skimage.io import imread
import numpy as np

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from data_manager import *
from metadata import *
from multiprocess import Pool

################################################

import argparse
import json

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Match intensity profile of neurotrace images to adjacent Nissl.')
parser.add_argument("stack", type=str, help="Stack name")
parser.add_argument("filename_pairs", type=str, help="Filename pairs")
parser.add_argument("num_regions", type=int, help="Number of regions to compute mapping", default=8)
args = parser.parse_args()

stack = args.stack
filename_pairs = json.loads(args.filename_pairs)
n_regions = int(args.num_regions)

#############################################################################

def match_histogram(source, template):
    """
    Returns: 
        s_values (array): unique source values
        interp_t_values (array): unique destination values
    """
    
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return s_values, interp_t_values
    
for nissl_fn, ntb_fn in filename_pairs:
    
    ntb_matched_values_all_examples_one_section = []
    region_bboxes_all_examples_one_section = []
    
    nissl_tb_mask = DataManager.load_thumbnail_mask_v3(stack=stack, fn=nissl_fn, version='cropped')
    
    t = time.time()
    ntb_im = DataManager.load_image(stack=stack, fn=ntb_fn, version='cropped_16bit', resol='lossless')
    sys.stderr.write('Load NTB: %.2f seconds.\n' % (time.time()-t))

    t = time.time()
    nissl_im = DataManager.load_image(stack=stack, fn=nissl_fn, version='cropped', resol='lossless')
    sys.stderr.write('Load Nissl: %.2f seconds.\n' % (time.time()-t))
    
    h, w = nissl_im.shape[:2]
    
    # Must sample regions before entering multiprocesses, 
    # otherwise the sampled results across processes will be the same.
    regions = []
    for _ in range(n_regions):
        while True:
            region1_w = 5000
            region1_h = 5000
            region1_x = np.random.randint(0, w - region1_w, 1)[0]
            region1_y = np.random.randint(0, h - region1_h, 1)[0]
#             print region1_x, region1_y, region1_w, region1_h
            
            tb_region1_xmin = region1_x / 32
            tb_region1_xmax = (region1_x + region1_w) / 32
            tb_region1_ymin = region1_y / 32
            tb_region1_ymax = (region1_y + region1_h) / 32
            
            if np.all(np.r_[nissl_tb_mask[tb_region1_ymin, tb_region1_xmin],
            nissl_tb_mask[tb_region1_ymin, tb_region1_xmax],
            nissl_tb_mask[tb_region1_ymax, tb_region1_xmin],
            nissl_tb_mask[tb_region1_ymax, tb_region1_xmax]]):
                break
        regions.append((region1_x, region1_y, region1_w, region1_h))
    
    def match_intensity_histogram_one_region(region):
        
        region1_x, region1_y, region1_w, region1_h = region
        
        ntb_blue_region1 = ntb_im[region1_y:region1_y+region1_h, region1_x:region1_x+region1_w, 2]
    
        nissl_region1 = img_as_ubyte(rgb2gray(nissl_im[region1_y:region1_y+region1_h, region1_x:region1_x+region1_w]))

        ###############
        
        t = time.time()

        ntb_blue_region1_inv = 5000 - ntb_blue_region1.astype(np.int)
        ntb_inv_vals, nissl_vals = match_histogram(ntb_blue_region1_inv, nissl_region1)
    
        ntb_blue_bins = np.arange(5001)
    
        ntb_blue_inv_bins = np.arange(5001)
        ntb_inv_to_nissl_mapping = np.interp(ntb_blue_inv_bins, ntb_inv_vals, nissl_vals)
        
        ntb_to_nissl_mapping = ntb_inv_to_nissl_mapping[5000 - ntb_blue_bins]
        ntb_to_nissl_mapping = np.round(ntb_to_nissl_mapping).astype(np.uint8)
                        
        ntb_matched_values_all_examples_one_section.append(ntb_to_nissl_mapping)
        region_bboxes_all_examples_one_section.append((region1_x, region1_y, region1_w, region1_h))
    
        sys.stderr.write('Compute matching: %.2f seconds.\n' % (time.time()-t))
        
        return ntb_to_nissl_mapping, (region1_x, region1_y, region1_w, region1_h)
            
    pool = Pool(4)
    res = pool.map(match_intensity_histogram_one_region, regions)
    ntb_matched_values_all_examples_one_section, region_bboxes_all_examples_one_section = zip(*res)
    pool.close()
    pool.join()
    

    fp = os.path.join(DATA_DIR, stack, stack + '_intensity_mapping', '%s_to_%s_intensity_mapping_all_regions.npy' % (ntb_fn, nissl_fn))
    create_parent_dir_if_not_exists(fp)
    np.save(fp, np.asarray(ntb_matched_values_all_examples_one_section))
    upload_to_s3(fp)

    fp = os.path.join(DATA_DIR, stack, stack + '_intensity_mapping', '%s_to_%s_region_bboxes.npy' % (ntb_fn, nissl_fn))
    np.save(fp, np.asarray(region_bboxes_all_examples_one_section))
    upload_to_s3(fp)

    median_mapping_one_section = np.median(ntb_matched_values_all_examples_one_section, axis=0)
    fp = DataManager.get_ntb_to_nissl_intensity_profile_mapping_filepath(stack=stack, ntb_fn=ntb_fn)
    np.save(fp, np.asarray(median_mapping_one_section))
    upload_to_s3(fp)