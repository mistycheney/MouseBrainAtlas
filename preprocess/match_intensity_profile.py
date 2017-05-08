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
    description='')
parser.add_argument("stack", type=str, help="Stack name")
parser.add_argument("filename_pairs", type=str, help="Filename pairs")
args = parser.parse_args()

stack = args.stack
filename_pairs = json.loads(args.filename_pairs)

#############################################################################

def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """
    # http://stackoverflow.com/a/33047048

    oldshape = source.shape
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

    return interp_t_values[bin_idx].reshape(oldshape)
    
for nissl_fn, ntb_fn in filename_pairs:
    
    ntb_matched_values_all_examples_one_section = []
    region_bboxes_all_examples_one_section = []
    
    nissl_tb_mask = DataManager.load_thumbnail_mask_v3(stack=stack, fn=nissl_fn, version='cropped')
    
    t = time.time()
    ntb_im_fp = DataManager.get_image_filepath(stack=stack, fn=ntb_fn, version='cropped_16bit', resol='lossless')
    download_from_s3(ntb_im_fp)
    ntb_im = imread(ntb_im_fp)
    sys.stderr.write('Load NTB: %.2f seconds.\n' % (time.time()-t))

    t = time.time()
    # nissl_im_fp = DataManager.get_image_filepath(stack=stack, fn=nissl_fn, version='cropped', resol='lossless')
    # download_from_s3(nissl_im_fp)
    # nissl_im = img_as_ubyte(grb2gray(imread(nissl_im_fp)))
    nissl_im_fp = DataManager.get_image_filepath(stack=stack, fn=nissl_fn, version='cropped_gray', resol='lossless')
    download_from_s3(nissl_im_fp)
    nissl_im = img_as_ubyte(imread(nissl_im_fp))
    sys.stderr.write('Load Nissl: %.2f seconds.\n' % (time.time()-t))
    
    h, w = nissl_im.shape[:2]
    
    def f(region_id):
        
        while True:
            region1_x = np.random.randint(0, w-5000, 1)[0]
            region1_y = np.random.randint(0, h-5000, 1)[0]
            region1_w = 5000
            region1_h = 5000
            print region1_x, region1_y, region1_w, region1_h
            
            tb_region1_xmin = region1_x / 32
            tb_region1_xmax = (region1_x + region1_w) / 32
            tb_region1_ymin = region1_y / 32
            tb_region1_ymax = (region1_y + region1_h) / 32
            
            if np.all(np.r_[nissl_tb_mask[tb_region1_ymin, tb_region1_xmin],
            nissl_tb_mask[tb_region1_ymin, tb_region1_xmax],
            nissl_tb_mask[tb_region1_ymax, tb_region1_xmin],
            nissl_tb_mask[tb_region1_ymax, tb_region1_xmax]]):
                break
        
        ntb_blue_region1 = 3000 - ntb_im[region1_y:region1_y+region1_h, region1_x:region1_x+region1_w, 2]
        ntb_blue_bins = np.arange(0, ntb_blue_region1.max()+2)
        ntb_blue_hist = np.histogram(ntb_blue_region1.flatten(), bins=ntb_blue_bins)[0]
    
        nissl_region1 = nissl_im[region1_y:region1_y+region1_h, region1_x:region1_x+region1_w]
        nissl_gray_bins = np.arange(0, 257)
        nissl_gray_hist = np.histogram(nissl_region1.flatten(), bins=nissl_gray_bins)[0]

        ###############
        
        t = time.time()

        ntb_region1_hist_matched = hist_match(ntb_blue_region1, nissl_region1)

        ntb_to_nissl = {}
        for ntb_v in ntb_blue_bins[:-1]:
            a = ntb_region1_hist_matched[ntb_blue_region1 == ntb_v]
            if len(a) > 0:
                ntb_to_nissl[ntb_v] = np.unique(a)[0]

        ntb_values = np.arange(0, 5000)
        ntb_matched_values = np.interp(ntb_values, 
                                       [ntb_v for ntb_v, nissl_v in sorted(ntb_to_nissl.items())], 
                                       [nissl_v for ntb_v, nissl_v in sorted(ntb_to_nissl.items())])
    
        sys.stderr.write('Compute matching: %.2f seconds.\n' % (time.time()-t))
        
        return ntb_matched_values, (region1_x, region1_y, region1_w, region1_h)
        
    
    n_regions = 8
    
    pool = Pool(4)
    res = pool.map(f, range(n_regions))
    ntb_matched_values_all_examples_one_section, region_bboxes_all_examples_one_section = zip(*res)
    pool.close()
    pool.join()
    
#     for region_id in range(10):
        
#         while True:
#             region1_x = np.random.randint(0, w-10000, 1)[0]
#             region1_y = np.random.randint(0, h-10000, 1)[0]
#             region1_w = 5000
#             region1_h = 5000
#             print region1_x, region1_y, region1_w, region1_h
            
#             tb_region1_xmin = region1_x / 32
#             tb_region1_xmax = (region1_x + region1_w) / 32
#             tb_region1_ymin = region1_y / 32
#             tb_region1_ymax = (region1_y + region1_h) / 32
            
#             if np.all(np.r_[nissl_tb_mask[tb_region1_ymin, tb_region1_xmin],
#             nissl_tb_mask[tb_region1_ymin, tb_region1_xmax],
#             nissl_tb_mask[tb_region1_ymax, tb_region1_xmin],
#             nissl_tb_mask[tb_region1_ymax, tb_region1_xmax]]):
#                 break
        
#         ntb_blue_region1 = 3000 - ntb_im[region1_y:region1_y+region1_h, region1_x:region1_x+region1_w, 2]
#         ntb_blue_bins = np.arange(0, ntb_blue_region1.max()+2)
#         ntb_blue_hist = np.histogram(ntb_blue_region1.flatten(), bins=ntb_blue_bins)[0]
    
#         nissl_region1 = img_as_ubyte(rgb2gray(nissl_im[region1_y:region1_y+region1_h, region1_x:region1_x+region1_w]))
#         nissl_gray_bins = np.arange(0, 257)
#         nissl_gray_hist = np.histogram(nissl_region1.flatten(), bins=nissl_gray_bins)[0]

#         ###############
        
#         t = time.time()

#         ntb_region1_hist_matched = hist_match(ntb_blue_region1, nissl_region1)

#         ntb_to_nissl = {}
#         for ntb_v in ntb_blue_bins[:-1]:
#             a = ntb_region1_hist_matched[ntb_blue_region1 == ntb_v]
#             if len(a) > 0:
#                 ntb_to_nissl[ntb_v] = np.unique(a)[0]

#         ntb_values = np.arange(0, 5000)
#         ntb_matched_values = np.interp(ntb_values, 
#                                        [ntb_v for ntb_v, nissl_v in sorted(ntb_to_nissl.items())], 
#                                        [nissl_v for ntb_v, nissl_v in sorted(ntb_to_nissl.items())])

    
#         sys.stderr.write('Compute matching: %.2f seconds.\n' % (time.time()-t))

#         ntb_matched_values_all_examples_one_section.append(ntb_matched_values)
#         region_bboxes_all_examples_one_section.append((region1_x, region1_y, region1_w, region1_h))


    fp = os.path.join(DATA_DIR, stack, stack + '_intensity_mapping', '%s_to_%s_intensity_mapping_all_regions.npy' % (ntb_fn, nissl_fn))
    create_parent_dir_if_not_exists(fp)
    np.save(fp, np.asarray(ntb_matched_values_all_examples_one_section))
    upload_to_s3(fp)
    
    median_mapping_one_section = np.median(ntb_matched_values_all_examples_one_section, axis=0)    
    fp = os.path.join(DATA_DIR, stack, stack + '_intensity_mapping', '%s_intensity_mapping.npy' % (ntb_fn))
    np.save(fp, np.asarray(median_mapping_one_section))
    upload_to_s3(fp)

    fp = os.path.join(DATA_DIR, stack, stack + '_intensity_mapping', '%s_to_%s_region_bboxes.npy' % (ntb_fn, nissl_fn))
    np.save(fp, np.asarray(region_bboxes_all_examples_one_section))
    upload_to_s3(fp)
