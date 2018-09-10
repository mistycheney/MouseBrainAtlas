#! /usr/bin/env python

import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Linearly normalize intensity to between 0 and 255')

parser.add_argument("input_spec", type=str, help="Input specification")
parser.add_argument("out_version", type=str, help="Output image version")
args = parser.parse_args()

import sys
import os

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from data_manager import *
from metadata import *
from distributed_utilities import *
from learning_utilities import *

input_spec = load_ini(args.input_spec)
image_name_list = input_spec['image_name_list']
stack = input_spec['stack']
prep_id = input_spec['prep_id']
if prep_id == 'None':
    prep_id = None
resol = input_spec['resol']
version = input_spec['version']
if version == 'None':
    version = None


from scipy.ndimage.interpolation import map_coordinates
from skimage.exposure import rescale_intensity, adjust_gamma
from skimage.transform import rotate

# for section in set(metadata_cache['valid_sections_all'][stack]) - set(metadata_cache['valid_sections'][stack]):
# for section in metadata_cache['valid_sections'][stack]:

for image_name in image_name_list:

#     print "Section", section
    
    t = time.time()
    
    img = DataManager.load_image_v2(stack=stack, prep_id=prep_id, fn=image_name, version=version, resol=resol)

    sys.stderr.write('Load image: %.2f seconds.\n' % (time.time() - t))

    t = time.time()
    tb_mask = DataManager.load_thumbnail_mask_v3(stack=stack, prep_id=None, fn=image_name)
#     raw_mask = rescale_by_resampling(tb_mask, new_shape=(img.shape[1], img.shape[0]))
    raw_mask = resize(tb_mask, img.shape) > .5
    
    save_data(raw_mask, 
          DataManager.get_image_filepath_v2(stack=stack, prep_id=prep_id, fn=image_name, version='mask', resol=resol, ext='bp'), 
          upload_s3=False)
    
    sys.stderr.write('Rescale mask: %.2f seconds.\n' % (time.time() - t))

    t = time.time()
    
    mean_std_all_regions = []
    cx_cy_all_regions = []
    region_size = 5000
    region_spacing = 3000
#     for cx in range(region_size/2, img.shape[1]-region_size/2+1, region_spacing):
#         for cy in range(region_size/2, img.shape[0]-region_size/2+1, region_spacing):
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
    
    fp = DataManager.get_intensity_normalization_result_filepath(what='region_centers', stack=stack, fn=image_name)
    create_parent_dir_if_not_exists(fp)    
    np.savetxt(fp, cx_cy_all_regions)
    
    fp = DataManager.get_intensity_normalization_result_filepath(what='mean_std_all_regions', stack=stack, fn=image_name)
    create_parent_dir_if_not_exists(fp)
    np.savetxt(fp, mean_std_all_regions)
    
    fp = DataManager.get_intensity_normalization_result_filepath(what='mean_map', stack=stack, fn=image_name)
    create_parent_dir_if_not_exists(fp)
    bp.pack_ndarray_file(mean_map.astype(np.float16), fp)
    
    fp = DataManager.get_intensity_normalization_result_filepath(what='std_map', stack=stack, fn=image_name)
    create_parent_dir_if_not_exists(fp)
    bp.pack_ndarray_file(std_map.astype(np.float16), fp)

    # Export normalized image.
    
    t = time.time()
    raw_mask = raw_mask & (std_map > 0)
    img_normalized = np.zeros(img.shape, np.float32)
    img_normalized[raw_mask] = (img[raw_mask] - mean_map[raw_mask]) / std_map[raw_mask]
    sys.stderr.write('Normalize: %.2f seconds.\n' % (time.time() - t)) #30s

    t = time.time()
    # FIX THIS! THIS only save uint16, not float16. Need to save as bp instead.
#     img_fp = DataManager.get_image_filepath_v2(stack=stack, prep_id=None, version='NtbNormalizedFloat', resol='down8', section=section, )
#     create_parent_dir_if_not_exists(img_fp)
#     imsave(img_fp, img_normalized[::8, ::8].astype(np.float16))
    save_data(img_normalized.astype(np.float16), 
              DataManager.get_intensity_normalization_result_filepath(what='normalized_float_map', stack=stack, fn=image_name),
             upload_s3=False)
    sys.stderr.write('Save float version: %.2f seconds.\n' % (time.time() - t)) #30s
        
#     t = time.time()
#     img_normalized_uint8 = rescale_intensity_v2(img_normalized, -1, 6)
#     sys.stderr.write('Rescale to uint8: %.2f seconds.\n' % (time.time() - t)) #30s
    
#     t = time.time()
#     img_fp = DataManager.get_image_filepath_v2(stack=stack, prep_id=None, version='NtbNormalized', resol='raw', section=section)
#     create_parent_dir_if_not_exists(img_fp)
#     imsave(img_fp, img_normalized_uint8)
#     sys.stderr.write('Save uint8 version: %.2f seconds.\n' % (time.time() - t)) #30s
    
    # Export histogram.
    
    plt.hist(img_normalized[raw_mask].flatten(), bins=100, log=True);
    fp = DataManager.get_intensity_normalization_result_filepath(what='float_histogram_png', stack=stack, fn=image_name)
    create_parent_dir_if_not_exists(fp)
    plt.savefig(fp)
    plt.close();
    
#     hist_fp = DataManager.get_intensity_normalization_result_filepath(what='float_histogram', stack=stack, section=section)
#     create_parent_dir_if_not_exists(hist_fp)
    
#     hist, bin_edges = np.histogram(img_normalized[valid_mask].flatten(), bins=np.arange(0,201,5));

#     plt.bar(bin_edges[:-1], np.log(hist));
#     plt.xticks(np.arange(0, 200, 20), np.arange(0, 200, 20));
#     plt.xlabel('Normalized pixel value (float)');
#     plt.title(metadata_cache['sections_to_filenames'][stack][section])

#     plt.savefig(hist_fp)
#     plt.close();

gamma_map = img_as_ubyte(adjust_gamma(np.arange(0, 256, 1) / 255., 8.))

low = -2.
high = 50.

for image_name in image_name_list:

    img_normalized = load_data(
              DataManager.get_intensity_normalization_result_filepath(what='normalized_float_map', stack=stack, fn=image_name),
             download_s3=False)    
    
    t = time.time()
    img_normalized_uint8 = rescale_intensity_v2(img_normalized, low, high)
    sys.stderr.write('Rescale to uint8: %.2f seconds.\n' % (time.time() - t))

    t = time.time()
    raw_mask = load_data(DataManager.get_image_filepath_v2(stack=stack, prep_id=prep_id, fn=image_name, version='mask', resol=resol, ext='bp'),
                        download_s3=False)
    img_normalized_uint8[~raw_mask] = 0
    sys.stderr.write('Load mask: %.2f seconds.\n' % (time.time() - t))
        
    img = 255 - img_normalized_uint8
    save_data(gamma_map[img], 
              DataManager.get_image_filepath_v2(stack=stack, prep_id=prep_id, fn=image_name, version=args.out_version, resol=resol),
             upload_s3=False)
