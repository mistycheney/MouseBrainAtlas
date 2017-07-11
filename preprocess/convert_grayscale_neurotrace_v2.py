#! /usr/bin/env python

import sys
import os
sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))

from skimage.exposure import rescale_intensity

from utilities2015 import *
from registration_utilities import *
from metadata import *
from data_manager import *

########################################

import argparse
import json

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Convert to grayscale with nissl-like intensity profile.')

parser.add_argument("stack", type=str, help="Stack")
parser.add_argument("filenames", type=str, help="filenames")
parser.add_argument("-l", "--low", type=int, help="Low intensity limit for linear contrast stretch")
parser.add_argument("-H", "--high", type=int, help="High intensity limit for linear contrast stretch")
parser.add_argument("-o", "--output_version", type=str, help="Output image version", default='gray')
# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
parser.add_argument('--not_use_section_specific', dest='use_sec_specific', action='store_false', help="Not use section-specific intensity mapping")
parser.set_defaults(use_section_specific=True)
args = parser.parse_args()

####################################

stack = args.stack
filenames = json.loads(args.filenames)
use_section_specific_mapping = args.use_sec_specific
output_version = args.output_version

for fn in filenames:
    
    # try:
    t = time.time()
    img_fp = DataManager.get_image_filepath_v2(stack=stack, fn=fn, prep_id=2, resol='lossless')
    download_from_s3(img_fp, local_root=DATA_ROOTDIR)
    
    sys.stderr.write('Download: %.2f seconds\n' % (time.time() - t))

    if stack not in all_nissl_stacks and fn.split('-')[1][0] == 'F':
        # Neurotrace sections.

        t = time.time()
        img_blue = imread(img_fp)[..., 2]
        sys.stderr.write('Read: %.2f seconds\n' % (time.time() - t))

        if not hasattr(args, 'low') or (hasattr(args, 'low') and args.low is None):
            sys.stderr.write("No linear limits arguments are given, so use nonlinear mapping.\n")

            if use_section_specific_mapping:
                try:
                    intensity_mapping_fp = DataManager.get_ntb_to_nissl_intensity_profile_mapping_filepath(stack=stack, ntb_fn=fn)
                    download_from_s3(intensity_mapping_fp)
                    intensity_mapping_ntb_to_nissl = np.load(intensity_mapping_fp)
                    load_default_mapping = False
                    sys.stderr.write("Loaded section specific mapping.\n")
                except:
                    sys.stderr.write("Error loading section-specific ntb-to-nissl intensity mapping. Load a default mapping instead.\n")
                    load_default_mapping = True
            else:
                load_default_mapping = True
                    
            if load_default_mapping:            
                intensity_mapping_fp = DataManager.get_ntb_to_nissl_intensity_profile_mapping_filepath()
                download_from_s3(intensity_mapping_fp)
                intensity_mapping_ntb_to_nissl = np.load(intensity_mapping_fp)

            t = time.time()
            img_blue_intensity_normalized = intensity_mapping_ntb_to_nissl[img_blue.astype(np.int)].astype(np.uint8)
            # print intensity_mapping_ntb_to_nissl
            # print img_blue.min(), img_blue.max(), intensity_mapping_ntb_to_nissl.shape
            # print img_blue_intensity_normalized.min(), img_blue_intensity_normalized.max()
            sys.stderr.write('Convert: %.2f seconds\n' % (time.time() - t))

            output_fp = DataManager.get_image_filepath_v2(stack=stack, fn=fn, prep_id=2, version=output_version, resol='lossless')

        else:
            sys.stderr.write("Linear limits arguments detected, so use linear mapping.\n")

            low_limit = args.low
            high_limit = args.high

            t = time.time()
            img_blue_intensity_normalized = rescale_intensity_v2(img_blue, low_limit, high_limit)
            sys.stderr.write('Convert: %.2f seconds\n' % (time.time() - t))

            output_fp = DataManager.get_image_filepath_v2(stack=stack, fn=fn, prep_id=2, version=output_version, resol='lossless')

        t = time.time()
        create_parent_dir_if_not_exists(output_fp)
        imsave(output_fp, img_blue_intensity_normalized)
        sys.stderr.write('Save: %.2f seconds\n' % (time.time() - t))

        t = time.time()
        upload_to_s3(output_fp, local_root=DATA_ROOTDIR)
        sys.stderr.write('Upload: %.2f seconds\n' % (time.time() - t))

    else:
        # Nissl sections.

        t = time.time()
        img = imread(img_fp)
        sys.stderr.write('Read: %.2f seconds\n' % (time.time() - t))

        t = time.time()
        img_gray = img_as_ubyte(rgb2gray(img))
        sys.stderr.write('Convert: %.2f seconds\n' % (time.time() - t))

        output_fp = DataManager.get_image_filepath_v2(stack=stack, fn=fn, prep_id=2, version=output_version, resol='lossless')
        create_parent_dir_if_not_exists(output_fp)

        t = time.time()
        imsave(output_fp, img_gray)
        sys.stderr.write('Save: %.2f seconds\n' % (time.time() - t))

        t = time.time()
        upload_to_s3(output_fp, local_root=DATA_ROOTDIR)
        sys.stderr.write('Upload: %.2f seconds\n' % (time.time() - t))

        # else:
        #     sys.stderr.write("Filename %s is not F or N.\n" % fn)

    # except Exception as e:
    #     sys.stderr.write('%s\n' % e)
        