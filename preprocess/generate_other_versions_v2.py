#! /usr/bin/env python

import sys
import os
import numpy as np
import json
from joblib import Parallel, delayed

# sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
# from utilities2015 import create_if_not_exists

import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Generate other versions of images, such as compressed jpeg and saturation channel as grayscale')

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("input_dir", type=str, help="input dir", default=None)
parser.add_argument("filenames", type=str, help="filenames, json string")
parser.add_argument("output_compressed_dir", type=str, help="output compressed dir", default=None)
parser.add_argument("output_saturation_dir", type=str, help="output saturation dir", default=None)
args = parser.parse_args()

stack = args.stack_name
# sys.stderr.write('==='+args.filenames + '\n')

filenames = json.loads(args.filenames)
input_dir = args.input_dir
output_compressed_dir = args.output_compressed_dir
output_saturation_dir = args.output_saturation_dir

os.system('mkdir ' + output_compressed_dir)
os.system('mkdir ' + output_saturation_dir)

from skimage.io import imread
from skimage.util import img_as_ubyte
import cv2

def convert_to_saturation(fn, out_fn, rescale=True):
    """
    Generate saturation channel as a grayscale image.
    """

# ImageMagick 18s
#     execute_command('convert %(fn)s -colorspace HSL -channel G %(out_fn)s' % {'fn': fn, 'out_fn': out_fn})

#     t = time.time()
    img = imread(fn)
#     sys.stderr.write('Read image: %.2f seconds\n' % (time.time() - t)) # ~4s

#     t1 = time.time()
    ma = img.max(axis=-1)
    mi = img.min(axis=-1)
#     sys.stderr.write('compute min and max color components: %.2f seconds\n' % (time.time() - t1)) # ~5s

#     t1 = time.time()
    s = np.nan_to_num(mi/ma.astype(np.float))
#     sys.stderr.write('min oiver max: %.2f seconds\n' % (time.time() - t1)) # ~2s

#     t1 = time.time()
    if rescale:
        pmax = s.max()
        pmin = s.min()
        s = (s - pmin) / (pmax - pmin)
#     sys.stderr.write('rescale: %.2f seconds\n' % (time.time() - t1)) # ~3s

#     t1 = time.time()
    cv2.imwrite(out_fn, img_as_ubyte(s))
#     imsave(out_fn, s)
#     sys.stderr.write('Compute saturation: %.2f seconds\n' % (time.time() - t1)) # skimage 6.5s; opencv 5s


def generate_versions(fn, which=['compressed', 'saturation']):

    input_fn=os.path.join(input_dir, fn)

    basename = os.path.splitext(os.path.basename(fn))[0]
    output_compressed_fn = os.path.join(output_compressed_dir, basename + '_compressed.jpg')
    output_saturation_fn = os.path.join(output_saturation_dir, basename + '_saturation.jpg')

    if 'compressed' in which:
        if os.path.exists(output_saturation_fn):
            sys.stderr.write('File exists: %s.\n' % output_compressed_fn)
        else:
            os.system("convert %(input_fn)s -format jpg %(output_compressed_fn)s" % \
                dict(input_fn=input_fn, output_compressed_fn=output_compressed_fn))

    if 'saturation' in which:
        if os.path.exists(output_saturation_fn):
            sys.stderr.write('File exists: %s.\n' % output_saturation_fn)
        else:
            convert_to_saturation(input_fn, output_saturation_fn, rescale=True)

Parallel(n_jobs=4)(delayed(generate_versions)(fn) for fn in filenames)
