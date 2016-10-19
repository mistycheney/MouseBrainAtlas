#! /usr/bin/env python

import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='visualize annotations, version 3')
parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("-f", "--first_sec", type=int, help="first index")
parser.add_argument("-l", "--last_sec", type=int, help="last index")
args = parser.parse_args()

######################################

import os
import sys
sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from visualization_utilities import *
from metadata import *

#####################################

stack = args.stack_name
sec = args.section

######################################

viz_dir = create_if_not_exists(annotation_midbrainIncluded_v2_rootdir + '/viz')

_ = annotation_v2_overlay_on('original', stack=stack, section=sec, users=['yuncong'], downscale_factor=8,
                    annotation_rootdir=annotation_midbrainIncluded_v2_rootdir,
                    export_filepath_fmt=os.path.join(viz_dir, stack, '%(stack)s_%(sec)04d_%(annofn)s.jpg'))



def convert_to_saturation(fn, out_fn, rescale=True):

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
    s = mi/ma.astype(np.float)
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
