#! /usr/bin/env python

import sys
import os
sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import execute_command

import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Neurotrace blue to nissl')

#parser.add_argument("input_dir", type=str, help="input dir")
#parser.add_argument("output_dir", type=str, help="output dir")
#parser.add_argument("filename", type=str, help="full filename, with extension")
# parser.add_argument("output_fn", type=str, help="output filename")
parser.add_argument("input_fp", type=str, help="input file path")
parser.add_argument("output_fp", type=str, help="output file path")
args = parser.parse_args()

#input_dir = args.input_dir
#filename = args.filename
#output_dir = args.output_dir
# output_fn = args.output_fn

#input_fp = os.path.join(input_dir, filename)
input_fp = args.input_fp
output_fp = args.output_fp
#basename = os.path.splitext(os.path.basename(filename))[0]
#blue_fp = os.path.join(output_dir, basename + '_blue.tif')
#output_fp = os.path.join(output_dir, basename + '_blueasgray.tif')

# Cannot chain: final output is not grayscale for some reason.
#execute_command('convert %(input_fp)s -channel B -separate %(blue_fp)s' % \
#                dict(input_fp=input_fp, blue_fp=blue_fp))
#execute_command('convert %(blue_fp)s -negate %(output_fp)s' % \
#                dict(blue_fp=blue_fp, output_fp=output_fp))
#execute_command('rm %(blue_fp)s' % dict(blue_fp=blue_fp))

from distributed_utilities import *
import tempfile

fd, tmp_fp = tempfile.mkstemp()
try:
    # Cannot chain: final output is not grayscale for some reason.
    download_from_s3(input_fp)
    execute_command('convert %(input_fp)s -channel B -separate %(blue_fp)s' % \
                    dict(input_fp=input_fp, blue_fp=tmp_fp))
    execute_command('convert %(blue_fp)s -negate %(output_fp)s' % \
                    dict(blue_fp=tmp_fp, output_fp=output_fp))
    upload_to_s3(output_fp)
finally:
    os.remove(tmp_fp)
