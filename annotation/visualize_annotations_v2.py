#! /usr/bin/env python

import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='visualize annotations, version 2')
parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("section", type=int, help="section index")
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

viz_dir = create_if_not_exists(annotation_midbrainIncluded_rootdir + '/viz')

_ = annotation_overlay_on('original', stack, sec, users=['yuncong'], downscale_factor=8,
                    annotation_rootdir=annotation_midbrainIncluded_rootdir,
                    export_filepath_fmt=os.path.join(viz_dir, stack, '%(stack)s_%(sec)04d_%(annofn)s.jpg'))
