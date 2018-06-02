#! /usr/bin/env python

import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Generating landmark indices and dump to file.')
parser.add_argument("stack_name", type=str, help="stack name")
args = parser.parse_args()

################################

import os
import sys
import time

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from metadata import *

from learning_utilities import *

################################

stack = args.stack_name

################################

t = time.time()
sys.stderr.write('Generating landmark indices and dump to file ...')

indices_allLandmarks_allSections = locate_annotated_patches(stack, force=True)
fn = os.path.join(patch_rootdir, '%(stack)s_indices_allLandmarks_allSection.h5' % {'stack':stack})
indices_allLandmarks_allSections.to_hdf(fn, 'framewise_indices')

sys.stderr.write('done in %f seconds\n' % (time.time() - t))
