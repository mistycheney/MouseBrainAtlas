#! /usr/bin/env python

import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Upload latest annotations to Gordon')

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("-l", "--local_dir", type=str, help="local annotation dir")
parser.add_argument("-r", "--remote_dir", type=str, help="remote annotation dir")
args = parser.parse_args()

import os
import sys
sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from data_manager import *
from metadata import *

# This script is supposed to be run on the local machine

# stack = 'MD594'
stack = args.stack_name
first_bs_sec, last_bs_sec = section_range_lookup[stack]

annotations_rootdir_local = args.local_dir if args.local_dir is not None else '/home/yuncong/CSHL_data_labelings_losslessAlignCropped_midbrainIncluded/'
annotations_rootdir_gordon = args.remote_dir if args.remote_dir is not None else '/home/yuncong/csd395/CSHL_data_labelings_losslessAlignCropped_midbrainIncluded/'

cmd = 'ssh yuncong@oasis-dm.sdsc.edu mkdir %(annotations_rootdir)s/%(stack)s' % \
{'stack': stack, 'annotations_rootdir': annotations_rootdir_gordon}

os.system(cmd)

import time
t = time.time()

for sec in range(first_bs_sec, last_bs_sec+1):

    annotations_dir_gordon = os.path.join(annotations_rootdir_gordon, stack, '%04d'%sec)

    username = 'yuncong'

    try:
        fn_path = DataManager.get_annotation_path(stack=stack, section=sec, username=username, annotation_rootdir=annotations_rootdir_local)[0]
        cmd = 'ssh yuncong@oasis-dm.sdsc.edu mkdir %(annotations_dir_gordon)s; \
        scp %(fn)s yuncong@oasis-dm.sdsc.edu:%(annotations_dir_gordon)s' % \
        {'fn': fn_path, 'annotations_dir_gordon': annotations_dir_gordon}
        print cmd
        os.system(cmd)
    except:
        continue

sys.stderr.write('upload annotations: %.2f seconds\n' % (time.time() - t)) # ~ s / entire stack
