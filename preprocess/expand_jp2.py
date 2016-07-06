#! /usr/bin/env python

import sys
import os
import shutil
import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Rename thumbnail images according to our naming format with consecutive section numbers')

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("first_sec", type=int, help="first section")
parser.add_argument("last_sec", type=int, help="last section")
args = parser.parse_args()

stack = args.stack_name

from preprocess_utility import run_distributed3

expanded_tif_dir = os.environ['DATA_DIR'] + '/' + stack + '_lossless_renamed'
if not os.path.exists(expanded_tif_dir):
    os.makedirs(expanded_tif_dir)

jp2_dir = os.environ['DATA_DIR'] + '/' + stack + '_lossless_renamed_jp2'

run_distributed3('kdu_expand_patched -i %(jp2_dir)s/%(stack)s_%%(secind)04d_lossless.jp2 -o %(expanded_tif_dir)s/%(stack)s_%%(secind)04d_lossless.tif' % \
                    {'jp2_dir': jp2_dir,
                    'stack': stack,
                    'expanded_tif_dir': expanded_tif_dir},
                first_sec=args.first_sec,
                last_sec=args.last_sec,
                # last_sec=5,
                stdout=open('/tmp/log', 'ab+'),
                take_one_section=True)


