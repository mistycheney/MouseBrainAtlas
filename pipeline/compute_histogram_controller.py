#! /usr/bin/env python

import os
import argparse
import sys

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Compute texton histograms')

parser.add_argument("stack", type=str, help="stack name")
parser.add_argument("first", type=int, help="first slice index")
parser.add_argument("last", type=int, help="last slice index")
parser.add_argument("texton_path", type=str, help="path to textons.npy")
args = parser.parse_args()

for secind in range(args.first, args.last + 1):
    os.system(os.environ['GORDON_PIPELINE_SCRIPT_DIR']+"/compute_histogram.py %(stack)s %(secind)s %(texton_path)s"%\
              {'stack': args.stack, 
               'secind': secind,
               'texton_path': args.texton_path
               })