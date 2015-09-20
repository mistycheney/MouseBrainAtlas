#! /usr/bin/env python

import os
import argparse
import sys

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Execute feature extraction pipeline')

parser.add_argument("stack", type=str, help="stack name")
parser.add_argument("first", type=int, help="first slice index")
parser.add_argument("last", type=int, help="last slice index")
args = parser.parse_args()

for secind in range(args.first, args.last + 1):
    os.system(os.environ['GORDON_PIPELINE_SCRIPT_DIR']+"/gabor_filter.py %(stack)s %(secind)s"%\
              {'stack': args.stack, 
               'secind': secind
               })