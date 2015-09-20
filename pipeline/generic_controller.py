#! /usr/bin/env python

import os
import argparse
import sys

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Generic command launcher')

parser.add_argument("command", type=str, help="command")
parser.add_argument("first", type=int, help="first slice index")
parser.add_argument("last", type=int, help="last slice index")
args = parser.parse_args()

for secind in range(args.first, args.last + 1):
    os.system(args.command % {'secind': secind})