#! /usr/bin/env python

import os
import argparse
import sys

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Generic command launcher')

parser.add_argument("command", type=str, help="command")
parser.add_argument("-f", "--first", type=int, help="first slice index")
parser.add_argument("-l", "--last", type=int, help="last slice index")
parser.add_argument("-i", "--interval", type=int, help="interval")
parser.add_argument("-q", "--list", type=str, help="list of section indices, separated by underscore")
args = parser.parse_args()

if args.list is not None:
    for secind in map(int, args.list.split('_')):
        os.system(args.command % {'secind': secind})
else:
    for secind in range(args.first, args.last + 1, args.interval):
        os.system(args.command % {'secind': secind})
