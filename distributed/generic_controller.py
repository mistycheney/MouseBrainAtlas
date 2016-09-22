#! /usr/bin/env python

import os
import argparse
import sys
import cPickle as pickle

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Generic command launcher. This is run on one node. It sequentially executes the given command with variable arguments.')

parser.add_argument("command", type=str, help="command")
# parser.add_argument("-f", "--first", type=int, help="first slice index")
# parser.add_argument("-l", "--last", type=int, help="last slice index")
# parser.add_argument("-i", "--interval", type=int, help="interval")
parser.add_argument("-q", "--list", type=str, help="pickled string of list of arguments")
args = parser.parse_args()

# if args.list is not None:
arg_list = pickle.loads(args.list)

if isinstance(arg_list, dict):
    # {'a': [1,2,3], 'b': ['x','y','z']}
    arg_list = [dict(zip(arg_list.keys(), vals)) for vals in zip(arg_list.values())]
else:
    # [{'a':1, 'b':'x'}, {'a':2, 'b':'y'}, {'a':3, 'b':'z'}]
    assert isinstance(arg_list, list)

for arg in arg_list:
    os.system(args.command % arg)
# else:
#     for secind in range(args.first, args.last + 1, args.interval):
#         os.system(args.command % {'secind': secind})
