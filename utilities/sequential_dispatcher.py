#! /usr/bin/env python

import os
import argparse
import sys
import json

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Generic command launcher. This is run on one node. It sequentially executes the given command with variable arguments.')

parser.add_argument("command", type=str, help="command")
parser.add_argument('kwargs_list_str', type=str, help="json-encoded string of list of arguments")
args = parser.parse_args()

kwargs_list = json.loads(args.kwargs_list_str)

if isinstance(kwargs_list, dict):
    # {'a': [1,2,3], 'b': ['x','y','z']}
    kwargs_list = [dict(zip(kwargs_list.keys(), vals)) for vals in zip(kwargs_list.values())]
else:
    # [{'a':1, 'b':'x'}, {'a':2, 'b':'y'}, {'a':3, 'b':'z'}]
    assert isinstance(kwargs_list, list)

for kwargs in kwargs_list:
    os.system(args.command % kwargs)