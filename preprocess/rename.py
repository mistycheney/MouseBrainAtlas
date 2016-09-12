#! /usr/bin/env python

import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Rename thumbnail images according to our naming format with consecutive section numbers')

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("infer_order", type=int, help="whether to infer order", default=0)
parser.add_argument("-f", "--thumbnail_fmt", type=str, help="thumbnail format", default='tif')
args = parser.parse_args()

import sys
import os
import shutil
import numpy as np

thumbnail_fmt = args.thumbnail_fmt

stack = args.stack_name
input_dir = '/home/yuncong/CSHL_data/' + stack

output_dir = os.environ['DATA_DIR'] + '/' + stack + '_thumbnail_renamed'
if os.path.exists(output_dir):
    os.system('rm -r '+output_dir)
os.makedirs(output_dir)

output_jp2_dir = os.environ['DATA_DIR'] + '/' + stack + '_lossless_renamed_jp2'
if os.path.exists(output_jp2_dir):
    os.system('rm -r '+output_jp2_dir)
os.makedirs(output_jp2_dir)

filenames = os.listdir(input_dir)

from collections import defaultdict

def infer_order():

    d = defaultdict(dict)
    for fn in filenames:
        if fn.endswith(thumbnail_fmt):
            if fn[:-4].split('-')[1].startswith('F'):
                d[int(fn[:-4].split('_')[-1])]['F'] = fn[:-4]
            else:
                d[int(fn[:-4].split('_')[-1])]['IHC'] = fn[:-4]
    d.default_factory = None

    a = set([f[:-4] for f in filenames if f.endswith(thumbnail_fmt)])

    complete_set = []
    last_label = 'IHC'
    for i in sorted(d.keys()):
        if 'F' in d[i]:
            complete_set.append(d[i]['F'])
            last_label = 'F'
            if 'IHC' in d[i]:
                complete_set.append(d[i]['IHC'])
                last_label = 'IHC'
        else:
            complete_set.append(d[i]['IHC'])
            last_label = 'IHC'

    with open(os.environ['DATA_DIR']  + '/' + stack + '_filename_map.txt', 'w') as f:
        f.write('\n'.join([fn + ' ' + str(ind+1) for ind, fn in enumerate(complete_set)]))

if bool(args.infer_order):
    infer_order()

with open(os.environ['DATA_DIR']  + '/' + stack + '_filename_map.txt', 'r') as f:
    complete_set = [l.split() for l in f.readlines()]
    complete_set = {int(i): fn for fn, i in complete_set}

d = {'input_dir': input_dir,
    'output_dir': output_dir,
    'output_jp2_dir': output_jp2_dir}

# print sorted([(new_ind, fn) for fn, new_ind in complete_set.iteritems()])

for new_ind, fn in complete_set.iteritems():

    if fn == 'Placeholder' or fn == 'Rescan':
        continue

    d['fn_base'] = fn
    d['new_fn_base'] = stack + '_%04d'%(new_ind)

    shutil.copy(input_dir + '/' + fn + '.'+thumbnail_fmt, output_dir + '/' + d['new_fn_base'] + '_thumbnail.'+thumbnail_fmt)

    os.system('ln -s %(input_dir)s/%(fn_base)s_lossless.jp2 %(output_jp2_dir)s/%(new_fn_base)s_lossless.jp2' % d)
