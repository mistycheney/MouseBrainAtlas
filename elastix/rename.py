#! /usr/bin/env python

import sys
import os
import shutil
import argparse

import numpy as np

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Rename thumbnail images according to our naming format with consecutive section numbers')

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("infer_order", type=int, help="whether to infer order", default=0)
args = parser.parse_args()

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

### Alex's rule ###  DOES NOT WORK !!!

# d = defaultdict(lambda: defaultdict(list))
# for fn in filenames:
#     if fn.endswith('tif'):
#         slide_str = fn[:-4].split('-')[1]
#         if slide_str.startswith('N'):
#             d[int(slide_str[1:])]['N'].append(fn[:-4])
#         else:
#             d[int(slide_str[3:])]['IHC'].append(fn[:-4])
# d.default_factory = None

# complete_set = []
# for i in sorted(d.keys()):
#     if 'N' in d[i]:
#         N_set = sorted(d[i]['N'], key=lambda x: int(x.split('_')[-1]))
#     else:
#         N_set = []

#     if 'IHC' in d[i]:
#         IHC_set = sorted(d[i]['IHC'], key=lambda x: int(x.split('_')[-1]))
#     else:
#         IHC_set = []

#     for i in range(max(len(N_set), len(IHC_set))):
#         if i < len(N_set):
#             complete_set.append(N_set[i])
#         if i < len(IHC_set):
#             complete_set.append(IHC_set[i])


### My rule ###

# remove = []
# swap = []
# if stack == 'MD595':
#     remove = ['MD595-N1-2015.09.14-19.07.48_MD595_1_0001', 'MD595-N84-2015.09.15-00.45.35_MD595_2_0251'] 
# elif stack == 'MD598':
#     swap = [(1,2), (3,4), (145,146), (147,148), (226,227)]
#     # pass
# elif stack == 'MD589':
#     swap = [(300,301), (302,303), (323,324), (325,326)]
#     # pass
# elif stack == 'MD594':
#     remove = [
#     'MD594-IHC30-2015.08.26-17.00.29_MD594_2_0089',
#     'MD594-IHC31-2015.08.26-17.04.03_MD594_2_0092',
#     'MD594-IHC32-2015.08.26-17.07.31_MD594_2_0095'
#     ] 
# elif stack == 'MD585':
#     pass

# complete_set = [x for x in complete_set if x not in remove]
# for a,b in swap:
#     tmp = complete_set[a-1]
#     complete_set[a-1] = complete_set[b-1]
#     complete_set[b-1] = tmp

# # if stack in ['MD589', 'MD594']:

# #     # revert remove, because in the old set we did not remove duplicates
# #     old_set = sorted([ fn for fn in complete_set + remove if fn[:-4].split('-')[1].startswith('IHC')], key=lambda a: int(a.split('_')[-1]))

# #     map_old_to_new = [ (i+1, complete_set.index(fn)+1) for i, fn in enumerate(old_set) if fn not in remove]
# #     with open(os.environ['DATA_DIR']  + '/' + stack+'_indexMapOldToNew.txt', 'w') as f:
# #         for o, n in map_old_to_new:
# #             f.write('%d %d\n'%(o, n))


if bool(args.infer_order):

    d = defaultdict(dict)
    for fn in filenames:
        if fn.endswith('tif'):
            if fn[:-4].split('-')[1].startswith('N'):
                d[int(fn[:-4].split('_')[-1])]['N'] = fn[:-4]
            else:
                d[int(fn[:-4].split('_')[-1])]['IHC'] = fn[:-4]
    d.default_factory = None

    a = set([f[:-4] for f in filenames if f.endswith('tif')])

    complete_set = []
    last_label = 'IHC'
    for i in sorted(d.keys()):
        # if i == 231:
        #     print d[i]
        # if last_label == 'IHC':
        if 'N' in d[i]:
            complete_set.append(d[i]['N'])
            last_label = 'N'
            if 'IHC' in d[i]:
                complete_set.append(d[i]['IHC'])
                last_label = 'IHC'
        else:
            complete_set.append(d[i]['IHC'])
            last_label = 'IHC'

    # print a - set(complete_set)
    # print set(complete_set) - a 

    with open(os.environ['DATA_DIR']  + '/' + stack + '_filename_map.txt', 'w') as f:
        f.write('\n'.join([fn + ' ' + str(ind+1) for ind, fn in enumerate(complete_set)]))


with open(os.environ['DATA_DIR']  + '/' + stack + '_filename_map.txt', 'r') as f:
    complete_set = [l.split() for l in f.readlines()]
    complete_set = dict([(fn, int(i)) for fn, i in complete_set])


d = {'input_dir': input_dir,
    'output_dir': output_dir,
    'output_jp2_dir': output_jp2_dir}

for fn, new_ind in complete_set.iteritems():
    d['fn_base'] = fn
    d['new_fn_base'] = stack + '_%04d'%(new_ind)

    shutil.copy(input_dir + '/' + fn + '.tif', output_dir + '/' + d['new_fn_base'] + '_thumbnail.tif')

    os.system('ln -s %(input_dir)s/%(fn_base)s_lossless.jp2 %(output_jp2_dir)s/%(new_fn_base)s_lossless.jp2' % d)


