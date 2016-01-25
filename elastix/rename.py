#! /usr/bin/env python

import sys
import os
import shutil
import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Rename thumbnail images according to our naming format with consecutive section numbers')

parser.add_argument("stack_name", type=str, help="stack name")
args = parser.parse_args()

stack = args.stack_name
input_dir = '/home/yuncong/CSHL_data/' + stack

output_dir = os.environ['DATA_DIR'] + '/' + stack + '_thumbnail_renamed'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_jp2_dir = os.environ['DATA_DIR'] + '/' + stack + '_lossless_renamed_jp2'
if not os.path.exists(output_jp2_dir):
    os.makedirs(output_jp2_dir)

# filenames = os.listdir(input_dir)
# tuple_sorted = sorted(sorted([fn[:-4] for fn in filenames if fn.endswith('tif')], key=lambda fn: len(fn.split('-')[1])), key=lambda fn: int(fn.split('_')[-1]))

# with open(os.environ['DATA_DIR']  + '/' + stack + '_filename_map.txt', 'w') as f:
#     f.write('\n'.join([fn + ' ' + str(ind) for ind, fn in enumerate(tuple_sorted)]))

# d = {'input_dir': input_dir,
#     'output_dir': output_dir,
#     'output_jp2_dir': output_jp2_dir}

# for new_ind, fn in enumerate(tuple_sorted):
#     d['fn_base'] = fn
#     d['new_fn_base'] = stack + '_%04d'%(new_ind+1)

#     shutil.copy(input_dir + '/' + fn + '.tif', output_dir + '/' + d['new_fn_base'] + '_thumbnail.tif')

#     # os.system('kdu_expand_patched -i %(input_dir)s/%(fn_base)s_lossless.jp2 -o %(output_dir)s/%(new_fn_base)s_lossless.tif' % d)
#     os.system('ln -s %(input_dir)s/%(fn_base)s_lossless.jp2 %(output_jp2_dir)s/%(new_fn_base)s_lossless.jp2' % d)


from preprocess_utility import run_distributed3

expanded_tif_dir = os.environ['DATA_DIR'] + '/' + stack + '_lossless_renamed'
if not os.path.exists(expanded_tif_dir):
    os.makedirs(expanded_tif_dir)

run_distributed3('kdu_expand_patched -i %(output_jp2_dir)s/%(stack)s_%%(secind)04d_lossless.jp2 -o %(expanded_tif_dir)s/%(stack)s_%%(secind)04d_lossless.tif' % \
                    {'output_jp2_dir': output_jp2_dir,
                    'stack': stack,
                    'expanded_tif_dir': expanded_tif_dir},
                first_sec=1,
                # last=len(tuple_sorted),
                last_sec=5,
                stdout=open('/tmp/log', 'ab+'),
                take_one_section=True)


