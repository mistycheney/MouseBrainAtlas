#!/usr/bin/env python

import sys
import os
import cPickle as pickle

from preprocess_utility import *
import time

sys.path.append(os.environ['REPO_DIR'] + '/utilities/')
from utilities2015 import *

import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Crop the brainstem')

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("-first_sec", type=int, help="first section")
parser.add_argument("-last_sec", type=int, help="last section")
parser.add_argument("-x", type=int, help="x on thumbnail")
parser.add_argument("-y", type=int, help="y on thumbnail")
parser.add_argument("-w", type=int, help="w on thumbnail")
parser.add_argument("-H", type=int, help="h on thumbnail") # -h is reserved for --help
args = parser.parse_args()

stack = args.stack_name

x0, y0, w0, h0 = brainstem_bbox_lookup[stack]
f0, l0 = section_range_lookup[stack]

x = args.x if args.x is not None else x0
y = args.y if args.y is not None else y0
w = args.w if args.w is not None else w0
h = args.H if args.H is not None else h0
l = args.last_sec if args.last_sec is not None else l0
f = args.first_sec if args.first_sec is not None else f0

# DATAPROC_DIR = os.environ['DATA_DIR']

# os.system("""mkdir %(dataproc_dir)s/%(stack)s_thumbnail_aligned_bbox; mogrify -path %(dataproc_dir)s/%(stack)s_thumbnail_aligned_bbox -fill none -stroke black -draw "stroke-width 2 fill-opacity 0 rectangle %(x1)d,%(y1)d %(x2)d,%(y2)d" %(dataproc_dir)s/%(stack)s_thumbnail_aligned/*"""% \
#     {'stack': args.stack_name,
#     'dataproc_dir': DATAPROC_DIR,
#     'x1': x,
#     'y1': y,
#     'x2': x+w-1,
#     'y2': y+h-1}) 

# os.system("""mkdir %(dataproc_dir)s/%(stack)s_thumbnail_aligned_cropped; mogrify -set filename:name %%t -crop %(w)dx%(h)d+%(x)d+%(y)d -write "%(dataproc_dir)s/%(stack)s_thumbnail_aligned_cropped/%%[filename:name]_cropped.tif" %(dataproc_dir)s/%(stack)s_thumbnail_aligned/*.tif"""%\
# 	{'stack': args.stack_name, 
# 	'dataproc_dir': os.environ['DATA_DIR'],
# 	'w':w, 'h':h, 'x':x, 'y':y})


# os.system("""mkdir %(dataproc_dir)s/%(stack)s_thumbnail_aligned_mask_cropped; mogrify -set filename:name %%t -crop %(w)dx%(h)d+%(x)d+%(y)d -write "%(dataproc_dir)s/%(stack)s_thumbnail_aligned_mask_cropped/%%[filename:name]_cropped.png" %(dataproc_dir)s/%(stack)s_thumbnail_aligned_mask/*.png"""%\
#     {'stack': args.stack_name, 
#     'dataproc_dir': os.environ['DATA_DIR'],
#     'w':w, 'h':h, 'x':x, 'y':y})


os.system("""mkdir %(dataproc_dir)s/%(stack)s_thumbnail_aligned_masked_cropped; mogrify -set filename:name %%t -crop %(w)dx%(h)d+%(x)d+%(y)d -write "%(dataproc_dir)s/%(stack)s_thumbnail_aligned_masked_cropped/%%[filename:name]_cropped.png" %(dataproc_dir)s/%(stack)s_thumbnail_aligned_masked/*.png"""%\
    {'stack': args.stack_name, 
    'dataproc_dir': os.environ['DATA_DIR'],
    'w':w, 'h':h, 'x':x, 'y':y})

sys.exit(0)

script_dir = os.path.join(os.environ['REPO_DIR'], 'elastix')


t = time.time()
sys.stderr.write('expanding...')

expanded_tif_dir = os.environ['DATA_DIR'] + '/' + stack + '_lossless_renamed'
if not os.path.exists(expanded_tif_dir):
    os.makedirs(expanded_tif_dir)

jp2_dir = os.environ['DATA_DIR'] + '/' + stack + '_lossless_renamed_jp2'

run_distributed3('kdu_expand_patched -i %(jp2_dir)s/%(stack)s_%%(secind)04d_lossless.jp2 -o %(expanded_tif_dir)s/%(stack)s_%%(secind)04d_lossless.tif' % \
                    {'jp2_dir': jp2_dir,
                    'stack': args.stack_name,
                    'expanded_tif_dir': expanded_tif_dir},
                first_sec=f,
                last_sec=l,
                exclude_nodes=[33],
                stdout=open('/tmp/log', 'ab+'),
                take_one_section=True)

sys.stderr.write('done in %f seconds\n' % (time.time() - t))


t = time.time()
sys.stderr.write('warping and cropping...')

run_distributed3(command='%(script_path)s %(stack)s %(lossless_renamed_dir)s %(lossless_aligned_cropped_dir)s %%(f)d %%(l)d lossless %(x)d %(y)d %(w)d %(h)d'%\
                            {'script_path': script_dir + '/warp_crop_IM.py', 
                            'stack': args.stack_name,
                            'lossless_renamed_dir': os.path.join( os.environ['DATA_DIR'] , args.stack_name + '_lossless_renamed'),
                            'lossless_aligned_cropped_dir': os.path.join( os.environ['DATA_DIR'] , args.stack_name + '_lossless_aligned_cropped'),
                            'x': x,
                            'y': y,
                            'w': w,
                            'h': h
                            }, 
                first_sec=f,
                last_sec=l,
                exclude_nodes=[33],
                take_one_section=False)

sys.stderr.write('done in %f seconds\n' % (time.time() - t))


t = time.time()
sys.stderr.write('generate downscaled version and grayscale version...')

run_distributed3(command='%(script_path)s %(stack)s %%(f)d %%(l)d'%\
                            {'script_path': script_dir + '/generate_other_versions.py', 
                            'stack': args.stack_name
                            }, 
                first_sec=f,
                last_sec=l,
                exclude_nodes=[33],
                take_one_section=False)

sys.stderr.write('done in %f seconds\n' % (time.time() - t))
