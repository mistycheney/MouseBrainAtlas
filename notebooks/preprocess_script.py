# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import glob, re, os, sys, subprocess, argparse

parser = argparse.ArgumentParser()
parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("start_frame", type=int, help="start frame")
parser.add_argument("stop_frame", type=int, help="stop frame")
parser.add_argument("reduce_level", type=int, help="reduce level (integer, 0 being the max size)")
parser.add_argument("-b", "--bbox_file", type=str, 
                    help="file containing bounding box coordinates, n rows by 4 cols, each row is <top>,<left>,<height>,<width>; if not specified, bounding boxes are automatically extracted")
parser.add_argument("-i", "--data_dir", type=str, help="data directory", default='.')
parser.add_argument("-o", "--out_dir", type=str, help="output directory")
args = parser.parse_args()

import cv2
import matplotlib.pyplot as plt
import numpy as np

import skimage
from skimage.filter import threshold_otsu, gaussian_filter
from skimage.measure import regionprops, label

stack_name = args.stack_name
DATA_DIR = os.path.realpath(args.data_dir)
IMG_DIR = os.path.join(DATA_DIR, stack_name)

# uncompress tarball, if not having done so

os.chdir(DATA_DIR)
if not os.path.exists(stack_name):
    if os.path.exists(stack_name+'.tar.gz'):
        return_code = subprocess.call('tar xfz %s.tar.gz'%stack_name, shell=True)
    elif os.path.exists(stack_name+'.tar'):
        return_code = subprocess.call('tar xf %s.tar'%stack_name, shell=True)

# find bounding box for all images

bbox = dict([])
if args.bbox_file is None:
    margin = 0 # at lowest level 120 by 90
#     images = dict([])
    os.chdir(IMG_DIR)
    for fn in glob.iglob('*.jpg'):
        m = re.match('.*_([0-9]{4}).jpg', fn)
        idx = int(m.groups()[0])
        img = cv2.imread(fn, 0)
        blurred = gaussian_filter(img, 2)
        thresholded = blurred < threshold_otsu(blurred) + 10./255.
        labelmap, n_components = label(thresholded, background=0, return_num=True)    
        component_props = skimage.measure.regionprops(labelmap+1, cache=True)
        major_comp_prop = sorted(component_props, key=lambda x: x.area, reverse=True)[0]
        y1, x1, y2, x2 = major_comp_prop.bbox
        bbox[idx] = [float(y1)/img.shape[0], float(x1)/img.shape[1], 
                     float(y2-y1)/img.shape[0], float(x2-x1)/img.shape[1]]
#         images[idx] = img
    n_images = np.max(bbox.keys())
else:
    os.chdir(IMG_DIR)
    mat_id_bbox = np.loadtxt(os.path.join(DATA_DIR, args.bbox_file), delimiter=',')
    for row in mat_id_bbox:
        bbox[row[0]] = row[1:]

# extract the bounding box at specific reduce level
level = args.reduce_level

if args.out_dir is None:
    output_dir = os.path.join(DATA_DIR, stack_name+'_reduce%d'%level)
else:
    output_dir = os.path.join(DATA_DIR, args.out_dir)
    
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for jp2 in glob.iglob('*.jp2'):
    m = re.match('.*_([0-9]{4}).jp2', jp2)
    idx = int(m.groups()[0])
    if idx < args.start_frame or idx > args.stop_frame: continue
    tif = '%s_%04d_reduce%d.tif'%(stack_name, idx, level)
    top,left,height,width = bbox[idx]

    command = "kdu_expand -i '%s' -o '%s' -reduce %d -region '{%f,%f},{%f,%f}'"%(jp2, os.path.join(output_dir, tif),
                                                                                 level,top,left,height,width)
    return_code = subprocess.call(command, shell=True)

