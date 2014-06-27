# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import glob, re, os, sys, subprocess, argparse

parser = argparse.ArgumentParser(
formatter_class=argparse.RawDescriptionHelpFormatter,
description="Dataset creation utility",
epilog="""
For example, the following command generates images for dataset PMD1305_region0 that are 4 times smaller in each dimension than the full resolution:

python %s ../dataset_defs/PMD1305_region0_reduce2.txt

A dataset definition file contains a single integer at the first row, which specifies the reduce level (0 being the full resolution).
Below this are multiple rows of comma separated values. Each row specifies:
stack_name, image_index, [top, left, height, width]

Here stack_name is a string such as PMD1305, image_index is an integer. 
The last four parameters specify a bounding box, and are optional. If specified, they must be float-point numbers between 0 and 1, not integers. If not specified, a bounding box is automatically computed for each image. The sizes and positions of the bounding boxes may differ for different images in the dataset.
For an example of a dataset definition file, see ../dataset_defs/PMD1305_region0_reduce2.txt, which specifies a particular brainstem regions in stack PMD1305.

By default, the script will create a sub-directory under the data directory, with the same name as the datasest definition file, to store the output tif files. The tif files are named <dataset_name>_reduce<reduce_level>_<image_index>.tif
"""%(os.path.basename(sys.argv[0]))
)

parser.add_argument("dataset_def", type=str, help="dataset definition file")
parser.add_argument("-i", "--data_dir", type=str, help="data directory (default: %(default)s)", default='/oasis/projects/nsf/csd181/yuncong/ParthaData')
parser.add_argument("-o", "--out_dir", type=str, help="output directory")
args = parser.parse_args()

import cv2
import matplotlib.pyplot as plt
import numpy as np

from skimage.filter import threshold_otsu, gaussian_filter
from skimage.measure import regionprops, label

data_dir = os.path.realpath(args.data_dir)

dataset_path = os.path.realpath(args.dataset_def)
dataset_dir, dataset_fn = os.path.split(dataset_path)
dataset_name, _ = os.path.splitext(dataset_fn)
    
dataset_mat = np.genfromtxt(dataset_path, delimiter=',', filling_values=-1, dtype=None, skip_header=1)

n_images = dataset_mat.shape[0]
stack_names = dataset_mat['f0']
image_indices = dataset_mat['f1']
if len(dataset_mat.dtype.names) > 2:
    bounding_boxes = dataset_mat[['f2','f3','f4','f5']].view((float, 4))
else:
    bounding_boxes = -1*np.ones((n_images, 4), dtype=np.float)

with open(dataset_path, 'r') as f:
    level = int(f.readline()[0])
    
# uncompress tarball, if not having done so
os.chdir(data_dir)
for stack_name in set(stack_names):
    if not os.path.exists(stack_name):
        if os.path.exists(stack_name+'.tar.gz'):
            return_code = subprocess.call('tar xfz %s.tar.gz'%stack_name, shell=True)
        elif os.path.exists(stack_name+'.tar'):
            return_code = subprocess.call('tar xf %s.tar'%stack_name, shell=True)

# if bounding boxes are not specified, compute them for all images, using the lowest level versions
os.chdir(data_dir)
for s, i, bb in zip(stack_names, image_indices, bounding_boxes):
    if bb[0] == -1: # meaning the bb is not specified
        margin = 0 # at lowest level, image dimension is 120 by 90
        fn = glob.iglob(os.path.join(s, '.*_%04d.jpg'%i))[0]
        img = cv2.imread(fn, 0)
        blurred = gaussian_filter(img, 2)
        thresholded = blurred < threshold_otsu(blurred) + 10./255.
        labelmap, n_components = label(thresholded, background=0, return_num=True)    
        component_props = regionprops(labelmap+1, cache=True)
        major_comp_prop = sorted(component_props, key=lambda x: x.area, reverse=True)[0]
        y1, x1, y2, x2 = major_comp_prop.bbox
        bb[:] = [float(y1)/img.shape[0], float(x1)/img.shape[1], 
              float(y2-y1)/img.shape[0], float(x2-x1)/img.shape[1]]
        
# create output dir
if args.out_dir is None:
    output_dir = os.path.join(data_dir, dataset_name)
else:
    output_dir = os.path.realpath(args.out_dir)
    
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# extract the bounding box at the specified reduce level
os.chdir(data_dir)
for sn, i, bb in zip(stack_names, image_indices, bounding_boxes):
    in_fn = glob.glob(os.path.join(sn, '*_%04d.jp2'%i))[0]
    out_name = '%s_%04d.tif'%(dataset_name, i)
    out_fn = os.path.join(output_dir, out_name)
    top,left,height,width = bb
    command = "kdu_expand -i '%s' -o '%s' -reduce %d -region '{%f,%f},{%f,%f}'"%(in_fn, out_fn, level, 
                                                                                 top, left, height, width)
    return_code = subprocess.call(command, shell=True)

