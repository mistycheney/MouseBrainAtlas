"""
Modify Sean's labelings on vertical RS141 slices, so that they are consistent 
with the new convention that all RS141 slices are horizontal.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.color import label2rgb, rgb2gray

import sys
import os

from joblib import Parallel, delayed
import cPickle as pickle
import subprocess
import pprint

sys.path.append(os.path.realpath('../notebooks'))
import utilities


data_dir = '/home/yuncong/BrainLocal/DavidData'
repo_dir = '/home/yuncong/BrainSaliencyDetection'

stack_name = 'RS141'
resolution = 'x5'
params_name = 'redNissl'

def exists_remote(host, path):
    return subprocess.call(['ssh', host, 'test -e ' + pipes.quote(path)]) == 0


for sli in range(25):
    slice_id = '%04d'%sli

    results_dir = os.path.join(data_dir, stack_name, resolution, slice_id, params_name, 'pipelineResults')
    labelings_dir = os.path.join(data_dir, stack_name, resolution, slice_id, params_name, 'labelings')

    if not os.path.exists(labelings_dir):
        os.makedirs(labelings_dir)


    instance_name = '_'.join([stack_name, resolution, slice_id, params_name])

    def load_array(suffix):
        return utilities.load_array(suffix, instance_name=instance_name, results_dir=results_dir)

    def save_array(arr, suffix):
        utilities.save_array(arr, suffix, instance_name=instance_name, results_dir=results_dir)

    def save_image(img, suffix):
        utilities.save_image(img, suffix, instance_name=instance_name, results_dir=results_dir, overwrite=True)

    def load_image(suffix):
        return utilities.load_image(suffix, instance_name=instance_name, results_dir=results_dir)

    img_rgb_horizontal = load_image('cropImg')
    img_rgb_vertical = np.transpose(img_rgb_horizontal, (1,0,2))[:,::-1,:]

    img_vertical = rgb2gray(img_rgb_vertical)

    sean_pipelineResults_dir_local ='/home/yuncong/BrainLocal/sean_pipelineResults/RS141_x5_%s_redNissl_pipelineResults/'%slice_id

    if not os.path.exists(sean_pipelineResults_dir_local):
        os.makedirs(sean_pipelineResults_dir_local)

    sean_pipelineResults_dir_remote = '/oasis/projects/nsf/csd181/yfreund/brain_registration/output1/s4myers/RS141_x5/redNissl/output/RS141_x5_%s_param_redNissl/' % slice_id

    segmentation_fn = 'RS141_x5_%s_param_redNissl_segmentation.npy'% slice_id
    results_file2 = os.path.join(sean_pipelineResults_dir_remote, segmentation_fn)

    if not os.path.exists(os.path.join(sean_pipelineResults_dir_local, segmentation_fn)):
        cmd = 'scp gcn:%s %s' %(results_file2, sean_pipelineResults_dir_local)
        print cmd
        subprocess.call(cmd, shell=True)

    segmentation_vertical = np.load(os.path.join(sean_pipelineResults_dir_local, segmentation_fn))

    sean_labeling_fn = '/home/yuncong/BrainLocal/sean_labelings/RS141_x5_%s_param_redNissl_labeling.pkl'%slice_id
    if not os.path.exists(sean_labeling_fn):
        continue

    labeling = pickle.load(open(sean_labeling_fn, 'r'))
    print labeling.keys()

    labellist_vertical = labeling['labellist']
    labelmap_vertical = labellist_vertical[segmentation_vertical]
    labelmap_horizontal = np.rot90(labelmap_vertical,1)

    def count_unique(keys):
        uniq_keys = np.unique(keys)
        bins = uniq_keys.searchsorted(keys)
        return uniq_keys, np.bincount(bins)

    segmentation_horizontal = load_array('segmentation')
    n_superpixels = segmentation_horizontal.max()+1
    labellist_horizontal = -1*np.ones((n_superpixels,), dtype=np.int)
    for sp in range(n_superpixels):
        in_sp_labels = labelmap_horizontal[segmentation_horizontal==sp]
        labels, counts = count_unique(in_sp_labels)
        dominant_label = int(labels[counts.argmax()])
        if dominant_label != -1:
            labellist_horizontal[sp] = dominant_label


    new_labelmap_horizontal = labellist_horizontal[segmentation_horizontal]

    hc_colors = np.loadtxt('../visualization/high_contrast_colors.txt', skiprows=1)

    new_labelmap_rgb_horizontal = label2rgb(new_labelmap_horizontal.astype(np.int), image=img_rgb_horizontal, 
                             colors=hc_colors[1:]/255., alpha=0.1, 
                             image_alpha=1, bg_color=hc_colors[0]/255.)

    import datetime
    dt = datetime.datetime.now().strftime("%m%d%Y%H%M%S")

    new_labeling = {
    'username': 'sean',
    'parent_labeling_name': None,
    'login_time': dt,
    'logout_time': dt,
    'init_labellist': None,
    'final_labellist': labellist_horizontal,
    'labelnames': labeling['names'],
    'history': None
    }

    new_labelmap_rgb_horizontal = utilities.regulate_img(new_labelmap_rgb_horizontal)
    new_preview_fn = os.path.join(labelings_dir, instance_name + '_%s_'%new_labeling['username'] + dt + '_preview.tif')
    cv2.imwrite(new_preview_fn, new_labelmap_rgb_horizontal)

    new_labeling_fn = os.path.join(labelings_dir, instance_name + '_%s_'%new_labeling['username'] + dt + '.pkl')
    pickle.dump(new_labeling, open(new_labeling_fn, 'w'))

