# -*- coding: utf-8 -*-

import os
import sys
import time
import cv2

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from metadata import *
from data_manager import *

from sklearn.cluster import KMeans
from sklearn.externals import joblib

import re


def compute_test_accuracy(weak_clfs, weak_clf_weights):

    H_test = np.zeros((n_test, n_classes))
    for weight, (center_index, radius, predictions, center_label) in zip(weak_clf_weights, weak_clfs):
        dist = np.minimum(test_data_normalized, train_data_normalized[center_index]).sum(axis=1)
        H_test[dist > radius, predictions] += weight

    strong_pred_test = np.argmax(H_test, axis=1)

    test_acc = np.count_nonzero(strong_pred_test == test_labels_clf) / float(n_test)
    return test_acc

def load_spm_histograms_train(stack, names, name_to_label, n_sample_per_class=1000, use_level0_only=False, secs=None, shuffle=True):

    train_data = []
    train_addresses = []
    train_labels = []

    for name in names:
        data, address = load_spm_histograms(stack, name, n_sample=n_sample_per_class,
                                            use_level0_only=use_level0_only, remove_blank_patches=True,
                                            secs=secs)
        train_data.append(data)
        train_addresses += address
        train_labels += [name_to_label[name] for _ in range(len(data))]

    train_data = np.concatenate(train_data)

    if shuffle:
        shuffled_indices = np.random.permutation(range(len(train_data)))
        train_data = train_data[shuffled_indices]
        train_addresses = [train_addresses[i] for i in shuffled_indices]
        train_labels = np.array([train_labels[i] for i in shuffled_indices])

    return train_data, train_labels, train_addresses


def load_spm_histograms(stack, name, n_sample_per_sec=None, n_sample=1000, use_level0_only=False, remove_blank_patches=True, secs=None):

    sift_dir = '/oasis/projects/nsf/csd395/yuncong/CSHL_SIFT_SPM_features'

    train = not name.startswith('roi')

    if train:
        file_dir = os.path.join(sift_dir, 'train', stack)
    else:
        file_dir = os.path.join(sift_dir, 'test', stack)

    filenames = []
    for fn in os.listdir(file_dir):
        if not fn.endswith('l0.bp'):
            continue

        res = re.findall('(%(stack)s_([0-9]{4})_%(name)s_histograms)_l0.bp' % {'stack':stack,'name':name}, fn)
        if len(res) == 0:
            continue

        prefix = res[0][0]
        sec = int(res[0][1])
        
        if secs is not None and sec not in secs:
            continue
        filenames.append((prefix, sec))

    if n_sample_per_sec is None:
        n_sample_per_sec = n_sample / len(filenames) + 1

    data = []
    addresses = []

    for prefix, sec in filenames:

        if use_level0_only:
            hists0 = bp.unpack_ndarray_file(os.path.join(file_dir, prefix + '_l0.bp'))
        else:
            hists0 = bp.unpack_ndarray_file(os.path.join(file_dir, prefix + '_l0.bp'))
            hists1 = bp.unpack_ndarray_file(os.path.join(file_dir, prefix + '_l1.bp'))
            hists2 = bp.unpack_ndarray_file(os.path.join(file_dir, prefix + '_l2.bp'))

        n = hists0.shape[0]

        random_indices = np.random.choice(range(n), min(n, n_sample_per_sec), replace=False)

        if use_level0_only:
            H = hists0[random_indices]
        else:
            H = np.c_[hists0[random_indices],
                      hists1[random_indices].reshape((len(random_indices), -1)),
                      hists2[random_indices].reshape((len(random_indices), -1))]

        data.append(H)
        addresses.append([(stack, sec, name, i) for i in random_indices])

    data = np.concatenate(data)
    addresses = list(chain(*addresses))

    if remove_blank_patches:

        blank_patch_indices = np.where(data.sum(axis=1)==0)[0]
        nonblack_patch_indices = list(set(range(len(data))) - set(blank_patch_indices))

        data = data[nonblack_patch_indices]
        addresses = [addresses[i] for i in nonblack_patch_indices]

    return data, addresses

# def compute_vocabulary():
#     '''
#     Compute or load the SIFT descriptor dictionary/vocabulary.
#     '''

#     output_dir = '/oasis/projects/nsf/csd395/yuncong/CSHL_SIFT_SPM_features/'

#     if os.path.exists(output_dir + '/vocab.pkl'):

#         # Load vocabulary
#         vocabulary = joblib.load(output_dir + '/vocab.pkl')

#     else:

#         if os.path.exists(output_dir + '/sift_descriptors_pool_arr.bp'):

#             # Load descriptor pool
#             descriptors_pool_arr = bp.unpack_ndarray_file(output_dir + '/sift_descriptors_pool_arr.bp')

#             t = time.time()

#             vocabulary = KMeans(init='random', n_clusters=M, n_init=10)
#             vocabulary.fit(descriptors_pool_arr)

#             sys.stderr.write('sift: %.2f seconds\n' % (time.time() - t)) # 300 seconds

#             cluster_centers = vocabulary.cluster_centers_

#             joblib.dump(vocabulary, output_dir + '/vocab.pkl')

#         else:

#             # Generate SIFT descriptor pool
#             descriptors_pool = []

#             sift = cv2.SIFT();

#             for sec in range(first_detect_sec, last_detect_sec+1, 10):

#                 print sec

#                 xmin, ymin, w, h = detect_bbox_lookup['MD589'] # in thumbnail resolution
#                 # convert to lossless resolution
#                 xmin = xmin * 32
#                 ymin = ymin * 32
#                 w = w * 32
#                 h = h * 32
#                 xmax = xmin + w - 1
#                 ymax = ymin + h - 1

#                 img = imread(DataManager.get_image_filepath(stack='MD589', section=sec, version='rgb-jpg'))[ymin:ymax+1, xmin:xmax+1]

#                 keypoints, descriptors = sift.detectAndCompute(img, None)

#                 n = 1000
#                 random_indices = np.random.choice(range(len(descriptors)), n, replace=False)

#                 descriptors_pool.append(descriptors[random_indices])

#             descriptors_pool_arr = np.vstack(descriptors_pool)
#             print len(descriptors_pool_arr), 'in descriptor pool'

#             bp.pack_ndarray_file(descriptors_pool_arr, output_dir + '/sift_descriptors_pool_arr.bp')

#     return vocabulary

def compute_sift_keypoints_and_descriptors(stack, section):
    
    try:
        keypoints = bp.unpack_ndarray_file(DataManager.get_sift_keypoints_filepath(stack=stack, section=section))
        descriptors = bp.unpack_ndarray_file(DataManager.get_sift_descriptors_filepath(stack=stack, section=section))
    except:
        image = DataManager.load_image_v2(stack=stack, section=section, prep_id=2, version='gray')
        mask = DataManager.load_image_v2(stack=stack, section=section, prep_id=2, version='mask', resol='thumbnail')
        xmin_tb, xmax_tb, ymin_tb, ymax_tb = bbox_2d(mask)
        xmin = xmin_tb * 32
        xmax = (xmax_tb + 1) * 32
        ymin = ymin_tb * 32
        ymax = (ymax_tb + 1) * 32

        sift = cv2.xfeatures2d.SIFT_create()
        img = image[ymin:ymax+1, xmin:xmax+1].copy()
        keypoints, descriptors = sift.detectAndCompute(img, None);
        keypoints = np.array([k.pt for k in keypoints]) + (xmin, ymin)
        
    return keypoints, descriptors

def compute_vocabulary(stacks=['MD589'], every_num_sec=10, n_per_sec=1000, M=200):
    '''
    Compute or load the SIFT descriptor dictionary/vocabulary.
    '''

    kmeans_fp = DataManager.get_sift_descriptor_vocabulary_filepath()

    if os.path.exists(kmeans_fp):
        # Load vocabulary
        vocabulary_kmeans = joblib.load(kmeans_fp)
    else:
        # Generate SIFT descriptor pool
        descriptors_pool = []
        for stack in stacks:
            for sec in metadata_cache['valid_sections'][stack][::every_num_sec]:
                _, descriptors = compute_sift_keypoints_and_descriptors(stack, section=sec)
                random_indices = np.random.choice(range(len(descriptors)), n_per_sec, replace=False)
                descriptors_pool.append(descriptors[random_indices])
        descriptors_pool_arr = np.concatenate(descriptors_pool)
        print len(descriptors_pool_arr), 'in descriptor pool'

        # bp.pack_ndarray_file(descriptors_pool_arr, output_dir + '/sift_descriptors_pool_arr.bp')

        vocabulary_kmeans = KMeans(init='random', n_clusters=M, n_init=10)
        vocabulary_kmeans.fit(descriptors_pool)
        # cluster_centers = vocabulary.cluster_centers_
            
    return vocabulary_kmeans


def compute_labelmap(stack, section=None, fn=None, force=False):
    """
    Compute SIFT descriptor word map.
    """
    
    labelmap_fp = DataManager.get_sift_descriptors_labelmap_filepath(stack, section=section, fn=fn)

    if os.path.exists(labelmap_fp) and not force:
        # Load labelmap
        labelmap = bp.unpack_ndarray_file(labelmap_fp)
    else:
        keypoints, descriptors = compute_sift_keypoints_and_descriptors(stack=stack, section=section)

        # Compute keypoints and assign labels
        print len(keypoints), 'keypoints' # ~ 500k

        t = time.time()
        vocabulary = compute_vocabulary()
        sys.stderr.write('compute vocab: %.2f seconds\n' % (time.time() - t))  # ~ 20 s

        t = time.time()
        keypoint_labels = vocabulary.predict(descriptors)
        sys.stderr.write('predict: %.2f seconds\n' % (time.time() - t))  # ~ 20 s

        # visualize keypoints (color indicating label)

        # viz = img.copy()
        # for (x, y), l in zip(keypoints_arr, keypoint_labels):
        #     cv2.circle(viz, (int(x), int(y)), 3, colors[l], -1)
        # display_image(viz)

        # Generate labelmap

        w, h = metadata_cache['image_shape'][stack]
        labelmap = np.zeros((h, w), np.int)
        keypoints_arr_int = keypoints.astype(np.int)
        labelmap[keypoints_arr_int[:,1], keypoints_arr_int[:,0]] = keypoint_labels + 1

    return labelmap



# from joblib import Parallel, delayed
from multiprocess import Pool
labelmap_global = None

def compute_spm_histograms(labelmap, sample_locs, patch_size, M):
    """
    Args:
        labelmap (2d-ndarray of int):
        sample_locs (2d-ndarray): List of (x,y) locations at which to sample the SPM histograms
        M (int): number of unique SIFT descriptor words, aka. size of vocabulary
        
    Returns:
        hists_arr0 ((1,M)-array of int)
        hists_arr1 ((4,M)-array of int)
        hists_arr2 ((16,M)-array of int)
    """

    global labelmap_global
    labelmap_global = labelmap

    # compute level-2 histograms
    l = 2

    grid_size = patch_size / 2**l

    if l == 2:
        rx = [-2, -1, 0, 1]
        ry = [-2, -1, 0, 1]
    elif l == 1:
        rx = [-1, 0]
        ry = [-1, 0]
    elif l == 0:
        rx = [-.5]
        ry = [-.5]

    rxs, rys = np.meshgrid(rx, ry)

    patch_coords_allGrid = []

    for grid_i, (rx, ry) in enumerate(np.c_[rxs.flat, rys.flat]):

        patch_xmin = sample_locs[:,0] + rx * grid_size
        patch_ymin = sample_locs[:,1] + ry * grid_size
        patch_xmax = sample_locs[:,0] + (rx + 1) * grid_size
        patch_ymax = sample_locs[:,1] + (ry + 1) * grid_size

        patch_coords_allGrid.append([patch_xmin, patch_ymin, patch_xmax, patch_ymax])


    all_coords = np.hstack(patch_coords_allGrid)
    patch_xmin = all_coords[0]
    patch_ymin = all_coords[1]
    patch_xmax = all_coords[2]
    patch_ymax = all_coords[3]

    def compute_histogram_particular_label(i):
        m = (labelmap_global == i).astype(np.uint8)
        mi = cv2.integral(m)
        ci = mi[patch_ymin, patch_xmin] + mi[patch_ymax, patch_xmax] - mi[patch_ymax, patch_xmin] - mi[patch_ymin, patch_xmax]
        return ci

    t = time.time()
    # hists = Parallel(n_jobs=16)(delayed(compute_histogram_particular_label)(i) for i in range(1, M+1))
    # hists = Parallel(n_jobs=8)(delayed(compute_histogram_particular_label)(i) for i in range(1, M+1))
    pool = Pool(8)
    hists = pool.map(compute_histogram_particular_label, range(1, M+1))
    # pool.terminate()
    pool.close()
    pool.join()
    # del pool
    sys.stderr.write('done in %f seconds\n' % (time.time() - t)) # ~ 13 seconds

    n_grid = (2**l)**2
    hists_arr2 = np.transpose(np.reshape(hists, (M, n_grid, -1)))
    print hists_arr2.shape

    # compute level-1 histograms based on level-2 histograms

    hists_arr1 = np.transpose([hists_arr2[:, [0,1,4,5], :].sum(axis=1),
                               hists_arr2[:, [2,3,6,7], :].sum(axis=1),
                               hists_arr2[:, [8,9,12,13], :].sum(axis=1),
                               hists_arr2[:, [10,11,14,15], :].sum(axis=1)],
                              [1,0,2])
    print hists_arr1.shape

    # compute level-0 histograms based on level-1 histograms

    hists_arr0 = hists_arr1.sum(axis=1)
    print hists_arr0.shape

    return hists_arr0, hists_arr1, hists_arr2
