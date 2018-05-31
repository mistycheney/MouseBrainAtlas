#! /usr/bin/env python

import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='visualize annotations, version 3')
parser.add_argument("stack_name", type=str, help="stack name")
# parser.add_argument("section", type=int, help="section index")
parser.add_argument("first_sec", type=int, help="section index")
parser.add_argument("last_sec", type=int, help="section index")
args = parser.parse_args()

######################################

import os
import sys
sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from visualization_utilities import *
from metadata import *

#####################################

# stack = args.stack_name
# sec = args.section

stack = args.stack_name
first_sec = args.first_sec
last_sec = args.last_sec

######################################

# viz_dir = create_if_not_exists(annotation_midbrainIncluded_v2_rootdir + '/viz')
#
# _ = annotation_v2_overlay_on('original', stack=stack, section=sec, users=['yuncong'], downscale_factor=8,
#                     annotation_rootdir=annotation_midbrainIncluded_v2_rootdir,
#                     export_filepath_fmt=os.path.join(viz_dir, stack, '%(stack)s_%(sec)04d_%(annofn)s.jpg'))

########################################

contour_df, _ = DataManager.load_annotation_v3(stack=stack, annotation_rootdir=annotation_midbrainIncluded_v2_rootdir)

downsample_factor = 8

anchor_filename = metadata_cache['anchor_fn'][stack]
sections_to_filenames = metadata_cache['sections_to_filenames'][stack]
filenames_to_sections = {f: s for s, f in sections_to_filenames.iteritems()
                        if f not in ['Placeholder', 'Nonexisting', 'Rescan']}

# Load transforms, defined on thumbnails
import cPickle as pickle
Ts = pickle.load(open(thumbnail_data_dir + '/%(stack)s/%(stack)s_elastix_output/%(stack)s_transformsTo_anchor.pkl' % dict(stack=stack), 'r'))

Ts_inv_downsampled = {}
for fn, T0 in Ts.iteritems():
    T = T0.copy()
    T[:2, 2] = T[:2, 2] * 32 / downsample_factor
    Tinv = np.linalg.inv(T)
    Ts_inv_downsampled[fn] = Tinv

# Load bounds
crop_xmin, crop_xmax, crop_ymin, crop_ymax = metadata_cache['cropbox'][stack]
print 'crop:', crop_xmin, crop_xmax, crop_ymin, crop_ymax

#######################################

paired_structures = ['5N', '6N', '7N', '7n', 'Amb', 'LC', 'LRt', 'Pn', 'Tz', 'VLL', 'RMC', 'SNC', 'SNR', '3N', '4N',
                    'Sp5I', 'Sp5O', 'Sp5C', 'PBG', '10N', 'VCA', 'VCP', 'DC']
singular_structures = ['AP', '12N', 'RtTg', 'SC', 'IC']
structures = paired_structures + singular_structures

structure_colors = {n: np.random.randint(0, 255, (3,)) for n in structures}

#########################################

viz_dir = create_if_not_exists(os.path.join(annotation_midbrainIncluded_v2_rootdir, 'viz', stack))

for section in range(first_sec, last_sec+1):

    t = time.time()

    fn = sections_to_filenames[section]
    if fn in ['Nonexisting', 'Rescan', 'Placeholder']:
        continue

    img = imread(DataManager.get_image_filepath(stack, fn=fn, resol='lossless', version='compressed'))
    viz = img[::downsample_factor, ::downsample_factor].copy()

    for name_u in structures:
        matched_contours = contour_df[(contour_df['name'] == name_u) & (contour_df['filename'] == fn)]
        for cnt_id, cnt in matched_contours.iterrows():
            n = len(cnt['vertices'])

            # Transform points
            vertices_on_aligned = np.dot(Ts_inv_downsampled[fn], np.c_[cnt['vertices']/downsample_factor, np.ones((n,))].T).T[:, :2]

            xs = vertices_on_aligned[:,0] - crop_xmin * 32 / downsample_factor
            ys = vertices_on_aligned[:,1] - crop_ymin * 32 / downsample_factor

            vertices_on_aligned_cropped = np.c_[xs, ys].astype(np.int)

            cv2.polylines(viz, [vertices_on_aligned_cropped], True, structure_colors[name_u], 2)

    sys.stderr.write('Overlay visualize: %.2f seconds\n' % (time.time() - t)) # 6 seconds

    viz_fn = os.path.join(viz_dir, '%(fn)s_annotation.jpg' % dict(fn=fn))
    imsave(viz_fn, viz)
