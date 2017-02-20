import cv2

import os
import sys
sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from data_manager import *

from joblib import Parallel, delayed
import time

from skimage.transform import rescale

def patch_boxes_overlay_on(bg, downscale_factor, locs, patch_size, colors=None, stack=None, sec=None):
    """
    Assume bg has the specified downscale_factor.
    """

    if bg == 'original':
        bg = imread(DataManager.get_image_filepath(stack=stack, section=sec, version='compressed'))[::downscale_factor, ::downscale_factor]

    # viz = bg.copy()
    viz = gray2rgb(bg)
    half_size = patch_size/2/downscale_factor
    if isinstance(locs[0], list):
        if colors is None:
            colors = random_colors(len(locs))
        for i, locs_oneColor in enumerate(locs):
            for x, y in locs_oneColor:
                x = x / downscale_factor
                y = y / downscale_factor
                cv2.rectangle(viz, (x-half_size, y-half_size), (x+half_size, y+half_size), colors[i], 2)
    else:
        if colors is None:
            colors = (255,0,0)
        for x, y in locs:
            x = x / downscale_factor
            y = y / downscale_factor
            cv2.rectangle(viz, (x-half_size, y-half_size), (x+half_size, y+half_size), colors, 2)

    return viz


# def scoremap_overlay(stack, sec, name, downscale_factor, image_shape=None, return_mask=True):
#     '''
#     Generate scoremap of structure.
#     name: name
#     '''
#
#     if image_shape is None:
#         image_shape = DataManager(stack=stack).get_image_dimension()
#
#     scoremap_bp_filepath = scoremaps_rootdir + '/%(stack)s/%(slice)04d/%(stack)s_%(slice)04d_roi1_denseScoreMapLossless_%(label)s.hdf' \
#         % {'stack': stack, 'slice': sec, 'label': name}
#
#     if not os.path.exists(scoremap_bp_filepath):
#         sys.stderr.write('No scoremap for %s for section %d\n' % (name, sec))
#         return
#
#     scoremap = load_hdf(scoremap_bp_filepath)
#
#     interpolation_xmin, interpolation_xmax, \
#     interpolation_ymin, interpolation_ymax = np.loadtxt(os.path.join(scoremaps_rootdir,
#                  '%(stack)s/%(sec)04d/%(stack)s_%(sec)04d_roi1_denseScoreMapLossless_%(label)s_interpBox.txt' % \
#                                                      {'stack': stack, 'sec': sec, 'label': name})).astype(np.int)
#
#     dense_score_map_lossless = np.zeros(image_shape)
#     dense_score_map_lossless[interpolation_ymin:interpolation_ymax+1,
#                              interpolation_xmin:interpolation_xmax+1] = scoremap
#
#     mask = dense_score_map_lossless > 0.
#
#     scoremap_viz = plt.cm.hot(dense_score_map_lossless[::downscale_factor, ::downscale_factor])[..., :3]
#     viz = img_as_ubyte(scoremap_viz)
#
#     if return_mask:
#         return viz, mask
#     else:
#         return viz

def scoremap_overlay(stack, structure, downscale, setting,
                    image_shape=None, return_mask=False, sec=None, fn=None,
                    color=(1,0,0)):
    '''
    Generate scoremap image of structure.
    name: structure name
    '''

    if fn is None:
        assert sec is not None
        fn = metadata_cache['sections_to_filenames'][stack][sec]
        if is_invalid(fn): return

    if image_shape is None:
        image_shape = metadata_cache['image_shape'][stack][::-1]

    try:
        dense_score_map_lossless = DataManager.load_scoremap(stack=stack, section=sec, fn=fn, setting=setting, structure=structure,
                                downscale=1)
    except:
        raise Exception('Error loading scoremap of %s for image %s.' % (structure, fn))

    # scoremap_bp_filepath, scoremap_interpBox_filepath = \
    # DataManager.get_scoremap_filepath(stack=stack, section=sec, fn=fn, setting=setting,
    #                                     structure=structure, return_bbox_fp=True)
    #
    # if not os.path.exists(scoremap_bp_filepath):
    #     # sys.stderr.write('No scoremap of %s for filename %s.\n' % (structure, fn))
    #     raise Exception('No scoremap of %s for filename %s.' % (structure, fn))
    #
    # scoremap = load_hdf(scoremap_bp_filepath)
    #
    # interpolation_xmin, interpolation_xmax, \
    # interpolation_ymin, interpolation_ymax = np.loadtxt(scoremap_interpBox_filepath).astype(np.int)
    #
    # dense_score_map_lossless = np.zeros(image_shape)
    # dense_score_map_lossless[interpolation_ymin:interpolation_ymax+1,
    #                          interpolation_xmin:interpolation_xmax+1] = scoremap

    mask = dense_score_map_lossless > 0.

    # scoremap_viz = plt.cm.hot(dense_score_map_lossless[::downscale, ::downscale])[..., :3]

    scoremap_d = dense_score_map_lossless[::downscale, ::downscale]
    h, w = scoremap_d.shape
    scoremap_viz = np.ones((h, w, 4))
    scoremap_viz[..., :3] = color
    scoremap_n = scoremap_d/scoremap_d.max()
    scoremap_viz[..., 3] = scoremap_n**3

    viz = img_as_ubyte(scoremap_viz)

    if return_mask:
        return viz, mask
    else:
        return viz

def scoremap_overlay_on(bg, stack, structure, downscale, setting, label_text=None, sec=None, fn=None):

    if fn is None:
        assert sec is not None
        fn = metadata_cache['sections_to_filenames'][stack][sec]
        if is_invalid(fn): return

    if bg == 'original':
        bg = imread(DataManager.get_image_filepath(stack=stack, section=sec, resol='lossless', version='compressed'))

    # t = time.time()
    ret = scoremap_overlay(stack=stack, sec=sec, fn=fn, structure=structure, downscale=downscale,
                            image_shape=bg.shape[:2], return_mask=True, setting=setting)
    # sys.stderr.write('scoremap_overlay: %.2f seconds.\n' % (time.time() - t))

    scoremap_viz, mask = ret

    m = mask[::downscale, ::downscale]

    viz = bg[::downscale, ::downscale].copy()
    viz = gray2rgb(viz)

    viz[m] = (.3 * img_as_ubyte(scoremap_viz[m, :3]) + .7 * viz[m]).astype(np.uint8)

    # put label name at left upper corner
    if label_text is not None:
        cv2.putText(viz, label_text, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 2, ((0,0,0)), 3)

    return viz

# def export_scoremaps(bg, stack, structures, downscale, setting, label_text=True, sec=None, fn=None):
#
#     if bg == 'original':
#         bg = imread(DataManager.get_image_filepath(stack=stack, fn=fn, resol='lossless',version='compressed'))
#
#     for structure in structures:
#         try:
#             export_filepath = DataManager.get_scoremap_viz_filepath(stack=stack, sec=sec, fn=fn, structure=structure, setting=setting)
#             scoremap_overlay_on(bg=bg, stack=stack, sec=sec, fn=fn, structure=structure, downscale=downscale, label_text=label_text, setting=setting)
#         except:
#             sys.stderr.write('Scoremap for %s does not exist.\n' % name)

# def export_scoremaps_multiprocess(bg, stack, structures, downscale_factor, setting,
#                     label_text=True, sections=None, filenames=None):
#
#     if filenames is None:
#         s2f = metadata_cache['sections_to_filenames'][stack]
#         filenames = [s2f[sec] for sec in sections]
#         filenames = [fn for fn in filenames if is_invalid(fn)]
#
#     t = time.time()
#
#     Parallel(n_jobs=8)(delayed(export_scoremaps)(bg=bg, stack=stack, structures=structures,
#                                                         downscale_factor=downscale_factor,
#                                                         label_text=label_text, filename=fn,
#                                                         setting=setting)
#                        for fn in filenames)
#
#     print time.time() - t

#     for sec in sections:
#         if bg == 'original':
#             img = imread(DataManager.get_image_filepath(stack=stack, section=sec, version='rgb-jpg'))

#         for name in names:
#             export_filepath = export_filepath_fmt % {'stack': stack, 'sec': sec, 'name': name}
#             scoremap_overlay_on(img, stack, sec, name, downscale_factor, export_filepath, label_text)


def export_scoremapPlusAnnotationVizs_worker(bg, stack, names, downscale_factor, users,
                                            export_filepath_fmt=None,
                                            sec=None, filename=None):

    if bg == 'original':
        if sec is not None:
            bg = imread(DataManager.get_image_filepath(stack=stack, section=sec, version='rgb-jpg'))
        if filename is not None:
            bg = imread(DataManager.get_image_filepath(stack=stack, fn=filename, version='rgb-jpg'))

    for name in names:
        scoremap_overlaid = scoremap_overlay_on(bg, stack, name, downscale_factor, label_text=True,
                                                sec=sec, fn=filename)
        if scoremap_overlaid is not None:
            annotation_overlaid = annotation_overlay_on(scoremap_overlaid, stack, [name], 8,
                                               export_filepath_fmt=export_filepath_fmt, users=users,
                                               sec=sec, fn=filename)

def export_scoremapPlusAnnotationVizs(bg, stack, names, downscale_factor, sections=None, filenames=None, users=None, export_filepath_fmt=None):

    if filenames is None:
        filenames_to_sections, _ = DataManager.load_sorted_filenames(stack)
        filenames = [filenames_to_sections[sec] for sec in sections]
        filenames = [fn for fn in filenames if fn not in ['Nonexisting', 'Placeholder', 'Rescan']]

    Parallel(n_jobs=4)(delayed(export_scoremapPlusAnnotationVizs_worker)(bg, stack, names, downscale_factor,
                                filename=fn,
                                export_filepath_fmt=export_filepath_fmt, users=users)
                                for fn in filenames)

#     for sec in sections:
#         export_scoremapPlusAnnotationVizs_worker(bg, stack, sec, names, downscale_factor,
#                                                         export_filepath_fmt=export_filepath_fmt)


def annotation_v3_overlay_on(bg, stack, orientation=None,
                            structure_names=None, downscale_factor=8,
                            users=None, colors=None, show_labels=True, export_filepath_fmt=None,
                            annotation_rootdir=None):
    """
    Works with annotation files version 3.
    It is user's responsibility to ensure bg is the same downscaling as the labeling.
    """

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

    #####################################
    paired_structures = ['5N', '6N', '7N', '7n', 'Amb', 'LC', 'LRt', 'Pn', 'Tz', 'VLL', 'RMC', 'SNC', 'SNR', '3N', '4N',
                    'Sp5I', 'Sp5O', 'Sp5C', 'PBG', '10N', 'VCA', 'VCP', 'DC']
    singular_structures = ['AP', '12N', 'RtTg', 'SC', 'IC']
    structures = paired_structures + singular_structures

    structure_colors = {n: np.random.randint(0, 255, (3,)) for n in structures}

    #######################################

    first_sec, last_sec = metadata_cache['section_limits'][stack]

    bar = show_progress_bar(first_sec, last_sec)

    # for section in [270]:
    for section in range(first_sec, last_sec+1):

        t = time.time()

        bar.value = section

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


# def annotation_v2_overlay_on(bg, stack, section=None, index=None, orientation=None,
#                             structure_names=None, downscale_factor=8,
#                             users=None, colors=None, show_labels=True, export_filepath_fmt=None,
#                             annotation_rootdir=None):
#     """
#     Works with annotation files version 2.
#     It is user's responsibility to ensure bg is the same downscaling as the labeling.
#
#     image identifier: MD589_sagittal_downsample8_indexOrSection
#     annotation identifier: MD589_sgittal_downsample8_username_timestamp
#     """
#
#     annotations = {}
#     timestamps = {}
#
#     if users is None:
#         users = ['yuncong', 'localAdjusted', 'autoAnnotate']
#
#     if colors is None:
#         colors = [(0,0,255), (255,0,0), (0,255,0), (0, 255, 255)]
#
#     for user in users:
#         try:
#             labeling, _, _ = DataManager.load_annotation_v2(stack=stack, username=user, orientation=orientation, downsample=1, annotation_rootdir=annotation_rootdir)
#         except:
#             sys.stderr.write('Cannot read labeling for %s\n' % user)
#             continue
#
#         if labeling['indexing_scheme'] == 'index':
#             assert index is not None
#             annotations[user] = labeling['polygons'][index]
#         elif labeling['indexing_scheme'] == 'section':
#             assert section is not None
#             annotations[user] = labeling['polygons'][section]
#         timestamps[user] = labeling['timestamp']
#
#     if len(annotations) == 0:
#         return
#
#     if bg == 'original':
#         assert section is not None
#         img = imread(DataManager.get_image_filepath(stack=stack, section=section, version='rgb-jpg'))
#         viz = img[::downscale_factor, ::downscale_factor].copy()
#     else:
#         viz = bg.copy()
#
#     added_labels = set([])
#
#     for user, anns in annotations.iteritems():
#         for ann in anns:
#
#             if structure_names is not None and ann['label'] not in structure_names:
#                 continue
#
#             vertices = np.array(ann['vertices']) / downscale_factor
#
#         #     for x,y in vertices:
#         #         cv2.circle(viz, (int(x), int(y)), 5, (255,0,0), -1)
#             cv2.polylines(viz, [vertices.astype(np.int)], True, colors[users.index(user)], 2)
#
#             if show_labels:
#                 if ann['label'] not in added_labels:
#                     lx, ly = np.array(ann['labelPos']) / downscale_factor
#                     cv2.putText(viz, ann['label'], (int(lx)-10, int(ly)+10),
#                                 cv2.FONT_HERSHEY_DUPLEX, 1, ((0,0,0)), 2)
#
#                     added_labels.add(ann['label'])
#
#
#     if export_filepath_fmt is not None:
#         if structure_names is not None:
#             if len(structure_names) == 1:
#                 assert section is not None
#                 export_filepath = export_filepath_fmt % {'stack': stack, 'sec': section, 'name': structure_names[0],
#                                                      'annofn': '_'.join(usr+'_'+ts for usr, ts in timestamps.iteritems())}
#         else:
#             assert section is not None
#             export_filepath = export_filepath_fmt % {'stack': stack, 'sec': section,
#                                                  'annofn': '_'.join(usr+'_'+ts for usr, ts in timestamps.iteritems())}
#
#         create_if_not_exists(os.path.dirname(export_filepath))
#         cv2.imwrite(export_filepath, viz[..., ::-1])
#
#     return viz


# def annotation_overlay_on(bg, stack, section, structure_names=None, downscale_factor=8,
#                           users=None, colors=None, show_labels=True,
#                          export_filepath_fmt=None, annotation_rootdir=None):
#     """
#     export_filepath_fmt should include stack, sec, name, annofn as arguments.
#     annofn is a concatenation of username-timestamp tuples joined by hyphens.
#     """
#     annotations = {}
#     timestamps = {}
#
#     if annotation_rootdir is None:
#         annotation_rootdir = annotation_midbrainIncluded_rootdir
#
#     if users is None:
#         users = ['yuncong', 'localAdjusted', 'autoAnnotate']
#
#     if colors is None:
#         colors = [(0,0,255), (255,0,0), (0,255,0), (0, 255, 255)]
#
#     for user in users:
#         ret = DataManager.load_annotation(stack=stack, section=section, username=user, annotation_rootdir=annotation_rootdir)
#         # ret = load_annotation()
#         if ret is not None:
#             annotations[user] = ret[0]
#             timestamps[user] = ret[2]
#
#     if len(annotations) == 0:
#         return
#
#     if bg == 'original':
#         img = imread(DataManager.get_image_filepath(stack=stack, section=section, version='rgb-jpg'))
#         viz = img[::downscale_factor, ::downscale_factor].copy()
#     else:
#         viz = bg.copy()
#
#     added_labels = set([])
#
#     for user, anns in annotations.iteritems():
#         for ann in anns:
#
#             if structure_names is not None and ann['label'] not in structure_names:
#                 continue
#
#             vertices = np.array(ann['vertices']) / downscale_factor
#
#         #     for x,y in vertices:
#         #         cv2.circle(viz, (int(x), int(y)), 5, (255,0,0), -1)
#             cv2.polylines(viz, [vertices.astype(np.int)], True, colors[users.index(user)], 2)
#
#             if show_labels:
#
#                 if ann['label'] not in added_labels:
#                     lx, ly = np.array(ann['labelPos']) / downscale_factor
#                     cv2.putText(viz, ann['label'], (int(lx)-10, int(ly)+10),
#                                 cv2.FONT_HERSHEY_DUPLEX, 1, ((0,0,0)), 2)
#
#                     added_labels.add(ann['label'])
#
#
#     if export_filepath_fmt is not None:
#         if structure_names is not None:
#             if len(structure_names) == 1:
#                 export_filepath = export_filepath_fmt % {'stack': stack, 'sec': section, 'name': structure_names[0],
#                                                      'annofn': '_'.join(usr+'_'+ts for usr, ts in timestamps.iteritems())}
#         else:
#             export_filepath = export_filepath_fmt % {'stack': stack, 'sec': section,
#                                                  'annofn': '_'.join(usr+'_'+ts for usr, ts in timestamps.iteritems())}
#
#         create_if_not_exists(os.path.dirname(export_filepath))
#         cv2.imwrite(export_filepath, viz[..., ::-1])
#
#     return viz


# def load_annotation(stack, section, username=None, timestamp='latest', path_only=False):
#
#     import cPickle as pickle
#     from itertools import chain
#
#     try:
#         if timestamp == 'latest':
#             if username is None:
#                 annotation_names = list(chain.from_iterable(labeling_list[stack][section].dropna().__iter__()))
#                 annotation_name = sorted(annotation_names)[-1]
#             else:
#                 annotation_name = sorted(labeling_list[stack][section][username])[-1]
#
#             _, _, _, timestamp, _ = annotation_name.split('_')
#         else:
#             sys.stderr.write('Timestamp is not latest, not implemented.\n')
#             return
#
#     except Exception as e:
#         sys.stderr.write('Annotation does not exist: %s, %d, %s, %s\n' % (stack, section, username, timestamp))
#         return
#
#     annotation_filepath = os.path.join(os.environ['LABELING_DIR'], stack, '%04d'%section, annotation_name)
#
#     if path_only:
#         return annotation_filepath
#     else:
#         return pickle.load(open(annotation_filepath, 'r')), timestamp


# def get_labeling_list():
#
#     from collections import defaultdict
#     import pandas as pd
#
#     labeling_list = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
#     for stack in os.listdir(os.environ['LABELING_DIR']):
#         if os.path.isdir(os.path.join(os.environ['LABELING_DIR'], stack)):
#             for sec in sorted(os.listdir(os.path.join(os.environ['LABELING_DIR'], stack))):
#                 for labeling_fn in os.listdir(os.path.join(os.environ['LABELING_DIR'], stack, sec)):
#                     user = labeling_fn.split('_')[2]
#                     labeling_list[stack][int(sec)][user].append(labeling_fn)
#
#     labeling_list.default_factory = None
#
#     reformed = {(stack, sec): labeling_list[stack][sec] for stack, secs in labeling_list.iteritems() for sec in secs }
#     df = pd.DataFrame(reformed)
#
#     return df
#
# labeling_list = get_labeling_list()
