import os
import sys
import time

try:
    import cv2
except:
    sys.stderr.write('Cannot load cv2.\n')
from skimage.transform import rescale

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from data_manager import *

def patch_boxes_overlay_on(bg, downscale_factor, locs, patch_size, colors=None, stack=None, sec=None):
    """
    Assume bg has the specified downscale_factor.
    """

    if bg == 'original':
        bg = imread(DataManager.get_image_filepath(stack=stack, section=sec, version='compressed'))[::downscale_factor, ::downscale_factor]
       
    # viz = bg.copy()
    viz = gray2rgb(bg).copy()
    # need copy() because of this bug http://stackoverflow.com/a/31316516
    
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
        if is_invalid(fn=fn): return

    # scoremap_d = dense_score_map_lossless[::downscale, ::downscale]
    # h, w = scoremap_d.shape
    # scoremap_viz = np.ones((h, w, 4))
    # scoremap_viz[..., :3] = color
    # scoremap_n = scoremap_d/scoremap_d.max()
    # scoremap_viz[..., 3] = scoremap_n**3

    try:
        dense_score_map = DataManager.load_downscaled_scoremap(stack=stack, section=sec, fn=fn,
                            setting=setting, structure=structure, downscale=downscale)
        mask = dense_score_map > 0.
        scoremap_viz = plt.cm.hot(dense_score_map)[..., :3]
    except:
        raise Exception('Error loading scoremap of %s for image %s.' % (structure, fn))

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
        if downscale == 32:
            fp = DataManager.get_image_filepath(stack=stack, section=sec, resol='thumbnail', version='cropped')
            # download_from_s3_to_ec2(fp)
            bg = imread(fp)
        else:
            fp = DataManager.get_image_filepath(stack=stack, section=sec, resol='lossless', version='compressed')
            # download_from_s3_to_ec2(fp)
            bg = imread(fp)[::downscale, ::downscale]

    # t = time.time()
    ret = scoremap_overlay(stack=stack, sec=sec, fn=fn, structure=structure, downscale=downscale,
                            image_shape=bg.shape[:2], return_mask=True, setting=setting)
    # sys.stderr.write('scoremap_overlay: %.2f seconds.\n' % (time.time() - t))
    scoremap_viz, mask = ret

    viz = gray2rgb(bg)
    viz[mask] = (.3 * img_as_ubyte(scoremap_viz[mask, :3]) + .7 * viz[mask]).astype(np.uint8)

    # put label name at left upper corner
    if label_text is not None:
        cv2.putText(viz, label_text, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 2, ((0,0,0)), 3)

    return viz


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
