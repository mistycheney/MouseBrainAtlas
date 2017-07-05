import os
import sys
import time

try:
    import cv2
except:
    sys.stderr.write('Cannot load cv2.\n')
from skimage.transform import rescale, resize

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from data_manager import *
from annotation_utilities import *
from registration_utilities import *

def patch_boxes_overlay_on(bg, downscale_factor, locs, patch_size, colors=None, stack=None, sec=None, img_version='compressed'):
    """
    Assume bg has the specified downscale_factor.
    """

    if bg == 'original':
        bg = DataManager.load_image(stack=stack, section=sec, version=img_version)[::downscale_factor, ::downscale_factor]
       
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

def generate_scoremap_layer(stack, structure, downscale, detector_id,
                    image_shape=None, return_mask=False, sec=None, fn=None,
                    color=(1,0,0), show_above=.01, cmap_name='jet'):
    '''
    Generate scoremap layer.
    Output are rescaled from down32 score maps.
    
    Args:
        structure: structure name
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
                            detector_id=detector_id, structure=structure, downscale=32)
        dense_score_map = np.minimum(rescale(dense_score_map, 32/float(downscale)), 1.)
        mask = dense_score_map > show_above
        # scoremap_viz = plt.cm.hot(dense_score_map)[..., :3]
        cmap = plt.get_cmap(cmap_name)
        scoremap_viz = cmap(dense_score_map)[..., :3]
    except Exception as e:
        raise Exception('Error loading scoremap of %s for image %s: %s\n' % (structure, fn, e))

    viz = img_as_ubyte(scoremap_viz)

    if return_mask:
        return viz, mask
    else:
        return viz
    
def scoremap_overlay_on(bg, stack, structure, out_downscale, detector_id, label_text=None, sec=None, fn=None, in_downscale=None, overlay_alpha=.3, image_version=None, show_above=.01, cmap_name='jet', overlay_bbox=None):

    if fn is None:
        assert sec is not None
        fn = metadata_cache['sections_to_filenames'][stack][sec]
        if is_invalid(fn): return

    if bg == 'original':
        if image_version is None:            
            classifier_properties = classifier_settings.loc[classifier_id]
            image_version = classifier_properties['input_img_version']

        if out_downscale == 32:
            bg = DataManager.load_image(stack=stack, section=sec, fn=fn, resol='thumbnail', version='cropped')
        else:
            im = DataManager.load_image(stack=stack, section=sec, fn=fn, resol='lossless', version=image_version)
            bg = im[::out_downscale, ::out_downscale]
    else:
        assert in_downscale is not None, "For user-given background image, its resolution `in_downscale` must be given."
        bg = rescale(bg, float(in_downscale)/out_downscale)

    # t = time.time()
    ret = generate_scoremap_layer(stack=stack, sec=sec, fn=fn, structure=structure, downscale=out_downscale,
                            image_shape=bg.shape[:2], return_mask=True, detector_id=detector_id, show_above=show_above,
                                 cmap_name=cmap_name)
    # sys.stderr.write('scoremap_overlay: %.2f seconds.\n' % (time.time() - t))
    scoremap_viz, mask = ret
    
    if overlay_bbox is not None:
        xmin,xmax,ymin,ymax = overlay_bbox
        scoremap_viz = scoremap_viz[ymin/out_downscale:(ymax+1)/out_downscale, xmin/out_downscale:(xmax+1)/out_downscale]
        mask = mask[ymin/out_downscale:(ymax+1)/out_downscale, xmin/out_downscale:(xmax+1)/out_downscale]

    viz = img_as_ubyte(gray2rgb(bg))
    viz[mask] = (overlay_alpha * scoremap_viz[mask, :3] + (1-overlay_alpha) * viz[mask]).astype(np.uint8)

    # Put label name at left upper corner.
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


def annotation_from_warped_atlas_overlay_on(bg, warped_volumes, volume_origin, stack_fixed, volume_downsample=32, 
                                            fn=None, sec=None, orientation='sagittal',
                            structures=None, out_downsample=32,
                            users=None, level_colors=None, levels=None, show_text=True,
                             contours=None, contour_width=1):
    """
    Args:
        levels (list of float): probability levels at which the contours are drawn.
        level_colors (dict {float: (3,)-ndarray of float}): contour color for each level
        volume_origin ((3,)-ndarray of float): the origin of the given volumes.
    """
    
    if level_colors is None:
        level_colors = {0.1: (0,255,255), 
                    0.25: (0,255,0), 
                    0.5: (255,0,0), 
                    0.75: (255,255,0), 
                    0.99: (255,0,255)}
    
    if levels is None:
        levels = level_colors.keys()
    
    t = time.time()
    
    xmin_vol_f, ymin_vol_f, zmin_vol_f = volume_origin
    
    if bg == 'original':
        if out_downsample == 32:
            bg = DataManager.load_image(stack=stack_fixed, section=sec, resol='thumbnail', version='cropped_tif')
        else:
            bg = DataManager.load_image(stack=stack_fixed, section=sec, resol='lossless', version='cropped_gray_jpeg')
    #         img = resize(img, np.array(metadata_cache['image_shape'][stack_fixed][::-1])/downsample_factor)
            bg = bg[::out_downsample, ::out_downsample]
    
    if bg.ndim == 2:
        bg = gray2rgb(bg)
        
    viz = bg.copy()
    
    assert orientation == 'sagittal'
    
    z = int(np.mean(DataManager.convert_section_to_z(stack=stack_fixed, sec=sec, downsample=volume_downsample)))
        
    # Find moving volume annotation contours.
    for name_s, vol in warped_volumes.iteritems():
        
        label_pos = None
        
        for level in levels:
            cnts = find_contours(vol[..., z], level=level) # rows, cols
            for cnt in cnts:
                # r,c to x,y
                cnt_on_cropped_volRes = cnt[:,::-1] + (xmin_vol_f, ymin_vol_f)
                cnt_on_cropped_imgRes = cnt_on_cropped_volRes * volume_downsample / out_downsample
                cv2.polylines(viz, [cnt_on_cropped_imgRes.astype(np.int)], 
                              True, level_colors[level], contour_width)
                
                if show_text:
                    if label_pos is None:
                        label_pos = np.mean(cnt_on_cropped_imgRes, axis=0)
    
        # Show text 
        if label_pos is not None:
            cv2.putText(viz, name_s, tuple(label_pos.astype(np.int)), 
                    cv2.FONT_HERSHEY_DUPLEX, 1, ((0,0,0)), 3)

    return viz
            

def annotation_by_human_overlay_on(bg, stack=None, fn=None, sec=None, orientation='sagittal',
                            structures=None, out_downsample=8,
                            users=None, colors=None, show_labels=True, 
                             contours=None):
    """
    Draw annotation contours on a user-given background image.
    
    Args:
        out_downsample (int): downsample factor of the output images.
        structures (list of str): list of structure names (unsided) to show.
        contours (pandas.DataFrame): rows are polygon indices.
        colors (dict {name_u: (3,)-ndarray }): contour color of each structure (unsided)
    """
    
    if fn is None:
        assert sec is not None
        fn = metadata_cache['sections_to_filenames'][stack][sec]
        if is_invalid(fn): 
            return
    
    if contours is None:    
        contours_df = DataManager.load_annotation_v4(stack=stack)
        contours = contours_df[(contours_df['orientation'] == orientation) & (contours_df['downsample'] == 1)]
        contours = contours.drop_duplicates(subset=['section', 'name', 'side', 'filename', 'downsample', 'creator'])
        contours = convert_annotation_v3_original_to_aligned_cropped(contours, stack=stack)
    
    if structures is None:
        structures = all_known_structures
    
    if colors is None:
        colors = {n: np.random.randint(0, 255, (3,)) for n in structures}
    
    if bg == 'original':
        img = DataManager.load_image(stack=stack, fn=fn, resol='lossless', version='jpeg')
        bg = img[::out_downsample, ::out_downsample].copy()
        
    if bg.ndim == 2:
        viz = gray2rgb(viz)
    else:
        viz = bg.copy()

    for name_u in structures:
        matched_contours = contours[(contours['name'] == name_u) & (contours['filename'] == fn)]
        for cnt_id, cnt_props in matched_contours.iterrows():
            verts_imgRes = cnt_props['vertices'] / out_downsample
            cv2.polylines(viz, [verts_imgRes.astype(np.int)], True, colors[name_u], 2)
            
            if show_labels:
                label_pos = np.mean(verts_imgRes, axis=0)
                if out_downsample == 32:
                    cv2.putText(viz, name_u, tuple(label_pos.astype(np.int)), cv2.FONT_HERSHEY_DUPLEX, 0.5, 
                            ((0,0,0)), 1)
                else:
                    cv2.putText(viz, name_u, tuple(label_pos.astype(np.int)), cv2.FONT_HERSHEY_DUPLEX, 1.5, 
                            ((0,0,0)), 2)
    
    return viz

# def annotation_v3_overlay_on(bg, stack, orientation=None,
#                             structure_names=None, downscale_factor=8,
#                             users=None, colors=None, show_labels=True, export_filepath_fmt=None,
#                             annotation_rootdir=None):
#     """
#     Works with annotation files version 3.
#     It is user's responsibility to ensure bg is the same downscaling as the labeling.
#     """

#     contour_df, _ = DataManager.load_annotation_v3(stack=stack, annotation_rootdir=annotation_midbrainIncluded_v2_rootdir)

#     downsample_factor = 8

#     anchor_filename = metadata_cache['anchor_fn'][stack]
#     sections_to_filenames = metadata_cache['sections_to_filenames'][stack]
#     filenames_to_sections = {f: s for s, f in sections_to_filenames.iteritems()
#                             if f not in ['Placeholder', 'Nonexisting', 'Rescan']}

#     # Load transforms, defined on thumbnails
#     import cPickle as pickle
#     Ts = pickle.load(open(thumbnail_data_dir + '/%(stack)s/%(stack)s_elastix_output/%(stack)s_transformsTo_anchor.pkl' % dict(stack=stack), 'r'))

#     Ts_inv_downsampled = {}
#     for fn, T0 in Ts.iteritems():
#         T = T0.copy()
#         T[:2, 2] = T[:2, 2] * 32 / downsample_factor
#         Tinv = np.linalg.inv(T)
#         Ts_inv_downsampled[fn] = Tinv

#     # Load bounds
#     crop_xmin, crop_xmax, crop_ymin, crop_ymax = metadata_cache['cropbox'][stack]
#     print 'crop:', crop_xmin, crop_xmax, crop_ymin, crop_ymax

#     #####################################
#     paired_structures = ['5N', '6N', '7N', '7n', 'Amb', 'LC', 'LRt', 'Pn', 'Tz', 'VLL', 'RMC', 'SNC', 'SNR', '3N', '4N',
#                     'Sp5I', 'Sp5O', 'Sp5C', 'PBG', '10N', 'VCA', 'VCP', 'DC']
#     singular_structures = ['AP', '12N', 'RtTg', 'SC', 'IC']
#     structures = paired_structures + singular_structures

#     structure_colors = {n: np.random.randint(0, 255, (3,)) for n in structures}

#     #######################################

#     first_sec, last_sec = metadata_cache['section_limits'][stack]

#     bar = show_progress_bar(first_sec, last_sec)

#     # for section in [270]:
#     for section in range(first_sec, last_sec+1):

#         t = time.time()

#         bar.value = section

#         fn = sections_to_filenames[section]
#         if fn in ['Nonexisting', 'Rescan', 'Placeholder']:
#             continue

#         img = imread(DataManager.get_image_filepath(stack, fn=fn, resol='lossless', version='compressed'))
#         viz = img[::downsample_factor, ::downsample_factor].copy()

#         for name_u in structures:
#             matched_contours = contour_df[(contour_df['name'] == name_u) & (contour_df['filename'] == fn)]
#             for cnt_id, cnt in matched_contours.iterrows():
#                 n = len(cnt['vertices'])

#                 # Transform points
#                 vertices_on_aligned = np.dot(Ts_inv_downsampled[fn], np.c_[cnt['vertices']/downsample_factor, np.ones((n,))].T).T[:, :2]

#                 xs = vertices_on_aligned[:,0] - crop_xmin * 32 / downsample_factor
#                 ys = vertices_on_aligned[:,1] - crop_ymin * 32 / downsample_factor

#                 vertices_on_aligned_cropped = np.c_[xs, ys].astype(np.int)

#                 cv2.polylines(viz, [vertices_on_aligned_cropped], True, structure_colors[name_u], 2)

#         sys.stderr.write('Overlay visualize: %.2f seconds\n' % (time.time() - t)) # 6 seconds

#         viz_fn = os.path.join(viz_dir, '%(fn)s_annotation.jpg' % dict(fn=fn))
#         imsave(viz_fn, viz)
