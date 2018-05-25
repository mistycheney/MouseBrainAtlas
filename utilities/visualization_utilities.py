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

def generate_scoremap_layer(stack, downscale, structure=None, scoremap=None, in_scoremap_downscale=32,
                            detector_id=None,
                    image_shape=None, return_mask=False, sec=None, fn=None,
                    color=(1,0,0), show_above=.01, cmap_name='jet', colorlist=None):
    '''
    Generate scoremap layer.
    Output are rescaled from down32 score maps.

    Args:
        downscale: downscale factor of the output scoremap
        structure: structure name. Needed if `scoremap` is not provided.
        show_above: only show scoremap with score higher than this value.
        scoremap: the dense score map. If not given, load previously generated ones.
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
        if scoremap is None:
            scoremap = DataManager.load_downscaled_scoremap(stack=stack, section=sec, fn=fn,
                            detector_id=detector_id, structure=structure, downscale=32)
        # Only use down32 because downsampling of 32 already exceeds the patch resolution (56 pixels).
        # Higher downsampling factor just shows artificial data, which is not better than rescaling down32.
        scoremap = np.minimum(rescale(scoremap, in_scoremap_downscale/float(downscale)), 1.)
        mask = scoremap > show_above
        if colorlist is None:
            cmap = plt.get_cmap(cmap_name)
            scoremap_viz = cmap(scoremap)[..., :3] # cmap() must take values between 0 and 1.
        else:
            scoremap_viz = colorlist[((len(colorlist)-1)*scoremap).astype(np.int)][..., :3]
            
    except Exception as e:
        raise Exception('Error loading scoremap of %s for image %s: %s\n' % (structure, fn, e))

    viz = img_as_ubyte(scoremap_viz)

    if return_mask:
        return viz, mask
    else:
        return viz

def scoremap_overlay_on(bg, stack, out_downscale, structure=None, scoremap=None, in_scoremap_downscale=32, detector_id=None, label_text=None, sec=None, fn=None, in_downscale=None, overlay_alpha=.3, image_version=None, show_above=.01, cmap_name='jet', overlay_bbox=None):
    """
    Draw scoremap on top of another image.
    
    Args:
        bg (2d-array of uint8): background image on top of which scoremap is drawn.
        structure (str): structure name. Needed if `scoremap` is not given.
        in_downscale (int): downscale factor of input background image.
    """

    if fn is None:
        assert sec is not None
        fn = metadata_cache['sections_to_filenames'][stack][sec]
        if is_invalid(fn): return

    t = time.time()
        
    if isinstance(bg, str) and bg == 'original':
        if image_version is None:
            classifier_properties = classifier_settings.loc[classifier_id]
            image_version = classifier_properties['input_img_version']

        # if out_downscale == 32 and image_version == '':
        #     # bg = DataManager.load_image(stack=stack, section=sec, fn=fn, resol='thumbnail', version='cropped')
        #     bg = DataManager.load_image_v2(stack=stack, section=sec, fn=fn, resol='thumbnail', prep_id=2)
        # else:
        # im = DataManager.load_image(stack=stack, section=sec, fn=fn, resol='lossless', version=image_version)
        try:
            im = DataManager.load_image_v2(stack=stack, section=sec, fn=fn, resol='lossless', prep_id=2, version=image_version)
            bg = im[::out_downscale, ::out_downscale]
        except:
            sys.stderr.write('Cannot load lossless jpeg, load downsampled jpeg instead.\n')
            bg = DataManager.load_image_v2(stack=stack, section=sec, fn=fn, resol='down'+str(out_downscale), prep_id=2, version=image_version)
    else:
        assert in_downscale is not None, "For user-given background image, its resolution `in_downscale` must be given."
        if in_downscale != out_downscale:
            bg = rescale(bg, float(in_downscale)/out_downscale)
    
    sys.stderr.write('Load and rescale background image: %.2f seconds\n' % (time.time() - t))

    t = time.time()
    scoremap_viz_mask = generate_scoremap_layer(stack=stack, scoremap=scoremap, in_scoremap_downscale=in_scoremap_downscale,
                                  sec=sec, fn=fn, structure=structure, downscale=out_downscale,
                        image_shape=bg.shape[:2], return_mask=True, detector_id=detector_id, show_above=show_above,
                             cmap_name=cmap_name)
    sys.stderr.write('Generate scoremap overlay: %.2f seconds.\n' % (time.time() - t))
    
    scoremap_viz, mask = scoremap_viz_mask

    if scoremap_viz.shape != bg.shape:
        t = time.time()
        scoremap_viz = resize(scoremap_viz, bg.shape + (3,), preserve_range=True)
        mask = resize(mask, bg.shape).astype(np.bool)
        sys.stderr.write('Scoremap size does not match background image size. Need to resize: %.2f seconds.\n' % (time.time() - t))

    if overlay_bbox is not None:
        xmin, xmax, ymin, ymax = overlay_bbox
        scoremap_viz = scoremap_viz[ymin/out_downscale:(ymax+1)/out_downscale, xmin/out_downscale:(xmax+1)/out_downscale]
        mask = mask[ymin/out_downscale:(ymax+1)/out_downscale, xmin/out_downscale:(xmax+1)/out_downscale]

    viz = img_as_ubyte(gray2rgb(bg))
    print scoremap_viz.shape, mask.shape
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


def annotation_from_multiple_warped_atlases_overlay_on(bg, warped_volumes_sets, stack_fixed, 
                                                       volume_downsample=None, 
                                                       volume_resolution=None,
                                            fn=None, sec=None, orientation='sagittal',
                            structures=None, out_downsample=None, out_resolution=None,
                            users=None, level_colors=None, levels=None, show_text=True, label_color=(0,0,0),
                             contours=None, contour_width=1, bg_img_version='grayJpeg'):
    """
    Args:
        warped_volumes_sets ({set_name: {structure: (3d probability array, (3,)-array origin wrt wholebrain)}})
        levels (list of float): probability levels at which the contours are drawn.
        level_colors (dict {set_name: dict {float: (3,)-ndarray of float}}): 256-based contour color for each level for each set
    """

    wholebrainXYcropped_origin_wrt_wholebrain = DataManager.get_domain_origin(stack=stack_fixed, 
                                                                              domain='wholebrainXYcropped',
                                                                             resolution=volume_resolution)
    # This is down32 of the raw resolution of the given stack.
    
    if level_colors is None:
        level_colors = {set_name: {0.1: (0,255,255),
                    0.25: (0,255,0),
                    0.5: (255,0,0),
                    0.75: (255,255,0),
                    0.99: (255,0,255)} for set_name in warped_volumes_sets.keys()}

    if levels is None:
        levels = level_colors.values()[0].keys()

    volume_resolution_um = convert_resolution_string_to_voxel_size(resolution=volume_resolution, stack=stack_fixed)
        
    t = time.time()

    if bg == 'original':

        # if out_downsample == 32:
        #     resol_str = 'thumbnail'
        # elif out_downsample == 1:
        #     resol_str = 'lossless'
        # else:
        #     resol_str = 'down'+str(out_downsample)

        out_resolution_um = convert_resolution_string_to_voxel_size(resolution=out_resolution, stack=stack_fixed)
        if stack_fixed == 'ChatCryoJane201710':
            out_downsample = out_resolution_um / XY_PIXEL_DISTANCE_LOSSLESS_AXIOSCAN
        else:
            out_downsample = out_resolution_um / XY_PIXEL_DISTANCE_LOSSLESS

        try:
            bg = DataManager.load_image_v2(stack=stack_fixed, section=sec, fn=fn, resol=out_resolution, prep_id=2, version=bg_img_version)
        except Exception as e:
            sys.stderr.write('Cannot load downsampled jpeg, load lossless instead: %s.\n' % e)
            bg = DataManager.load_image_v2(stack=stack_fixed, section=sec, fn=fn, resol='lossless', prep_id=2, version=bg_img_version)
            bg = rescale_by_resampling(bg, 1./out_downsample)
                
    if bg.ndim == 2:
        bg = gray2rgb(bg)

    viz = bg.copy()

    assert orientation == 'sagittal', 'Currently only support drawing on sagittal sections'

    z_wrt_wholebrain = DataManager.convert_section_to_z(stack=stack_fixed, sec=sec, resolution=volume_resolution, mid=True, z_begin=0)

    # Find moving volume annotation contours.
    # for set_name, warped_volumes in warped_volumes_sets.iteritems():
    #     for name_s, vol in warped_volumes.iteritems():
    for set_name in warped_volumes_sets.keys(): # This avoids loading entire warped_volumes (maybe?)
        for name_s, (vol, vol_origin_wrt_wholebrain) in warped_volumes_sets[set_name].iteritems():
            # structure does not include level z, skip
            bbox = bbox_3d(vol)
            zmin_wrt_wholebrain = bbox[4] + vol_origin_wrt_wholebrain[2]
            zmax_wrt_wholebrain = bbox[5] + vol_origin_wrt_wholebrain[2]
            # print zmin_wrt_wholebrain, zmax_wrt_wholebrain, z_wrt_wholebrain
            if z_wrt_wholebrain < zmin_wrt_wholebrain or z_wrt_wholebrain > zmax_wrt_wholebrain:
                continue

            print set_name, name_s

            label_pos = None

            for level in levels:
                cnts_rc_wrt_vol = find_contours(vol[..., int(np.round(z_wrt_wholebrain - vol_origin_wrt_wholebrain[2]))], level=level)
                for cnt_rc_wrt_vol in cnts_rc_wrt_vol:
                    cnt_wrt_cropped_volRes = cnt_rc_wrt_vol[:,::-1] + (vol_origin_wrt_wholebrain[0], vol_origin_wrt_wholebrain[1]) - wholebrainXYcropped_origin_wrt_wholebrain[:2]
                    cnt_wrt_cropped_imgRes = cnt_wrt_cropped_volRes * volume_resolution_um / out_resolution_um
                    cv2.polylines(viz, [cnt_wrt_cropped_imgRes.astype(np.int)],
                                  True, level_colors[set_name][level], contour_width)

                    if show_text:
                        if label_pos is None:
                            label_pos = np.mean(cnt_wrt_cropped_imgRes, axis=0)

            # Show text
            if label_pos is not None:
                cv2.putText(viz, name_s, tuple(label_pos.astype(np.int)),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (label_color), 3)

    return viz

def annotation_by_human_overlay_on(bg, stack=None, fn=None, sec=None, orientation='sagittal',
                            structures=None, out_downsample=8,
                            users=None, colors=None, show_labels=True,
                             contours=None, timestamp='latest', suffix='contours', return_timestamp=False):
    """
    Draw annotation contours on a user-given background image.

    Args:
        out_downsample (int): downsample factor of the output images.
        structures (list of str): list of structure names (sided) to show.
        contours (pandas.DataFrame): rows are polygon indices.
        colors (dict {name_s: (3,)-ndarray }): contour color of each structure (sided)
    """

    if fn is None:
        assert sec is not None
        fn = metadata_cache['sections_to_filenames'][stack][sec]
        if is_invalid(fn):
            return

    if contours is None:
        contours_df, timestamp = DataManager.load_annotation_v4(stack=stack, timestamp=timestamp, suffix=suffix, return_timestamp=True)
        contours = contours_df[(contours_df['orientation'] == orientation) & (contours_df['resolution'] == 'raw')]
        contours = contours.drop_duplicates(subset=['section', 'name', 'side', 'filename', 'resolution', 'creator'])
        contours = convert_annotation_v3_original_to_aligned_cropped_v2(contours, stack=stack, out_resolution='raw')

    if structures is None:
        structures = all_known_structures_sided

    if colors is None:
        colors = {n: np.random.randint(0, 255, (3,)) for n in structures}

    if bg == 'original':
        img = DataManager.load_image_v2(stack=stack, prep_id=2, fn=fn, resol='raw', version='jpeg')
        bg = img[::out_downsample, ::out_downsample].copy()

    if bg.ndim == 2:
        viz = gray2rgb(viz)
    else:
        viz = bg.copy()

    for name_s in structures:
        matched_contours_raw = contours[(contours['name'] == convert_to_original_name(name_s)) & (contours['filename'] == fn)]
        for cnt_id, cnt_props in matched_contours_raw.iterrows():
            verts_imgRes = cnt_props['vertices'] / out_downsample
            cv2.polylines(viz, [verts_imgRes.astype(np.int)], True, colors[name_s], 2)

            if show_labels:
                label_pos = np.mean(verts_imgRes, axis=0)
                if name_s in paired_structures:
                    label = convert_to_original_name(name_s) + '(R)'
                elif '_L' in name_s:
                    label = convert_to_original_name(name_s) + '(L)'
                else:
                    label = convert_to_original_name(name_s)
                if out_downsample == 32:
                    cv2.putText(viz, label, tuple(label_pos.astype(np.int)), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                            ((0,0,0)), 1)
                else:
                    cv2.putText(viz, label, tuple(label_pos.astype(np.int)), cv2.FONT_HERSHEY_TRIPLEX, 1.0,
                            ((0,0,0)), 2)

    if return_timestamp:
        return viz, timestamp
    else:
        return viz
