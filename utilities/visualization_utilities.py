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

def get_structure_contours_from_structure_volumes_v4(volumes, stack, 
                                                     resolution, level, 
                                                     out_resolution,
                                                     orientation='sagittal',
                                                     sections=None,
                                                     positions=None,
                                                     sample_every=1,
                                                    use_unsided_name_as_key=False):
    """
    Re-section volumes and obtain contour coordinates on planes at requested positions.

    Args:
        
        volumes (dict of (3D array, 3-tuple)): {structure: (volume, origin_wrt_wholebrain)}. volume is a 3d array of probability values.
        resolution (str): resolution of input volumes.
        level (float or dict or dict of list): the cut-off probability at which surfaces are generated from probabilistic volumes. Default is 0.5.
        out_resolution (str): resolution of output contours.
        orientation (str): sagittal, horizontal or coronal.
        sample_every (int): how sparse to sample contour vertices.
        positions (list of int): provide either positions or sections to indicate which planes to find contours for.
        sections (list of int): provide either positions or sections to indicate which planes to find contours for.
        
    Returns:
        Dict {section: {name_s: contour vertices}}: if positions are given, vertices are wrt wholebrainWithMargin in raw resolution; if sections are given, vertices are wrt alignedBrainstemCrop in raw resolution
    """

    use_volume_instead_of_images = positions is not None
    print 'use_volume_instead_of_images', use_volume_instead_of_images
    
    from collections import defaultdict
    
    structure_contours_wrt_outputFrame_rawResol = defaultdict(lambda: defaultdict(dict))

    converter = CoordinatesConverter(stack=stack, section_list=metadata_cache['sections_to_filenames'][stack].keys())
    converter.register_new_resolution('structure_volume_resol', resol_um=convert_resolution_string_to_um(resolution=resolution, stack=stack))
    
    if use_volume_instead_of_images:
        converter.register_new_resolution('intensity_volume_resol', resol_um=convert_resolution_string_to_um(resolution=out_resolution, stack=stack))
    else:
        converter.register_new_resolution('image', resol_um=convert_resolution_string_to_um(resolution=out_resolution, stack=stack))

    
    for name_s, (structure_volume_volResol, origin_wrt_wholebrain_volResol) in volumes.iteritems():

        # Generate structure-specific coordinate frames.
        converter.derive_three_view_frames(base_frame_name=name_s, 
        origin_wrt_wholebrain_um=convert_resolution_string_to_um(resolution=resolution, stack=stack) * origin_wrt_wholebrain_volResol,
        zdim_um=convert_resolution_string_to_um(resolution=resolution, stack=stack) * structure_volume_volResol.shape[2])

        if use_volume_instead_of_images:

            n = len(positions)
            positions_of_all_sections_wrt_structureVolume = converter.convert_frame_and_resolution(
            p = np.c_[np.nan * np.ones((n,)), np.nan * np.ones((n,)), positions],
            in_wrt=('wholebrainWithMargin', orientation), in_resolution='intensity_volume_resol',
            out_wrt=(name_s, orientation), out_resolution='structure_volume_resol')[..., 2].flatten()
        else:
            positions_of_all_sections_wrt_structureVolume = converter.convert_frame_and_resolution(
            p=np.array(sections)[:,None],
            in_wrt=('wholebrain', 'sagittal'), in_resolution='section',
            out_wrt=(name_s, 'sagittal'), out_resolution='structure_volume_resol')[..., 2].flatten()
            
        structure_ddim = structure_volume_volResol.shape[2]
        
        valid_mask = (positions_of_all_sections_wrt_structureVolume >= 0) & (positions_of_all_sections_wrt_structureVolume < structure_ddim)
        if np.count_nonzero(valid_mask) == 0:
#             sys.stderr.write("%s, valid_mask is empty.\n" % name_s)
            continue

        positions_of_all_sections_wrt_structureVolume = positions_of_all_sections_wrt_structureVolume[valid_mask]
        positions_of_all_sections_wrt_structureVolume = np.round(positions_of_all_sections_wrt_structureVolume).astype(np.int)
        
        if isinstance(level, dict):
            level_this_structure = level[name_s]
        else:
            level_this_structure = level

        if isinstance(level_this_structure, float):
            level_this_structure = [level_this_structure]
                        
        for one_level in level_this_structure:

            contour_2d_wrt_structureVolume_sectionPositions_volResol = \
            find_contour_points_3d(structure_volume_volResol >= one_level,
                                    along_direction=orientation,
                                    sample_every=sample_every,
                                    positions=positions_of_all_sections_wrt_structureVolume)

            for d_wrt_structureVolume, cnt_uv_wrt_structureVolume in contour_2d_wrt_structureVolume_sectionPositions_volResol.iteritems():

                contour_3d_wrt_structureVolume_volResol = np.column_stack([cnt_uv_wrt_structureVolume, np.ones((len(cnt_uv_wrt_structureVolume),)) * d_wrt_structureVolume])

                if use_volume_instead_of_images:
                    contour_3d_wrt_outputFrame_uv_rawResol_section = converter.convert_frame_and_resolution(
                    p=contour_3d_wrt_structureVolume_volResol,
                    in_wrt=(name_s, orientation), in_resolution='structure_volume_resol',
                    out_wrt=('wholebrainWithMargin', orientation), out_resolution='intensity_volume_resol')
                else:
                    contour_3d_wrt_outputFrame_uv_rawResol_section = converter.convert_frame_and_resolution(
                    p=contour_3d_wrt_structureVolume_volResol,
                    in_wrt=(name_s, 'sagittal'), in_resolution='structure_volume_resol',
                    out_wrt=('wholebrainXYcropped', orientation), out_resolution='image_image_section')                    

                assert len(np.unique(contour_3d_wrt_outputFrame_uv_rawResol_section[:,2])) == 1
                pos = int(contour_3d_wrt_outputFrame_uv_rawResol_section[0,2])

                if use_unsided_name_as_key:
                    name = convert_to_unsided_label(name_s)
                else:
                    name = name_s

                structure_contours_wrt_outputFrame_rawResol[pos][name][one_level] = contour_3d_wrt_outputFrame_uv_rawResol_section[..., :2]
                
    return structure_contours_wrt_outputFrame_rawResol


# def get_structure_contours_from_structure_volumes_v3_volume(volumes, stack, 
#                                                             positions, orientation,
#                                                      resolution, level, 
#                                                      out_resolution,
#                                                      sample_every=1,
#                                                     use_unsided_name_as_key=False):
#     """
#     Re-section atlas volumes and obtain structure contours on requested voxel positions.
#     Resolution of output contours are in volume resolution.

#     Args:
#         volumes (dict of (3D array, 3-tuple)): {structure: (volume, origin_wrt_wholebrain)}. volume is a 3d array of probability values.
#         positions (int list):
#         orientation (str): sagittal, horizontal or coronal.
#         resolution (str): resolution of input volumes.
#         level (float or dict or dict of list): the cut-off probability at which surfaces are generated from probabilistic volumes. Default is 0.5.
#         sample_every (int): how sparse to sample contour vertices.
#         out_resolution (str): resolution of output contours.

#     Returns:
#         Dict {section: {name_s: contour vertices}}: wrt alignedBrainstemCrop in raw resolution
#     """

#     from collections import defaultdict
    
#     # assert orientation == 'sagittal'
    
#     structure_contours_wrt_outputFrame_rawResol = defaultdict(lambda: defaultdict(dict))

#     converter = CoordinatesConverter(stack=stack, section_list=metadata_cache['sections_to_filenames'][stack].keys())

#     converter.register_new_resolution('structure_volume_resol', resol_um=convert_resolution_string_to_um(resolution=resolution, stack=stack))
#     converter.register_new_resolution('intensity_volume_resol', resol_um=convert_resolution_string_to_um(resolution=out_resolution, stack=stack))
    
#     for name_s, (structure_volume_volResol, origin_wrt_wholebrain_volResol) in volumes.iteritems():

#         # Generate structure-specific coordinate frames.
#         converter.derive_three_view_frames(base_frame_name=name_s, 
#         origin_wrt_wholebrain_um=convert_resolution_string_to_um(resolution=resolution, stack=stack) * origin_wrt_wholebrain_volResol,
#         zdim_um=convert_resolution_string_to_um(resolution=resolution, stack=stack) * structure_volume_volResol.shape[2])

#         n = len(positions)
                    
#         positions_of_all_sections_wrt_structureVolume = converter.convert_frame_and_resolution(
#         p = np.c_[np.nan * np.ones((n,)), np.nan * np.ones((n,)), positions],
#         in_wrt=('wholebrainWithMargin', orientation), in_resolution='intensity_volume_resol',
#         out_wrt=(name_s, orientation), out_resolution='structure_volume_resol')[..., 2].flatten()
                        
#         structure_ddim = structure_volume_volResol.shape[2]
        
#         valid_mask = (positions_of_all_sections_wrt_structureVolume >= 0) & (positions_of_all_sections_wrt_structureVolume < structure_ddim)
#         if np.count_nonzero(valid_mask) == 0:
# #             sys.stderr.write("%s, valid_mask is empty.\n" % name_s)
#             continue

#         positions_of_all_sections_wrt_structureVolume = positions_of_all_sections_wrt_structureVolume[valid_mask]
#         positions_of_all_sections_wrt_structureVolume = np.round(positions_of_all_sections_wrt_structureVolume).astype(np.int)
        
#         if isinstance(level, dict):
#             level_this_structure = level[name_s]
#         else:
#             level_this_structure = level

#         if isinstance(level_this_structure, float):
#             level_this_structure = [level_this_structure]
                        
#         for one_level in level_this_structure:

#             contour_2d_wrt_structureVolume_sectionPositions_volResol = \
#             find_contour_points_3d(structure_volume_volResol >= one_level,
#                                     along_direction=orientation,
#                                     sample_every=sample_every,
#                                     positions=positions_of_all_sections_wrt_structureVolume)

#             for d_wrt_structureVolume, cnt_uv_wrt_structureVolume in contour_2d_wrt_structureVolume_sectionPositions_volResol.iteritems():

#                 contour_3d_wrt_structureVolume_volResol = np.column_stack([cnt_uv_wrt_structureVolume, np.ones((len(cnt_uv_wrt_structureVolume),)) * d_wrt_structureVolume])

#     #             contour_3d_wrt_wholebrain_uv_rawResol_section = converter.convert_frame_and_resolution(
#     #                 p=contour_3d_wrt_structureVolume_volResol,
#     #                 in_wrt=(name_s, 'sagittal'), in_resolution='structure_volume',
#     #                 out_wrt=('wholebrain', 'sagittal'), out_resolution='image_image_section')

#                 contour_3d_wrt_outputFrame_uv_rawResol_section = converter.convert_frame_and_resolution(
#                     p=contour_3d_wrt_structureVolume_volResol,
#                     in_wrt=(name_s, orientation), in_resolution='structure_volume_resol',
#                     out_wrt=('wholebrainWithMargin', orientation), out_resolution='intensity_volume_resol')

#                 assert len(np.unique(contour_3d_wrt_outputFrame_uv_rawResol_section[:,2])) == 1
#                 pos = int(contour_3d_wrt_outputFrame_uv_rawResol_section[0,2])

#                 if use_unsided_name_as_key:
#                     name = convert_to_unsided_label(name_s)
#                 else:
#                     name = name_s

#                 structure_contours_wrt_outputFrame_rawResol[pos][name][one_level] = contour_3d_wrt_outputFrame_uv_rawResol_section[..., :2]
                
#     return structure_contours_wrt_outputFrame_rawResol


# def get_structure_contours_from_structure_volumes_v3(volumes, stack, sections, 
#                                                      resolution, level, 
#                                                      out_resolution,
#                                                      sample_every=1,
#                                                     use_unsided_name_as_key=False,
#                                                     ):
#     """
#     Re-section atlas volumes and obtain structure contours on each section.
#     v3 supports multiple levels.

#     Args:
#         volumes (dict of (3D array, 3-tuple)): {structure: (volume, origin_wrt_wholebrain)}. volume is a 3d array of probability values.
#         sections (list of int):
#         resolution (str): resolution of input volumes.
#         level (float or dict or dict of list): the cut-off probability at which surfaces are generated from probabilistic volumes. Default is 0.5.
#         sample_every (int): how sparse to sample contour vertices.
#         out_resolution (str): resolution of output contours.

#     Returns:
#         Dict {section: {name_s: contour vertices}}: wrt alignedBrainstemCrop in raw resolution
#     """

#     from collections import defaultdict
    
#     structure_contours_wrt_alignedBrainstemCrop_rawResol = defaultdict(lambda: defaultdict(dict))

#     converter = CoordinatesConverter(stack=stack, section_list=metadata_cache['sections_to_filenames'][stack].keys())

#     converter.register_new_resolution('structure_volume_resol', resol_um=convert_resolution_string_to_um(resolution=resolution, stack=stack))
#     converter.register_new_resolution('image', resol_um=convert_resolution_string_to_um(resolution=out_resolution, stack=stack))
    
#     for name_s, (structure_volume_volResol, origin_wrt_wholebrain_volResol) in volumes.iteritems():

#         # Generate structure-specific coordinate frames.
#         converter.derive_three_view_frames(base_frame_name=name_s, 
#         origin_wrt_wholebrain_um=convert_resolution_string_to_um(resolution=resolution, stack=stack) * origin_wrt_wholebrain_volResol,
#         zdim_um=convert_resolution_string_to_um(resolution=resolution, stack=stack) * structure_volume_volResol.shape[2])

#         positions_of_all_sections_wrt_structureVolume = converter.convert_frame_and_resolution(
#         p=np.array(sections)[:,None],
#         in_wrt=('wholebrain', orientation), in_resolution='section',
#         out_wrt=(name_s, 'sagittal'), out_resolution='structure_volume_resol')[..., 2].flatten()
            
#         structure_ddim = structure_volume_volResol.shape[2]
        
#         valid_mask = (positions_of_all_sections_wrt_structureVolume >= 0) & (positions_of_all_sections_wrt_structureVolume < structure_ddim)
#         if np.count_nonzero(valid_mask) == 0:
# #             sys.stderr.write("%s, valid_mask is empty.\n" % name_s)
#             continue

#         positions_of_all_sections_wrt_structureVolume = positions_of_all_sections_wrt_structureVolume[valid_mask]
#         positions_of_all_sections_wrt_structureVolume = np.round(positions_of_all_sections_wrt_structureVolume).astype(np.int)
        
#         if isinstance(level, dict):
#             level_this_structure = level[name_s]
#         else:
#             level_this_structure = level

#         if isinstance(level_this_structure, float):
#             level_this_structure = [level_this_structure]
                                        
#         for one_level in level_this_structure:

#             contour_2d_wrt_structureVolume_sectionPositions_volResol = \
#             find_contour_points_3d(structure_volume_volResol >= one_level,
#                                     along_direction=orientation,
#                                     sample_every=sample_every,
#                                     positions=positions_of_all_sections_wrt_structureVolume)

#             for d_wrt_structureVolume, cnt_uv_wrt_structureVolume in contour_2d_wrt_structureVolume_sectionPositions_volResol.iteritems():

#                 contour_3d_wrt_structureVolume_volResol = np.column_stack([cnt_uv_wrt_structureVolume, np.ones((len(cnt_uv_wrt_structureVolume),)) * d_wrt_structureVolume])

#     #             contour_3d_wrt_wholebrain_uv_rawResol_section = converter.convert_frame_and_resolution(
#     #                 p=contour_3d_wrt_structureVolume_volResol,
#     #                 in_wrt=(name_s, 'sagittal'), in_resolution='structure_volume',
#     #                 out_wrt=('wholebrain', 'sagittal'), out_resolution='image_image_section')

#                 contour_3d_wrt_alignedBrainstemCrop_uv_rawResol_section = converter.convert_frame_and_resolution(
#                     p=contour_3d_wrt_structureVolume_volResol,
#                     in_wrt=(name_s, 'sagittal'), in_resolution='structure_volume_resol',
#                     out_wrt=('wholebrainXYcropped', orientation), out_resolution='image_image_section')

#                 assert len(np.unique(contour_3d_wrt_alignedBrainstemCrop_uv_rawResol_section[:,2])) == 1
#                 sec = int(contour_3d_wrt_alignedBrainstemCrop_uv_rawResol_section[0,2])

#                 if use_unsided_name_as_key:
#                     name = convert_to_unsided_label(name_s)
#                 else:
#                     name = name_s

#                 structure_contours_wrt_alignedBrainstemCrop_rawResol[sec][name][one_level] = contour_3d_wrt_alignedBrainstemCrop_uv_rawResol_section[..., :2]
        
#     return structure_contours_wrt_alignedBrainstemCrop_rawResol


def annotation_from_multiple_warped_atlases_overlay_on_v3(warped_volumes_sets, 
                                                              stack, 
                                                          volume_resolution=None,
                                                          orientation='sagittal',
                                                          structures=None, out_resolution=None,                                                                                           level_colors=None, levels=None, show_text=True, label_color=(0,0,0),
                                                          contour_width=1,
                                                             bg_volume=None, bg_img_version=None, 
                                                          sections=None, positions=None):
    """
    Overlay 2-D resection contours of structures (warped atlases or hand-drawn structures) on a given brain (either the sagittal stack images or the reconstructed intensity volume).

    Args:
        warped_volumes_sets (dict): {set_name: {structure: (3d probability array, (3,)-array origin wrt wholebrain)}}
        stack (str): the brain to draw contours on
        volume_resolution (str): resolution of the loaded warped volumes.
        orientation (str): direction of the resection contours. One of sagittal, coronal or horizontal.
        structures (str list): list of structures to draw.
        out_resolution (str): resolution of the background images or background volume.
        levels (list of float): probability levels at which the contours are drawn.
        level_colors (dict {set_name: dict {float: (3,)-ndarray of float}}): 256-based contour color for each level for each set
        show_text (bool): whether to show label text.
        contour_width (int): contour line width in pixels on output images.
        bg_volume (3-d array): the background volume to draw contours on.
        bg_img_version (str): version of the background images to draw contours on.
        sections (list of int): the section numbers of the images to draw contours on. Used if background are images in stack. Default is all valid sections.
        positions (list of int): the positions of the resectioned planes to draw contours on. Used if background is intensity volume. Default is all positions of intensity volume at given direction.
    """
    
    if level_colors is None:
        level_colors = {set_name: LEVEL_TO_COLOR_LINE 
                        for set_name in warped_volumes_sets.keys()}        

    if levels is None:
        levels = level_colors.values()[0].keys()
        
    if bg_volume is None and bg_img_version is not None:
        
        if sections is None:
            # valid_secmin = np.min(metadata_cache['valid_sections'][stack])
            # valid_secmax = np.max(metadata_cache['valid_sections'][stack])
            # sections = [sec for sec in range(valid_secmin, valid_secmax+1) if not is_invalid(sec=sec, stack=stack)]
            sections = metadata_cache['valid_sections'][stack]
        
        contours_all_sets_all_sections_all_structures_all_levels_outResol = \
        {set_name: \
         get_structure_contours_from_structure_volumes_v4(volumes={s: warped_volumes_sets[set_name][s] 
                                                                      for s in structures}, 
                                                             stack=stack, 
                                                             sections=sections,
                                                            resolution=volume_resolution, 
                                                             out_resolution=out_resolution,
                                                             level=levels, 
                                                             sample_every=5)
        for set_name in warped_volumes_sets.keys()}

    elif bg_volume is not None and bg_img_version is None:
        
        if positions is None:
            if orientation == 'sagittal':
                depth_dim = bg_volume.shape[2]
            elif orientation == 'coronal':
                depth_dim = bg_volume.shape[1]
            elif orientation == 'horizontal':
                depth_dim = bg_volume.shape[0]
            positions = np.arange(0, depth_dim)
            
        contours_all_sets_all_sections_all_structures_all_levels_outResol = \
        {set_name: \
         get_structure_contours_from_structure_volumes_v4(volumes={s: warped_volumes_sets[set_name][s] 
                                                                      for s in structures}, 
                                                             stack=stack, 
                                                                positions=positions, 
                                                            orientation=orientation,                                                        resolution=volume_resolution, 
                                                             out_resolution=out_resolution,
                                                             level=levels, 
                                                             sample_every=5)
        for set_name in warped_volumes_sets.keys()}
             
    else:
        raise Exception("Must provide either `bg_volume` or `bg_image_version`.")
            
    sections_spanned = set.union(*[set(x.keys()) for set_name, x in contours_all_sets_all_sections_all_structures_all_levels_outResol.iteritems()])
        
    vizs_all_sections = {}
        
    for sec in sections_spanned:
        
        if bg_volume is None and bg_img_version is not None:
            viz = DataManager.load_image_v2(stack=stack, prep_id=2, resol=out_resolution, version=bg_img_version, section=sec)
        elif bg_volume is not None and bg_img_version is None:
            if orientation == 'sagittal':
                viz = bg_volume[..., sec].copy()
            elif orientation == 'coronal':
                viz = bg_volume[:, sec, ::-1].copy()
            elif orientation == 'horizontal':
                viz = bg_volume[sec, :, ::-1].T.copy()
        else:
            raise Exception("Must provide either `bg_volume` or `bg_image_version`.")

        # Convert to RGB so colored contours can be drawn on it.
        if viz.ndim == 2:
            viz = gray2rgb(viz)

        for set_name, cnts_all_sections_all_structures_all_levels_outResol \
        in contours_all_sets_all_sections_all_structures_all_levels_outResol.iteritems():            
        
            for name_s, cnt_all_levels_outResol in cnts_all_sections_all_structures_all_levels_outResol[sec].iteritems():

                if show_text:
                    # Put label at the center of the contour of arbitrary level.
                    label_pos_outResol = np.mean(cnt_all_levels_outResol.values()[0], axis=0)
                    cv2.putText(viz, name_s, tuple(label_pos_outResol.astype(np.int)),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (label_color), 3)

                for level in set(cnt_all_levels_outResol.keys()) & set(levels):
                    cnt_outResol = cnt_all_levels_outResol[level]
                    cv2.polylines(viz, [cnt_outResol.astype(np.int)], 
                                  isClosed=True, 
                                  color=level_colors[set_name][level], 
                                  thickness=contour_width)

        vizs_all_sections[sec] = viz

    return vizs_all_sections
    

# def annotation_from_multiple_warped_atlases_overlay_on_v2_volume(warped_volumes_sets, 
#                                                               stack_fixed, 
#                                                           volume_resolution=None,
#                                                           fn=None, sections=None, orientation='sagittal',
#                                                           structures=None, out_resolution=None,
#                                                           level_colors=None, levels=None, show_text=True, label_color=(0,0,0),
#                                                           contours=None, contour_width=1,
#                                                              bg_volume=None):
#     """
#     Overlay contours on the intensity volume of the given brain.

#     Args:
#         volume_resolution (str): resolution of the loaded volume.
#         out_resolution (str): resolution of the output images.
#         structures (str list): list of structures to draw.
#         warped_volumes_sets ({set_name: {structure: (3d probability array, (3,)-array origin wrt wholebrain)}})
#         levels (list of float): probability levels at which the contours are drawn.
#         level_colors (dict {set_name: dict {float: (3,)-ndarray of float}}): 256-based contour color for each level for each set
#         contour_width (int): contour line width in pixels on output images.
#     """
    
#     # assert orientation == 'sagittal', 'This function currently only supports drawing on sagittal sections.'
    
#     if level_colors is None:
#         level_colors = {set_name: LEVEL_TO_COLOR_LINE 
#                         for set_name in warped_volumes_sets.keys()}        

#     if levels is None:
#         levels = level_colors.values()[0].keys()
            
#     if orientation == 'sagittal':
#         depth_dim = bg_volume.shape[2]
#     elif orientation == 'coronal':
#         depth_dim = bg_volume.shape[0]
#     elif orientation == 'horizontal':
#         depth_dim = bg_volume.shape[1]
            
#     contours_all_sets_all_sections_all_structures_all_levels_outResol = \
#     {set_name: \
#      get_structure_contours_from_structure_volumes_v3_volume(volumes={s: warped_volumes_sets[set_name][s] 
#                                                                   for s in structures}, 
#                                                          stack=stack_fixed, 
#                                                             positions=np.arange(0,depth_dim), 
#                                                         orientation=orientation,                                                        resolution=volume_resolution, 
#                                                          out_resolution=out_resolution,
#                                                          level=levels, 
#                                                          sample_every=5)
#     for set_name in warped_volumes_sets.keys()}
                        
#     sections_spanned = set.union(*[set(x.keys()) for set_name, x in contours_all_sets_all_sections_all_structures_all_levels_outResol.iteritems()])
        
#     vizs_all_sections = {}
        
#     for sec in sections_spanned:
                
#         if orientation == 'sagittal':
#             viz = bg_volume[..., sec].copy()
#         elif orientation == 'coronal':
#             viz = bg_volume[:, sec, :].copy()
#         elif orientation == 'horizontal':
#             viz = bg_volume[sec, :, :].copy()

#         # Convert to RGB so colored contours can be drawn on it.
#         if viz.ndim == 2:
#             viz = gray2rgb(viz)

#         for set_name, cnts_all_sections_all_structures_all_levels_outResol \
#         in contours_all_sets_all_sections_all_structures_all_levels_outResol.iteritems():            
        
#             for name_s, cnt_all_levels_outResol in cnts_all_sections_all_structures_all_levels_outResol[sec].iteritems():

#                 if show_text:
#                     # Put label at the center of the contour of arbitrary level.
#                     label_pos_outResol = np.mean(cnt_all_levels_outResol.values()[0], axis=0)
#                     cv2.putText(viz, name_s, tuple(label_pos_outResol.astype(np.int)),
#                             cv2.FONT_HERSHEY_DUPLEX, 1, (label_color), 3)

#                 for level in set(cnt_all_levels_outResol.keys()) & set(levels):
#                     cnt_outResol = cnt_all_levels_outResol[level]
#                     cv2.polylines(viz, [cnt_outResol.astype(np.int)], 
#                                   isClosed=True, 
#                                   color=level_colors[set_name][level], 
#                                   thickness=contour_width)

#         vizs_all_sections[sec] = viz

#     return vizs_all_sections


# def annotation_from_multiple_warped_atlases_overlay_on_v2(warped_volumes_sets, stack_fixed, 
#                                                        volume_resolution=None,
#                                             fn=None, sections=None, orientation='sagittal',
#                             structures=None, 
#                                                           out_resolution=None,
#                             level_colors=None, levels=None, show_text=True, label_color=(0,0,0),
#                              contours=None, contour_width=1, bg_img_version='grayJpeg'):
#     """
#     Args:
#         bg_img_version (str): version of the background image.
#         volume_resolution (str): resolution of the loaded volume.
#         out_resolution (str): resolution of the output images.
#         structures (str list): list of structures to draw.
#         warped_volumes_sets ({set_name: {structure: (3d probability array, (3,)-array origin wrt wholebrain)}})
#         levels (list of float): probability levels at which the contours are drawn.
#         level_colors (dict {set_name: dict {float: (3,)-ndarray of float}}): 256-based contour color for each level for each set
#         contour_width (int): contour line width in pixels on output images.
#     """
    
#     assert orientation == 'sagittal', 'This function currently only supports drawing on sagittal sections.'
    
#     if level_colors is None:
#         level_colors = {set_name: LEVEL_TO_COLOR_LINE 
#                         for set_name in warped_volumes_sets.keys()}        

#     if levels is None:
#         levels = level_colors.values()[0].keys()
            
#     contours_all_sets_all_sections_all_structures_all_levels_outResol = \
#     {set_name: \
#      get_structure_contours_from_structure_volumes_v3(volumes={s: warped_volumes_sets[set_name][s] 
#                                                                   for s in structures}, 
#                                                          stack=stack_fixed, 
#                                                          sections=sections,
#                                                         resolution=volume_resolution, 
#                                                          out_resolution=out_resolution,
#                                                          level=levels, 
#                                                          sample_every=5)
#     for set_name in warped_volumes_sets.keys()}
                
#     sections_spanned = set.union(*[set(x.keys()) for set_name, x in contours_all_sets_all_sections_all_structures_all_levels_outResol.iteritems()])
        
#     vizs_all_sections = {}
        
#     for sec in sections_spanned:

#         if is_invalid(sec=sec, stack=stack_fixed):
#             continue

#         viz = DataManager.load_image_v2(stack=stack_fixed, prep_id=2, resol=out_resolution, version=bg_img_version, section=sec)

#         # Convert to RGB so colored contours can be drawn on it.
#         if viz.ndim == 2:
#             viz = gray2rgb(viz)

#         for set_name, cnts_all_sections_all_structures_all_levels_outResol \
#         in contours_all_sets_all_sections_all_structures_all_levels_outResol.iteritems():            
        
#             for name_s, cnt_all_levels_outResol in cnts_all_sections_all_structures_all_levels_outResol[sec].iteritems():

#                 if show_text:
#                     # Put label at the center of the contour of arbitrary level.
#                     label_pos_outResol = np.mean(cnt_all_levels_outResol.values()[0], axis=0)
#                     cv2.putText(viz, name_s, tuple(label_pos_outResol.astype(np.int)),
#                             cv2.FONT_HERSHEY_DUPLEX, 1, (label_color), 3)

#                 for level in set(cnt_all_levels_outResol.keys()) & set(levels):
#                     cnt_outResol = cnt_all_levels_outResol[level]
#                     cv2.polylines(viz, [cnt_outResol.astype(np.int)], 
#                                   isClosed=True, 
#                                   color=level_colors[set_name][level], 
#                                   thickness=contour_width)

#         vizs_all_sections[sec] = viz

#     return vizs_all_sections


# def annotation_from_multiple_warped_atlases_overlay_on(bg, warped_volumes_sets, stack_fixed, 
#                                                        volume_downsample=None, 
#                                                        volume_resolution=None,
#                                             fn=None, sec=None, orientation='sagittal',
#                             structures=None, out_downsample=None, out_resolution=None,
#                             users=None, level_colors=None, levels=None, show_text=True, label_color=(0,0,0),
#                              contours=None, contour_width=1, bg_img_version='grayJpeg'):
#     """
#     Args:
#         warped_volumes_sets ({set_name: {structure: (3d probability array, (3,)-array origin wrt wholebrain)}})
#         levels (list of float): probability levels at which the contours are drawn.
#         level_colors (dict {set_name: dict {float: (3,)-ndarray of float}}): 256-based contour color for each level for each set
#     """

#     wholebrainXYcropped_origin_wrt_wholebrain = DataManager.get_domain_origin(stack=stack_fixed, 
#                                                                               domain='wholebrainXYcropped',
#                                                                              resolution=volume_resolution)
#     # This is down32 of the raw resolution of the given stack.
    
#     if level_colors is None:
#         level_colors = {set_name: {0.1: (0,255,255),
#                     0.25: (0,255,0),
#                     0.5: (255,0,0),
#                     0.75: (255,255,0),
#                     0.99: (255,0,255)} for set_name in warped_volumes_sets.keys()}

#     if levels is None:
#         levels = level_colors.values()[0].keys()

#     volume_resolution_um = convert_resolution_string_to_voxel_size(resolution=volume_resolution, stack=stack_fixed)
        
#     t = time.time()

#     if bg == 'original':

#         # if out_downsample == 32:
#         #     resol_str = 'thumbnail'
#         # elif out_downsample == 1:
#         #     resol_str = 'lossless'
#         # else:
#         #     resol_str = 'down'+str(out_downsample)

#         out_resolution_um = convert_resolution_string_to_voxel_size(resolution=out_resolution, stack=stack_fixed)
#         if stack_fixed == 'ChatCryoJane201710':
#             out_downsample = out_resolution_um / XY_PIXEL_DISTANCE_LOSSLESS_AXIOSCAN
#         else:
#             out_downsample = out_resolution_um / XY_PIXEL_DISTANCE_LOSSLESS

#         try:
#             bg = DataManager.load_image_v2(stack=stack_fixed, section=sec, fn=fn, resol=out_resolution, prep_id=2, version=bg_img_version)
#         except Exception as e:
#             sys.stderr.write('Cannot load downsampled jpeg, load lossless instead: %s.\n' % e)
#             bg = DataManager.load_image_v2(stack=stack_fixed, section=sec, fn=fn, resol='lossless', prep_id=2, version=bg_img_version)
#             bg = rescale_by_resampling(bg, 1./out_downsample)
                
#     if bg.ndim == 2:
#         bg = gray2rgb(bg)

#     viz = bg.copy()

#     assert orientation == 'sagittal', 'Currently only support drawing on sagittal sections'

#     z_wrt_wholebrain = DataManager.convert_section_to_z(stack=stack_fixed, sec=sec, resolution=volume_resolution, mid=True, z_begin=0)

#     # Find moving volume annotation contours.
#     # for set_name, warped_volumes in warped_volumes_sets.iteritems():
#     #     for name_s, vol in warped_volumes.iteritems():
#     for set_name in warped_volumes_sets.keys(): # This avoids loading entire warped_volumes (maybe?)
#         for name_s, (vol, vol_origin_wrt_wholebrain) in warped_volumes_sets[set_name].iteritems():
#             # structure does not include level z, skip
#             bbox = bbox_3d(vol)
#             zmin_wrt_wholebrain = bbox[4] + vol_origin_wrt_wholebrain[2]
#             zmax_wrt_wholebrain = bbox[5] + vol_origin_wrt_wholebrain[2]
#             # print zmin_wrt_wholebrain, zmax_wrt_wholebrain, z_wrt_wholebrain
#             if z_wrt_wholebrain < zmin_wrt_wholebrain or z_wrt_wholebrain > zmax_wrt_wholebrain:
#                 continue

#             print set_name, name_s

#             label_pos = None

#             for level in levels:
#                 cnts_rc_wrt_vol = find_contours(vol[..., int(np.round(z_wrt_wholebrain - vol_origin_wrt_wholebrain[2]))], level=level)
#                 for cnt_rc_wrt_vol in cnts_rc_wrt_vol:
#                     cnt_wrt_cropped_volRes = cnt_rc_wrt_vol[:,::-1] + (vol_origin_wrt_wholebrain[0], vol_origin_wrt_wholebrain[1]) - wholebrainXYcropped_origin_wrt_wholebrain[:2]
#                     cnt_wrt_cropped_imgRes = cnt_wrt_cropped_volRes * volume_resolution_um / out_resolution_um
#                     cv2.polylines(viz, [cnt_wrt_cropped_imgRes.astype(np.int)],
#                                   True, level_colors[set_name][level], contour_width)

#                     if show_text:
#                         if label_pos is None:
#                             label_pos = np.mean(cnt_wrt_cropped_imgRes, axis=0)

#             # Show text
#             if label_pos is not None:
#                 cv2.putText(viz, name_s, tuple(label_pos.astype(np.int)),
#                         cv2.FONT_HERSHEY_DUPLEX, 1, (label_color), 3)

#     return viz

def annotation_by_human_overlay_on(bg, stack=None, fn=None, sec=None, orientation='sagittal',
                            structures=None, out_downsample=8,
                            users=None, colors=None, show_labels=True,
                             contours=None, timestamp='latest', suffix='contours', return_timestamp=False):
    """
    Draw annotation contours on a user-given background image.

    Args:
        timestamp (str): timestamp of the annnotation file to load.
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
