import sys
import os
import subprocess

from pandas import read_hdf, DataFrame
from datetime import datetime
import re

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from metadata import *
# try:
#     from vis3d_utilities import *
# except:
#     sys.stderr.write("No vtk")
from distributed_utilities import *

use_image_cache = False
image_cache = {}

def get_random_masked_regions(region_shape, stack, num_regions=1, sec=None, fn=None):
    """
    Return a random region that is on mask.

    Args:
        region_shape ((width, height)-tuple):

    Returns:
        list of (region_x, region_y, region_w, region_h)
    """

    if fn is None:
        fn = metadata_cache['sections_to_filenames'][stack][sec]
    tb_mask = DataManager.load_thumbnail_mask_v2(stack=stack, fn=fn)
    img_w, img_h = metadata_cache['image_shape'][stack]
    h, w = region_shape

    regions = []
    for _ in range(num_regions):
        while True:
            xmin = np.random.randint(0, img_w, 1)[0]
            ymin = np.random.randint(0, img_h, 1)[0]

            if xmin + w >= img_w or ymin + h >= img_h:
                continue

            tb_xmin = xmin / 32
            tb_xmax = (xmin + w) / 32
            tb_ymin = ymin / 32
            tb_ymax = (ymin + h) / 32

            if np.count_nonzero(np.r_[tb_mask[tb_ymin, tb_xmin], \
                                      tb_mask[tb_ymin, tb_xmax], \
                                      tb_mask[tb_ymax, tb_xmin], \
                                      tb_mask[tb_ymax, tb_xmax]]) >= 3:
                break
        regions.append((xmin, ymin, w, h))

    return regions


def invert_section_to_filename_mapping(section_to_filename):
    filename_to_section = {fn: sec for sec, fn in section_to_filename.iteritems() if not is_invalid(fn)}
    return filename_to_section

def is_invalid(fn=None, sec=None, stack=None):
    if sec is not None:
        assert stack is not None
        fn = metadata_cache['sections_to_filenames'][stack][sec]
    else:
        assert fn is not None
    return fn in ['Nonexisting', 'Rescan', 'Placeholder']

def volume_type_to_str(t):
    if t == 'score':
        return 'scoreVolume'
    elif t == 'annotation':
        return 'annotationVolume'
    elif t == 'annotationAsScore':
        return 'annotationAsScoreVolume'
    elif t == 'annotationSmoothedAsScore':
        return 'annotationSmoothedAsScoreVolume'
    elif t == 'outer_contour':
        return 'outerContourVolume'
    elif t == 'intensity':
        return 'intensityVolume'
    elif t == 'intensity_metaimage':
        return 'intensityMetaImageVolume'
    else:
        raise Exception('Volume type %s is not recognized.' % t)


################
## Conversion ##
################

class CoordinatesConverter(object):
    def __init__(self, stack=None, section_list=None):
        """
        """
        # A 3-D frame is defined by the following information:
        # - plane: the anatomical name of the 2-D plane spanned by x and y axes.
        # - zdim_um: ??
        # - origin_wrt_wholebrain_um: the origin with respect to wholebrain, in microns.

        # Some are derivable from data_feeder
        # some from stack name
        # some must be assigned dynamically

        self.frames = {'wholebrain': {'origin_wrt_wholebrain_um': (0,0,0),
        'zdim_um': None},
        # 'sagittal': {'origin_wrt_wholebrain_um': None,
        # 'plane': 'sagittal', 'zdim_um': None},
        # 'coronal': {'origin_wrt_wholebrain_um': None,
        # 'plane': 'coronal', 'zdim_um': None},
        # 'horizontal': {'origin_wrt_wholebrain_um': None,
        # 'plane': 'horizontal', 'zdim_um': None},
        }

        self.resolutions = {}

        if stack is not None:
            self.stack = stack

            # Define frame:wholebrainXYcropped
            cropbox_origin_xy_wrt_wholebrain_tbResol = DataManager.load_cropbox_v2(stack=stack, prep_id='alignedBrainstemCrop', only_2d=True)[[0,2]]

            self.derive_three_view_frames(base_frame_name='wholebrainXYcropped',
            origin_wrt_wholebrain_um=np.r_[cropbox_origin_xy_wrt_wholebrain_tbResol, 0] * convert_resolution_string_to_um(resolution='thumbnail', stack=self.stack))

            # Define frame:wholebrainWithMargin
            intensity_volume_spec = dict(name=stack, resolution='10.0um', prep_id='wholebrainWithMargin', vol_type='intensity')
            thumbnail_volume, thumbnail_volume_origin_wrt_wholebrain_10um = DataManager.load_original_volume_v2(intensity_volume_spec, return_origin_instead_of_bbox=True)
            thumbnail_volume_origin_wrt_wholebrain_um = thumbnail_volume_origin_wrt_wholebrain_10um * 10.

            # thumbnail_volume, (thumbnail_volume_origin_wrt_wholebrain_dataResol_x, thumbnail_volume_origin_wrt_wholebrain_dataResol_y, _) = \
            # DataManager.load_original_volume_v2(intensity_volume_spec, return_origin_instead_of_bbox=True)
            # thumbnail_volume_origin_wrt_wholebrain_um = np.r_[thumbnail_volume_origin_wrt_wholebrain_dataResol_x * 10., thumbnail_volume_origin_wrt_wholebrain_dataResol_y * 10., 0.]

            self.derive_three_view_frames(base_frame_name='wholebrainWithMargin',
                                   origin_wrt_wholebrain_um=thumbnail_volume_origin_wrt_wholebrain_um,
                                         zdim_um=thumbnail_volume.shape[2] * 10.)

            # Define resolution:raw
            self.register_new_resolution('raw', convert_resolution_string_to_um(resolution='raw', stack=stack))

        if section_list is not None:
            self.section_list = section_list

    def set_data(self, data_feeder, stack=None):

        self.data_feeder = data_feeder
        self.stack = stack

        for frame_name, frame in self.frames.iteritems():
            if hasattr(self.data_feeder, 'z_dim'):
                frame['zdim_um'] = self.data_feeder.z_dim * convert_resolution_string_to_um(resolution=self.data_feeder.resolution, stack=self.stack)

        self.resolutions['image'] = {'um': convert_resolution_string_to_um(resolution=self.data_feeder.resolution, stack=self.stack)}
        if hasattr(self.data_feeder, 'sections'):
            self.section_list = self.data_feeder.sections

            # cropbox_origin_xy_wrt_wholebrain_tbResol = DataManager.load_cropbox_v2(stack=stack, prep_id=self.prep_id)[[0,2]]
            # self.frames['wholebrainXYcropped'] = dict(origin_wrt_wholebrain_um=np.r_[cropbox_origin_xy_wrt_wholebrain_tbResol, 0] * convert_resolution_string_to_um(resolution='thumbnail', stack=stack),
            # plane='sagittal', 'zdim_um'=None)

    def derive_three_view_frames(self, base_frame_name, origin_wrt_wholebrain_um=(0,0,0), zdim_um=None):
        """
        Generate three new coordinate frames, based on a given bounding box.
        Names of the new frames are <base_frame_name>_sagittal, <base_frame_name>_coronal and <base_frame_name>_horizontal.

        Args:
            base_frame_name (str):
        """

        if base_frame_name == 'data': # define by data feeder
            if hasattr(self.data_feeder, 'z_dim'):
                zdim_um = self.data_feeder.z_dim * convert_resolution_string_to_um(resolution=self.data_feeder.resolution, stack=self.stack)

        self.register_new_frame(base_frame_name, origin_wrt_wholebrain_um, zdim_um)

    def register_new_frame(self, frame_name, origin_wrt_wholebrain_um, zdim_um=None):
        """
        Args:
            frame_name (str): frame identifier
        """
        # assert frame_name not in self.frames, 'Frame name %s already exists.' % frame_name

        if frame_name not in self.frames:
            self.frames[frame_name] = {'origin_wrt_wholebrain_um': None, 'zdim_um': None}

        if zdim_um is not None:
            self.frames[frame_name]['zdim_um'] = zdim_um

        if origin_wrt_wholebrain_um is not None:
            self.frames[frame_name]['origin_wrt_wholebrain_um'] = origin_wrt_wholebrain_um

        # if plane is not None:
        #     self.frames[frame_name]['plane'] = plane

    def register_new_resolution(self, resol_name, resol_um):
        """
        Args:
            resol_name (str): resolution identifier
            resol_um (float): pixel/voxel size in micron
        """
        # assert resol_name not in self.resolutions, 'Resolution name %s already exists.' % resol_name
        self.resolutions[resol_name] = {'um': resol_um}

    def get_resolution_um(self, resol_name):
        
        if resol_name in self.resolutions:
            res_um = self.resolutions[resol_name]['um']
        else:
            res_um = convert_resolution_string_to_um(resolution=resol_name, stack=self.stack)
        return res_um
        
    def convert_three_view_frames(self, p, base_frame_name, in_plane, out_plane, p_resol):
        """
        Convert among the three frames specified by the second method in this presentation
        https://docs.google.com/presentation/d/1o5aQbXY5wYC0BNNiEZm7qmjvngbD_dVoMyCw_tAQrkQ/edit#slide=id.g2d31ede24d_0_0

        Args:
            in_plane (str): one of sagittal, coronal and horizontal
            out_plane (str): one of sagittal, coronal and horizontal
        """

        if in_plane == 'coronal' or in_plane == 'horizontal' or out_plane == 'coronal' or out_plane == 'horizontal':
            zdim_um = self.frames[base_frame_name]['zdim_um']
            zdim = zdim_um / convert_resolution_string_to_um(resolution=p_resol)

        if in_plane == 'sagittal':
            p_sagittal = p
        elif in_plane == 'coronal':
            x = p[..., 2]
            y = p[..., 1]
            z = zdim - p[..., 0]
            p_sagittal = np.column_stack([x,y,z])
        elif in_plane == 'horizontal':
            x = p[..., 0]
            y = p[..., 2]
            z = zdim - p[..., 1]
            p_sagittal = np.column_stack([x,y,z])
        else:
            raise Exception("Plane %s is not recognized." % in_plane)

        if out_plane == 'sagittal':
            p_out = p_sagittal
        elif out_plane == 'coronal':
            x = zdim - p_sagittal[..., 2]
            y = p_sagittal[..., 1]
            z = p_sagittal[..., 0]
            p_out = np.column_stack([x,y,z])
        elif out_plane == 'horizontal':
            x = p_sagittal[..., 0]
            y = zdim - p_sagittal[..., 2]
            z = p_sagittal[..., 1]
            p_out = np.column_stack([x,y,z])
        else:
            raise Exception("Plane %s is not recognized." % out_plane)

        return p_out

    def convert_resolution(self, p, in_resolution, out_resolution):
        """
        Rescales coordinates according to the given input and output resolution.
        This function does not change physical position of coordinate origin or the direction of the axes.
        """

        p = np.array(p)
        if p.ndim != 2:
            print p, in_resolution, out_resolution
        assert p.ndim == 2

        
        import re
        m = re.search('^(.*?)_(.*?)_(.*?)$', in_resolution)
        if m is not None:
            in_x_resol, in_y_resol, in_z_resol = m.groups()
            assert in_x_resol == in_y_resol                    
            uv_um = p[..., :2] * self.get_resolution_um(resol_name=in_x_resol)
            d_um = np.array([SECTION_THICKNESS * (sec - 0.5) for sec in p[..., 2]])
            p_um = np.column_stack([uv_um, d_um])
        else:
            if in_resolution == 'image':
                p_um = p * self.resolutions['image']['um']

            elif in_resolution == 'image_image_index':
                uv_um = p[..., :2] * self.get_resolution_um(resol_name='image')
                i_um = np.array([SECTION_THICKNESS * (self.section_list[int(idx)] - 0.5) for idx in p[..., 2]])
                p_um = np.column_stack([uv_um, i_um])
            elif in_resolution == 'section':
                uv_um = np.array([(np.nan, np.nan) for _ in p])
                # d_um = np.array([SECTION_THICKNESS * (sec - 0.5) for sec in p])
                d_um = SECTION_THICKNESS * (p[:, 0] - 0.5)
                p_um = np.column_stack([np.atleast_2d(uv_um), np.atleast_1d(d_um)])
            elif in_resolution == 'index':
                uv_um = np.array([(np.nan, np.nan) for _ in p])
                # i_um = np.array([SECTION_THICKNESS * (self.section_list[int(idx)] - 0.5) for idx in p])
                i_um = SECTION_THICKNESS * (np.array(self.section_list)[p[:,0].astype(np.int)] - 0.5)
                p_um = np.column_stack([uv_um, i_um])
            else:
                if in_resolution in self.resolutions:
                    p_um = p * self.resolutions[in_resolution]['um']
                else:
                    p_um = p * convert_resolution_string_to_um(resolution=in_resolution, stack=self.stack)

                    
        m = re.search('^(.*?)_(.*?)_(.*?)$', out_resolution)
        if m is not None:
            out_x_resol, out_y_resol, out_z_resol = m.groups()
            assert out_x_resol == out_y_resol                    
            uv_outResol = p_um[..., :2] / self.get_resolution_um(resol_name=out_x_resol)
            sec_outResol = np.array([1 + int(np.floor(d_um / SECTION_THICKNESS)) for d_um in np.atleast_1d(p_um[..., 2])])
            p_outResol = np.column_stack([np.atleast_2d(uv_outResol), np.atleast_1d(sec_outResol)])
        else:
            if out_resolution == 'image':
                p_outResol = p_um / self.resolutions['image']['um']
            # elif out_resolution == 'image_image_section':
            #     uv_outResol = p_um[..., :2] / self.resolutions['image']['um']
            #     sec_outResol = np.array([1 + int(np.floor(d_um / SECTION_THICKNESS)) for d_um in np.atleast_1d(p_um[..., 2])])
            #     p_outResol = np.column_stack([np.atleast_2d(uv_outResol), np.atleast_1d(sec_outResol)])
            elif out_resolution == 'image_image_index':
                uv_outResol = p_um[..., :2] / self.get_resolution_um(resol_name='image')
                if hasattr(self, 'section_list'):
                    i_outResol = []
                    for d_um in p_um[..., 2]:
                        sec = 1 + int(np.floor(d_um / SECTION_THICKNESS))
                        if sec in self.section_list:
                            index = self.section_list.index(sec)
                        else:
                            index = np.nan
                        i_outResol.append(index)
                    i_outResol = np.array(i_outResol)
                else:
                    i_outResol = p_um[..., 2] / self.resolutions['image']['um']
                p_outResol = np.column_stack([uv_outResol, i_outResol])
            elif out_resolution == 'section':
                uv_outResol = p_um[..., :2] / self.resolutions['image']['um']
                sec_outResol = np.array([1 + int(np.floor(d_um / SECTION_THICKNESS)) for d_um in np.atleast_1d(p_um[..., 2])])
                p_outResol = np.column_stack([np.atleast_2d(uv_outResol), np.atleast_1d(sec_outResol)])[..., 2][:, None]
            elif out_resolution == 'index':
                uv_outResol = np.array([(np.nan, np.nan) for _ in p_um])
                # uv_outResol = p_um[..., :2] / self.resolutions['image']['um']
                if hasattr(self, 'section_list'):
                    i_outResol = []
                    for d_um in p_um[..., 2]:
                        sec = 1 + int(np.floor(d_um / SECTION_THICKNESS))
                        if sec in self.section_list:
                            index = self.section_list.index(sec)
                        else:
                            index = np.nan
                        i_outResol.append(index)
                    i_outResol = np.array(i_outResol)
                else:
                    i_outResol = p_um[..., 2] / self.resolutions['image']['um']
                p_outResol = np.column_stack([uv_outResol, i_outResol])[..., 2][:, None]
            else:
                if out_resolution in self.resolutions:
                    p_outResol = p_um / self.resolutions[out_resolution]['um']
                else:
                    p_outResol = p_um / convert_resolution_string_to_um(resolution=out_resolution, stack=self.stack)

        assert p_outResol.ndim == 2

        return p_outResol

    def convert_from_wholebrain_um(self, p_wrt_wholebrain_um, wrt, resolution):
        """
        Convert the coordinates expressed in "wholebrain" frame in microns to
        coordinates expressed in the given frame and resolution.

        Args:
            p_wrt_wholebrain_um (list of 3-tuples): list of points
            wrt (str): name of output frame
            resolution (str): name of output resolution.
        """

        p_wrt_wholebrain_um = np.array(p_wrt_wholebrain_um)
        # assert np.atleast_2d(p_wrt_wholebrain_um).shape[1] == 3, "Coordinates of each point must have three elements."
        assert p_wrt_wholebrain_um.ndim == 2

        if wrt == 'wholebrain':
            p_wrt_outdomain_um = p_wrt_wholebrain_um
        else:
            assert isinstance(wrt, tuple)
            base_frame_name, plane = wrt
            p_wrt_outSagittal_origin_um = p_wrt_wholebrain_um - self.frames[base_frame_name]['origin_wrt_wholebrain_um']
            # print wrt, 'origin_wrt_wholebrain_um', self.frames[base_frame_name]['origin_wrt_wholebrain_um']
            # print 'p_wrt_outSagittal_origin_um', np.nanmean(p_wrt_outSagittal_origin_um[None, :], axis=0)
            assert p_wrt_outSagittal_origin_um.ndim == 2
            p_wrt_outdomain_um = self.convert_three_view_frames(p=p_wrt_outSagittal_origin_um, base_frame_name=base_frame_name,
                                                                in_plane='sagittal',
                                                                out_plane=plane,
                                                                p_resol='um')

        assert p_wrt_outdomain_um.ndim == 2
        p_wrt_outdomain_outResol = self.convert_resolution(p_wrt_outdomain_um, in_resolution='um', out_resolution=resolution)
        assert p_wrt_outdomain_outResol.ndim == 2
        return p_wrt_outdomain_outResol

    def convert_to_wholebrain_um(self, p, wrt, resolution):
        """
        Convert the coordinates expressed in given frame and resolution to
        coordinates expressed in "wholebrain" frame in microns.

        Args:
            p (list of 3-tuples): list of points
            wrt (str): name of input frame
            resolution (str): name of input resolution.
        """

        p = np.array(p)
        # assert np.atleast_2d(p).shape[1] == 3, "Coordinates must have three elements."
        p_um = self.convert_resolution(p, in_resolution=resolution, out_resolution='um')

        if wrt == 'wholebrain':
            p_wrt_wholebrain_um = p_um
        else:
            assert isinstance(wrt, tuple)
            base_frame_name, plane = wrt

            # print 'p_um', np.nanmean(p_um[None, :], axis=0)
            assert p_um.ndim == 2
            p_wrt_inSagittal_um = self.convert_three_view_frames(p=p_um, base_frame_name=base_frame_name,
                                                                in_plane=plane,
                                                                out_plane='sagittal',
                                                                p_resol='um')
            assert p_wrt_inSagittal_um.ndim == 2
            # print 'p_wrt_inSagittal_um', np.nanmean(p_wrt_inSagittal_um[None, :], axis=0)
            inSagittal_origin_wrt_wholebrain_um = self.frames[base_frame_name]['origin_wrt_wholebrain_um']
            # print 'inSagittal_origin_wrt_wholebrain_um', np.nanmean(inSagittal_origin_wrt_wholebrain_um[None, :], axis=0)
            p_wrt_wholebrain_um = p_wrt_inSagittal_um + inSagittal_origin_wrt_wholebrain_um
            # print 'p_wrt_wholebrain_um', np.nanmean(p_wrt_wholebrain_um[None, :], axis=0)

        return p_wrt_wholebrain_um

    def convert_frame_and_resolution(self, p, in_wrt, in_resolution, out_wrt, out_resolution,
                                     stack=None):
        """
        Converts between coordinates that are expressed in different frames and different resolutions.

        Use this in combination with DataManager.get_domain_origin().

        `wrt` can be either 3-D frames or 2-D frames.
        Detailed definitions of various frames can be found at https://goo.gl/o2Yydw.

        There are two ways to specify 3-D frames.

        1. The "absolute" way:
        - wholebrain: formed by stacking all sections of prep1 (aligned + padded) images
        - wholebrainXYcropped: formed by stacking all sections of prep2 images
        - brainstemXYfull: formed by stacking sections of prep1 images that contain brainstem
        - brainstem: formed by stacking brainstem sections of prep2 images
        - brainstemXYFullNoMargin: formed by stacking brainstem sections of prep4 images

        2. The "relative" way:
        - x_sagittal: frame of lo-res sagittal scene = sagittal frame of the intensity volume, with origin at the most left/rostral/dorsal position.
        - x_coronal: frame of lo-res coronal scene = coronal frame of the intensity volume, with origin at the most left/rostral/dorsal position.
        - x_horizontal: frame of lo-res horizontal scene = horizontal frame of the intensity volume, with origin at the most left/rostral/dorsal position.

        Build-in 2-D frames include:
        - {0: 'original', 1: 'alignedPadded', 2: 'alignedCroppedBrainstem', 3: 'alignedCroppedThalamus', 4: 'alignedNoMargin', 5: 'alignedWithMargin', 6: 'originalCropped'}

        Resolution specifies the physical units of the coordinate axes.
        Build-in `resolution` for 3-D coordinates can be any of these strings:
        - raw
        - down32
        - vol
        - image: gscene resolution, determined by data_feeder.resolution
        - raw_raw_index: (u in raw resolution, v in raw resolution, i in terms of data_feeder index)
        - image_image_index: (u in image resolution, v in image resolution, i in terms of data_feeder index)
        - image_image_section: (u in image resolution, v in image resolution, i in terms of section index)
        """

        if in_wrt == 'original' and out_wrt == 'alignedPadded':
            
            in_x_resol, in_y_resol, in_z_resol = in_resolution.split('_')
            assert in_x_resol == in_y_resol
            assert in_z_resol == 'section'
            in_image_resolution = in_x_resol
            
            out_x_resol, out_y_resol, out_z_resol = out_resolution.split('_')
            assert out_x_resol == out_y_resol
            assert out_z_resol == 'section'
            out_image_resolution = out_x_resol

            uv_um = p[..., :2] * convert_resolution_string_to_um(stack=stack, resolution=in_image_resolution)

            p_wrt_outdomain_outResol = np.zeros(p.shape)

            Ts_anchor_to_individual_section_image_resol = DataManager.load_transforms(stack=stack, resolution='1um', use_inverse=True)

            different_sections = np.unique(p[:, 2])
            for sec in different_sections:
                curr_section_mask = p[:, 2] == sec
                fn = metadata_cache['sections_to_filenames'][stack][sec]
                T_anchor_to_individual_section_image_resol = Ts_anchor_to_individual_section_image_resol[fn]
                uv_wrt_alignedPadded_um_curr_section = np.dot(T_anchor_to_individual_section_image_resol,
                                          np.c_[uv_um[curr_section_mask, :2],
                                                np.ones((np.count_nonzero(curr_section_mask),))].T).T[:, :2]

                uv_wrt_alignedPadded_outResol_curr_section = \
                uv_wrt_alignedPadded_um_curr_section / convert_resolution_string_to_um(stack=stack, resolution=out_image_resolution)

                p_wrt_outdomain_outResol[curr_section_mask] = \
                np.column_stack([uv_wrt_alignedPadded_outResol_curr_section,
                           sec * np.ones((len(uv_wrt_alignedPadded_outResol_curr_section),))])

            return p_wrt_outdomain_outResol
                
        elif in_wrt == 'alignedPadded' and out_wrt == 'original':

            
            in_x_resol, in_y_resol, in_z_resol = in_resolution.split('_')
            assert in_x_resol == in_y_resol
            assert in_z_resol == 'section'
            in_image_resolution = in_x_resol
            
            out_x_resol, out_y_resol, out_z_resol = out_resolution.split('_')
            assert out_x_resol == out_y_resol
            assert out_z_resol == 'section'
            out_image_resolution = out_x_resol
            
            uv_um = p[..., :2] * convert_resolution_string_to_um(stack=stack, resolution=in_image_resolution)

            p_wrt_outdomain_outResol = np.zeros(p.shape)

            Ts_anchor_to_individual_section_image_resol = DataManager.load_transforms(stack=stack, resolution='1um', use_inverse=True)
            Ts_anchor_to_individual_section_image_resol = {fn: np.linalg.inv(T) for fn, T in Ts_anchor_to_individual_section_image_resol.iteritems()}

            different_sections = np.unique(p[:, 2])
            for sec in different_sections:
                curr_section_mask = p[:, 2] == sec
                fn = metadata_cache['sections_to_filenames'][stack][sec]
                T_anchor_to_individual_section_image_resol = Ts_anchor_to_individual_section_image_resol[fn]
                uv_wrt_alignedPadded_um_curr_section = np.dot(T_anchor_to_individual_section_image_resol,
                                          np.c_[uv_um[curr_section_mask, :2],
                                                np.ones((np.count_nonzero(curr_section_mask),))].T).T[:, :2]

                uv_wrt_alignedPadded_outResol_curr_section = \
                uv_wrt_alignedPadded_um_curr_section / convert_resolution_string_to_um(stack=stack, resolution=out_image_resolution)

                p_wrt_outdomain_outResol[curr_section_mask] = \
                np.column_stack([uv_wrt_alignedPadded_outResol_curr_section,
                           sec * np.ones((len(uv_wrt_alignedPadded_outResol_curr_section),))])

            return p_wrt_outdomain_outResol

        else:
            p = np.array(p)
            assert p.ndim == 2
            # print 'p', np.nanmean(p[None,:], axis=0)
            p_wrt_wholebrain_um = self.convert_to_wholebrain_um(p, wrt=in_wrt, resolution=in_resolution)
            assert p_wrt_wholebrain_um.ndim == 2
            # print 'p_wrt_wholebrain_um', np.nanmean(p_wrt_wholebrain_um[None,:], axis=0)
            p_wrt_outdomain_outResol = self.convert_from_wholebrain_um(p_wrt_wholebrain_um=p_wrt_wholebrain_um, wrt=out_wrt, resolution=out_resolution)
            # print 'p_wrt_outdomain_outResol', np.nanmean(p_wrt_outdomain_outResol[None,:], axis=0)
            # print
            return p_wrt_outdomain_outResol

from skimage.transform import resize

def images_to_volume_v2(images, spacing_um, in_resol_um, out_resol_um, crop_to_minimal=True):
    """
    Stack images in parallel at specified z positions to form volume.

    Args:
        images (dict of 2D images): key is section index. First section has index 1.
        spacing_um (float): spacing between adjacent sections or thickness of each section, in micron.
        in_resol_um (float): image planar resolution in micron.
        out_resol_um (float): isotropic output voxel size, in micron.

    Returns:
        (volume, volume origin relative to the image origin of section 1)
    """

    if isinstance(images, dict):

        shapes = np.array([im.shape[:2] for im in images.values()])
        assert len(np.unique(shapes[:,0])) == 1, 'Height of all images must be the same.'
        assert len(np.unique(shapes[:,1])) == 1, 'Width of all images must be the same.'

        ydim, xdim = map(int, np.ceil(shapes[0] * float(in_resol_um) / out_resol_um))
        sections = sorted(images.keys())
        # if last_sec is None:
        #     last_sec = np.max(sections)
        # if first_sec is None:
        #     first_sec = np.min(sections)
    elif callable(images):
        try:
            ydim, xdim = images(100).shape[:2]
        except:
            ydim, xdim = images(200).shape[:2]
        # assert last_sec is not None
        # assert first_sec is not None
    else:
        raise Exception('images must be dict or function.')

    voxel_z_size = float(spacing_um) / out_resol_um
    zdim = int(np.ceil(np.max(sections) * voxel_z_size)) + 1

    dtype = images.values()[0].dtype
    volume = np.zeros((ydim, xdim, zdim), dtype)

    for i in range(len(sections)-1):
        # z1 = int(np.floor((sections[i]-1) * voxel_z_size))
        # z2 = int(np.ceil(sections[i+1] * voxel_z_size))
        z1 = int(np.round((sections[i]-1) * voxel_z_size))
        z2 = int(np.round(sections[i+1] * voxel_z_size))
        if isinstance(images, dict):
            im = images[sections[i]]
        elif callable(images):
            im = images(sections[i])

        if dtype == np.uint8:
            volume[:, :, z1:z2+1] = img_as_ubyte(resize(im, (ydim, xdim)))[..., None]
            # assert in_resol_um == out_resol_um
            # volume[:, :, z1:z2+1] = im[..., None]
        else:
            volume[:, :, z1:z2+1] = resize(im, (ydim, xdim))[..., None]
        # else:
        #     raise Exception("dtype must be uint8 ot float32")

    if crop_to_minimal:
        return crop_volume_to_minimal(volume)
    else:
        return volume

# def get_prep_str(prep_id):
#     return prep_id_to_str[prep_id]

class DataManager(object):

    ################################################
    ##   Conversion between coordinate systems    ##
    ################################################

    @staticmethod
    def get_crop_bbox_rel2uncropped(stack):
        """
        Returns the bounding box of domain "brainstem" wrt domain "wholebrain".
        This assumes resolution of "down32".
        """

        crop_xmin_rel2uncropped, crop_xmax_rel2uncropped, \
        crop_ymin_rel2uncropped, crop_ymax_rel2uncropped, \
        = metadata_cache['cropbox'][stack]

        s1, s2 = metadata_cache['section_limits'][stack]
        crop_zmin_rel2uncropped = int(np.floor(np.mean(DataManager.convert_section_to_z(stack=stack, sec=s1, downsample=32, z_begin=0))))
        crop_zmax_rel2uncropped = int(np.ceil(np.mean(DataManager.convert_section_to_z(stack=stack, sec=s2, downsample=32, z_begin=0))))

        crop_bbox_rel2uncropped = \
        np.array([crop_xmin_rel2uncropped, crop_xmax_rel2uncropped, \
        crop_ymin_rel2uncropped, crop_ymax_rel2uncropped, \
        crop_zmin_rel2uncropped, crop_zmax_rel2uncropped])
        return crop_bbox_rel2uncropped

    @staticmethod
    def get_score_bbox_rel2uncropped(stack):
        """
        Returns the bounding box of score volume wrt domain "wholebrain".
        """

        score_vol_f_xmin_rel2cropped, score_vol_f_xmax_rel2cropped, \
        score_vol_f_ymin_rel2cropped, score_vol_f_ymax_rel2cropped, \
        score_vol_f_zmin_rel2uncropped, score_vol_f_zmax_rel2uncropped, \
        = DataManager.load_original_volume_bbox(stack=stack, volume_type='score',
                                              structure='7N', downscale=32, prep_id=2, detector_id=15)

        s1, s2 = metadata_cache['section_limits'][stack]
        crop_zmin_rel2uncropped = int(np.floor(np.mean(DataManager.convert_section_to_z(stack=stack, sec=s1, downsample=32, z_begin=0))))
        crop_zmax_rel2uncropped = int(np.ceil(np.mean(DataManager.convert_section_to_z(stack=stack, sec=s2, downsample=32, z_begin=0))))

        score_vol_f_zmin_rel2cropped = score_vol_f_zmin_rel2uncropped - crop_zmin_rel2uncropped
        score_vol_f_zmax_rel2cropped = score_vol_f_zmax_rel2uncropped - crop_zmax_rel2uncropped

        score_vol_f_bbox_rel2cropped = np.array([score_vol_f_xmin_rel2cropped, score_vol_f_xmax_rel2cropped, \
        score_vol_f_ymin_rel2cropped, score_vol_f_ymax_rel2cropped, \
        score_vol_f_zmin_rel2cropped, score_vol_f_zmax_rel2cropped,])

        return score_vol_f_bbox_rel2cropped


    ########################
    ##   Stacy's data    ##
    ########################

    @staticmethod
    def get_stacy_markers_filepath(stack, structure):
        return os.path.join(ROOT_DIR, 'stacy_data', 'markers', stack, stack + '_markers_%s.bp' % structure)

    ########################
    ##   Lauren's data    ##
    ########################

    @staticmethod
    def get_lauren_markers_filepath(stack, structure, resolution):
        return os.path.join(ROOT_DIR, 'lauren_data', 'markers', stack, stack + '_markers_%s_%s.bp' % (resolution, structure))

    ##############
    ##   SPM    ##
    ##############

    @staticmethod
    def get_spm_histograms_filepath(stack, level, section=None, fn=None):
        """
        Args:
            level (int): 0, 1 or 2
        """
        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][section]
        return os.path.join(CSHL_SPM_ROOTDIR, 'sift_histograms', stack, fn + '_sift_histograms_l%d.bp' % level)

    @staticmethod
    def get_sift_descriptor_vocabulary_filepath():
        """
        Return a sklearn.KMeans classifier object.
        """
        return os.path.join(CSHL_SPM_ROOTDIR, 'sift_vocabulary.clf')

    @staticmethod
    def get_sift_descriptors_labelmap_filepath(stack, section=None, fn=None):
        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][section]
        return os.path.join(CSHL_SPM_ROOTDIR, 'sift_labelmap', stack, fn + '_sift_labelmap.bp')

    @staticmethod
    def get_sift_descriptors_filepath(stack, section=None, fn=None):
        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][section]
        return os.path.join(CSHL_SPM_ROOTDIR, 'sift_descriptors', stack, fn + '_sift_descriptors.bp')

    @staticmethod
    def get_sift_keypoints_filepath(stack, section=None, fn=None):
        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][section]
        return os.path.join(CSHL_SPM_ROOTDIR, 'sift_keypoints', stack, fn + '_sift_keypoints.bp')


    ##########################
    ###    Annotation    #####
    ##########################

    # @staticmethod
    # def get_structure_pose_corrections(stack, stack_m=None,
    #                             detector_id_m=None,
    #                             detector_id_f=None,
    #                             warp_setting=None, trial_idx=None):
    #     basename = DataManager.get_warped_volume_basename(**locals())
    #     fp = os.path.join(ANNOTATION_ROOTDIR, stack, basename + '_' + 'structure3d_corrections' + '.pkl')
    #     return fp

    @staticmethod
    def get_annotated_structures(stack):
        """
        Return existing structures on every section in annotation.
        """
        contours, _ = load_annotation_v3(stack, annotation_rootdir=ANNOTATION_ROOTDIR)
        annotated_structures = {sec: list(set(contours[contours['section']==sec]['name']))
                                for sec in range(first_sec, last_sec+1)}
        return annotated_structures

    @staticmethod
    def load_annotation_to_grid_indices_lookup(stack, win_id, by_human, stack_m='atlasV5',
                                                detector_id_m=None,
                                                detector_id_f=None,
                                               prep_id_m=None,
                                               prep_id_f=2,
                                                warp_setting=17, trial_idx=None, timestamp=None,
                                              return_locations=False, suffix=None):

        from learning_utilities import grid_parameters_to_sample_locations

        grid_indices_lookup_fp = DataManager.get_annotation_to_grid_indices_lookup_filepath(**locals())
        download_from_s3(grid_indices_lookup_fp)

        if not os.path.exists(grid_indices_lookup_fp):
            raise Exception("Do not find structure to grid indices lookup file. Please generate it using `generate_annotation_to_grid_indices_lookup`\
         in notebook `learning/identify_patch_class_from_labeling`")
        else:
            grid_indices_lookup = load_hdf_v2(grid_indices_lookup_fp)

            locations_lookup = defaultdict(lambda: defaultdict(list))

            if not return_locations:
                return grid_indices_lookup
            else:
                grids_to_locations = grid_parameters_to_sample_locations(win_id=win_id, stack=stack)

                for sec, indices_this_sec in grid_indices_lookup.iterrows():
                    for label, indices in indices_this_sec.dropna(how='all').iteritems():
                        locations_lookup[label][sec] = [grids_to_locations[i] for i in indices]

            return DataFrame(locations_lookup)

    @staticmethod
    def get_annotation_to_grid_indices_lookup_filepath(stack, win_id, by_human, stack_m='atlasV5',
                                                       detector_id_m=None, detector_id_f=None,
                                                       prep_id_m=None, prep_id_f=2,
                                                       warp_setting=17, trial_idx=None, timestamp=None,
                                                       suffix=None, **kwargs):

        if timestamp is not None:
            if timestamp == 'latest':
                if by_human:
                    download_from_s3(os.path.join(ANNOTATION_ROOTDIR, stack), is_dir=True,
                                 include_only="*win%(win_id)d*grid_indices_lookup*" % {'win_id':win_id}, redownload=True)
                else:
                    download_from_s3(os.path.join(ANNOTATION_ROOTDIR, stack), is_dir=True,
                                 include_only="*win%(win_id)d*warp*grid_indices_lookup*" % {'win_id':win_id}, redownload=True)
                timestamps = []
                for fn in os.listdir(os.path.join(ANNOTATION_ROOTDIR, stack)):
                    if 'lookup' not in fn:
                        continue

                    ts = None
                    if by_human:
                        if suffix is not None:
                            m = re.match('%(stack)s_annotation_%(suffix)s_(.*)_win%(win_id)d_grid_indices_lookup.hdf' % {'stack':stack, 'win_id':win_id, 'suffix':suffix}, fn)
                            if m is not None:
                                ts = m.groups()[0]
                        else:
                            m = re.match('%(stack)s_annotation_win%(win_id)d_(.*)_grid_indices_lookup.hdf' % {'stack':stack, 'win_id':win_id}, fn)
                            if m is not None:
                                ts = m.groups()[0]
                    else:
                        m = re.match('%(stack)s_annotation_win%(win_id)d_(.*)_grid_indices_lookup.hdf' % {'stack':stack, 'win_id':win_id}, fn)
                        if m is not None:
                            ts = m.groups()[0]

                    if ts is None:
                        print fn
                    timestamps.append((datetime.strptime(ts, '%m%d%Y%H%M%S'), ts))
                assert len(timestamps) > 0, 'No annotation files can be found.'
                timestamp = sorted(timestamps)[-1][1]
                print "latest timestamp: ", timestamp
            elif timestamp == 'now':
                timestamp = datetime.now().strftime("%m%d%Y%H%M%S")

        if by_human:

            if suffix is not None:
                fp = os.path.join(ANNOTATION_ROOTDIR, stack, '%(stack)s_annotation_%(suffix)s_%(timestamp)s_win%(win)d_grid_indices_lookup.hdf' % {'stack':stack, 'win':win_id, 'timestamp':timestamp, 'suffix':suffix})
            else:
                fp = os.path.join(ANNOTATION_ROOTDIR, stack, '%(stack)s_annotation_win%(win)d_%(timestamp)s_grid_indices_lookup.hdf' % {'stack':stack, 'win':win_id, 'timestamp':timestamp})
        else:

            basename = DataManager.get_warped_volume_basename(stack_m=stack_m, stack_f=stack,
                                                              detector_id_m=detector_id_m,
                                                              detector_id_f=detector_id_f,
                                                              prep_id_m=prep_id_m,
                                                              prep_id_f=prep_id_f,
                                                              warp_setting=warp_setting,
                                                              trial_idx=trial_idx)
            fp = os.path.join(ANNOTATION_ROOTDIR, stack, '%(stack)s_annotation_%(basename)s_win%(win)d_%(timestamp)s_grid_indices_lookup.hdf' % {'stack':stack, 'basename': basename, 'win':win_id, 'timestamp':timestamp})
        return fp

    @staticmethod
    def get_annotation_filepath(stack, by_human, stack_m=None,
                                detector_id_m=None,
                                detector_id_f=None,
                                prep_id_m=None,
                                prep_id_f=None,
                                warp_setting=None, trial_idx=None, suffix=None, timestamp=None,
                               return_timestamp=False,
                               annotation_rootdir=ANNOTATION_ROOTDIR,
                               download_s3=True):
        """
        Return the annotation file path.

        Args:
            timestamp (str): can be "latest".
            return_timestamp (bool)
        Returns:
            fp
            timestamp (str): actual timestamp
        """


        if by_human:
            # if suffix is None:
            #     fp = os.path.join(ANNOTATION_ROOTDIR, stack, '%(stack)s_annotation_v3.h5' % {'stack':stack})
            # else:
            if timestamp is not None:
                if timestamp == 'latest':
                    if download_s3:
                        download_from_s3(os.path.join(annotation_rootdir, stack), is_dir=True, include_only="*%s*" % suffix, redownload=True)
                    timestamps = []
                    for fn in os.listdir(os.path.join(annotation_rootdir, stack)):
                        m = re.match('%(stack)s_annotation_%(suffix)s_([0-9]*?).hdf' % {'stack':stack, 'suffix': suffix}, fn)
                        # print fn, m
                        if m is not None:
                            ts = m.groups()[0]
                            timestamps.append((datetime.strptime(ts, '%m%d%Y%H%M%S'), ts))
                    assert len(timestamps) > 0
                    timestamp = sorted(timestamps)[-1][1]
                    print "latest timestamp: ", timestamp
                elif timestamp == 'now':
                    timestamp = datetime.now().strftime("%m%d%Y%H%M%S")

                fp = os.path.join(annotation_rootdir, stack, '%(stack)s_annotation_%(suffix)s_%(timestamp)s.hdf' % {'stack':stack, 'suffix':suffix, 'timestamp': timestamp})
            else:
                fp = os.path.join(annotation_rootdir, stack, '%(stack)s_annotation_%(suffix)s.hdf' % {'stack':stack, 'suffix':suffix})
        else:
            basename = DataManager.get_warped_volume_basename(stack_m=stack_m, stack_f=stack,
                                                              detector_id_m=detector_id_m,
                                                              detector_id_f=detector_id_f,
                                                              prep_id_m=prep_id_m,
                                                              prep_id_f=prep_id_f,
                                                              warp_setting=warp_setting,
                                                              trial_idx=trial_idx)
            if suffix is not None:
                if timestamp is not None:
                    if timestamp == 'latest':
                        if download_s3:
                            download_from_s3(os.path.join(annotation_rootdir, stack), is_dir=True, include_only="*%s*"%suffix, redownload=True)
                        timestamps = []
                        for fn in os.listdir(os.path.join(annotation_rootdir, stack)):
                            m = re.match('%(stack)s_annotation_%(suffix)s_(.*?).hdf' % {'stack':stack, 'suffix': suffix}, fn)
                            if m is not None:
                                ts = m.groups()[0]
                                timestamps.append((datetime.strptime(ts, '%m%d%Y%H%M%S'), ts))
                        assert len(timestamps) > 0
                        timestamp = sorted(timestamps)[-1][1]
                        print "latest timestamp: ", timestamp
                    elif timestamp == 'now':
                        timestamp = datetime.now().strftime("%m%d%Y%H%M%S")

                    fp = os.path.join(annotation_rootdir, stack, 'annotation_%(basename)s_%(suffix)s_%(timestamp)s.hdf' % {'basename': basename, 'suffix': suffix, 'timestamp': timestamp})
                else:
                    fp = os.path.join(annotation_rootdir, stack, 'annotation_%(basename)s_%(suffix)s.hdf' % {'basename': basename, 'suffix': suffix})
            else:
                fp = os.path.join(annotation_rootdir, stack, 'annotation_%(basename)s.hdf' % {'basename': basename})

        if return_timestamp:
            return fp, timestamp
        else:
            return fp

    @staticmethod
    def load_annotation_v4(stack=None, by_human=True, stack_m=None,
                                detector_id_m=None,
                                detector_id_f=None,
                                warp_setting=None, trial_idx=None, timestamp=None, suffix=None,
                          return_timestamp=False,
                          annotation_rootdir=ANNOTATION_ROOTDIR,
                          download_s3=True):
        if by_human:
            if return_timestamp:
                fp, timestamp = DataManager.get_annotation_filepath(stack, by_human=True, suffix=suffix, timestamp=timestamp,
                                                    return_timestamp=True, annotation_rootdir=annotation_rootdir,
                                                                   download_s3=download_s3)
            else:
                fp = DataManager.get_annotation_filepath(stack, by_human=True, suffix=suffix, timestamp=timestamp,
                                                    return_timestamp=False, annotation_rootdir=annotation_rootdir,
                                                        download_s3=download_s3)
            if download_s3:
                download_from_s3(fp)
            contour_df = read_hdf(fp)
            if return_timestamp:
                return contour_df, timestamp
            else:
                return contour_df

        else:
            if return_timestamp:
                fp, timestamp = DataManager.get_annotation_filepath(stack, by_human=False,
                                                     stack_m=stack_m,
                                                      detector_id_m=detector_id_m,
                                                      detector_id_f=detector_id_f,
                                                      warp_setting=warp_setting, trial_idx=trial_idx,
                                                    suffix=suffix, timestamp=timestamp,
                                                                   return_timestamp=True,
                                                                   annotation_rootdir=annotation_rootdir,
                                                                   download_s3=download_s3)
            else:
                fp = DataManager.get_annotation_filepath(stack, by_human=False,
                                     stack_m=stack_m,
                                      detector_id_m=detector_id_m,
                                      detector_id_f=detector_id_f,
                                      warp_setting=warp_setting, trial_idx=trial_idx,
                                    suffix=suffix, timestamp=timestamp,
                                                   return_timestamp=False,
                                                   annotation_rootdir=annotation_rootdir,
                                                        download_s3=download_s3)
            if download_s3:
                download_from_s3(fp)
            annotation_df = load_hdf_v2(fp)

            if return_timestamp:
                return annotation_df, timestamp
            else:
                return annotation_df



    @staticmethod
    def get_annotation_viz_dir(stack):
        return os.path.join(ANNOTATION_VIZ_ROOTDIR, stack)

    @staticmethod
    def get_annotation_viz_filepath(stack, sec=None, fn=None):
        if fn is None:
            fn = metadata_cache['sections_to_filenames'][sec]
        return os.path.join(ANNOTATION_VIZ_ROOTDIR, stack, fn + '_annotation_viz.tif')


    ########################################################

    @staticmethod
    def load_data(filepath, filetype):

        if not os.path.exists(filepath):
            sys.stderr.write('File does not exist: %s\n' % filepath)

        if filetype == 'bp':
            return bp.unpack_ndarray_file(filepath)
        elif filetype == 'image':
            return imread(filepath)
        elif filetype == 'hdf':
            try:
                return load_hdf(filepath)
            except:
                return load_hdf_v2(filepath)
        elif filetype == 'bbox':
            return np.loadtxt(filepath).astype(np.int)
        elif filetype == 'annotation_hdf':
            contour_df = read_hdf(filepath, 'contours')
            return contour_df
        elif filetype == 'pickle':
            import cPickle as pickle
            return pickle.load(open(filepath, 'r'))
        elif filetype == 'file_section_map':
            with open(filepath, 'r') as f:
                fn_idx_tuples = [line.strip().split() for line in f.readlines()]
                filename_to_section = {fn: int(idx) for fn, idx in fn_idx_tuples}
                section_to_filename = {int(idx): fn for fn, idx in fn_idx_tuples}
            return filename_to_section, section_to_filename
        elif filetype == 'label_name_map':
            label_to_name = {}
            name_to_label = {}
            with open(filepath, 'r') as f:
                for line in f.readlines():
                    name_s, label = line.split()
                    label_to_name[int(label)] = name_s
                    name_to_label[name_s] = int(label)
            return label_to_name, name_to_label
        elif filetype == 'anchor':
            with open(filepath, 'r') as f:
                anchor_fn = f.readline().strip()
            return anchor_fn
        elif filetype == 'transform_params':
            with open(filepath, 'r') as f:
                lines = f.readlines()

                global_params = one_liner_to_arr(lines[0], float)
                centroid_m = one_liner_to_arr(lines[1], float)
                xdim_m, ydim_m, zdim_m  = one_liner_to_arr(lines[2], int)
                centroid_f = one_liner_to_arr(lines[3], float)
                xdim_f, ydim_f, zdim_f  = one_liner_to_arr(lines[4], int)

            return global_params, centroid_m, centroid_f, xdim_m, ydim_m, zdim_m, xdim_f, ydim_f, zdim_f
        else:
            sys.stderr.write('File type %s not recognized.\n' % filetype)

    @staticmethod
    def get_anchor_filename_filename(stack):
        fn = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_anchor.txt')
        return fn

    @staticmethod
    def load_anchor_filename(stack):
        fp = DataManager.get_anchor_filename_filename(stack)
        # download_from_s3(fp, local_root=THUMBNAIL_DATA_ROOTDIR)
        anchor_fn = DataManager.load_data(fp, filetype='anchor')
        return anchor_fn


    @staticmethod
    def load_section_limits_v2(stack, anchor_fn=None, prep_id=2):
        """
        """

        d = load_data(DataManager.get_section_limits_filename_v2(stack=stack, anchor_fn=anchor_fn, prep_id=prep_id))
        return np.r_[d['left_section_limit'], d['right_section_limit']]

    @staticmethod
    def get_section_limits_filename_v2(stack, anchor_fn=None, prep_id=2):
        """
        Return path to file that specified the cropping box of the given crop specifier.

        Args:
            prep_id (int or str): 2D frame specifier
        """

        if isinstance(prep_id, str):
            prep_id = prep_str_to_id_2d[prep_id]

        if anchor_fn is None:
            anchor_fn = DataManager.load_anchor_filename(stack=stack)

        fp = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_alignedTo_' + anchor_fn + '_prep' + str(prep_id) + '_sectionLimits.json')
        return fp

    @staticmethod
    def get_cropbox_filename_v2(stack, anchor_fn=None, prep_id=2):
        """
        Return path to file that specified the cropping box of the given crop specifier.

        Args:
            prep_id (int or str): 2D frame specifier
        """

        if isinstance(prep_id, str):
            prep_id = prep_str_to_id_2d[prep_id]

        if anchor_fn is None:
            anchor_fn = DataManager.load_anchor_filename(stack=stack)

        fp = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_alignedTo_' + anchor_fn + '_prep' + str(prep_id) + '_cropbox.json')
        return fp

    @staticmethod
    def get_cropbox_filename(stack, anchor_fn=None, prep_id=2):
        """
        Get the filename to brainstem crop box.

        """

        if anchor_fn is None:
            anchor_fn = DataManager.load_anchor_filename(stack=stack)

        if prep_id == 3:
            fn = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_alignedTo_' + anchor_fn + '_cropbox_thalamus.txt')
        else:
            fn = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_alignedTo_' + anchor_fn + '_cropbox.txt')

        return fn
    #
    # @staticmethod
    # def get_cropbox_thalamus_filename(stack, anchor_fn=None):
    #     """
    #     Get the filename to thalamus crop box.
    #     """
    #
    #     if anchor_fn is None:
    #         anchor_fn = DataManager.load_anchor_filename(stack=stack)
    #     fn = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_alignedTo_' + anchor_fn + '_cropbox_thalamus.txt')
    #     return fn

    @staticmethod
    def get_domain_origin(stack, domain, resolution, loaded_cropbox_resolution='down32'):
        """
        Loads the 3D origin of a domain for a given stack.

        If specimen, the origin is wrt to wholebrain.
        If atlas, the origin is wrt to atlas space.

        Use this in combination with convert_frame_and_resolution().

        Args:
            domain (str): domain name
            resolution (str): output resolution
            loaded_cropbox_resolution (str): the resolution in which the loaded crop boxes are defined
        """

        out_resolution_um = convert_resolution_string_to_voxel_size(resolution=resolution, stack=stack)

        if stack.startswith('atlas'):
            if domain == 'atlasSpace':
                origin_loadedResol = np.zeros((3,))
                loaded_cropbox_resolution_um = 0. # does not matter
            elif domain == 'canonicalAtlasSpace':
                origin_loadedResol = np.zeros((3,))
                loaded_cropbox_resolution_um = 0. # does not matter
            elif domain == 'atlasSpaceBrainstem': # obsolete?
                b = DataManager.load_original_volume_bbox(stack=stack, volume_type='score',
                                        downscale=32,
                                          structure='7N_L')
                origin_loadedResol = b[[0,2,4]]
                loaded_cropbox_resolution_um = convert_resolution_string_to_voxel_size(resolution='down32', stack='MD589')
            else:
                raise
        else:

            print 'loaded_cropbox_resolution', loaded_cropbox_resolution
            loaded_cropbox_resolution_um = convert_resolution_string_to_voxel_size(resolution=loaded_cropbox_resolution, stack=stack)

            if domain == 'wholebrain':
                origin_loadedResol = np.zeros((3,))
            elif domain == 'wholebrainXYcropped':
                crop_xmin_rel2uncropped, crop_ymin_rel2uncropped = metadata_cache['cropbox'][stack][[0,2]]
                origin_loadedResol = np.array([crop_xmin_rel2uncropped, crop_ymin_rel2uncropped, 0])
            elif domain == 'brainstemXYfull':
                s1, s2 = metadata_cache['section_limits'][stack]
                crop_zmin_rel2uncropped = int(np.floor(np.mean(DataManager.convert_section_to_z(stack=stack, sec=s1, downsample=32, z_begin=0))))
                origin_loadedResol = np.array([0, 0, crop_zmin_rel2uncropped])
            elif domain == 'brainstem':
                crop_xmin_rel2uncropped, crop_ymin_rel2uncropped = metadata_cache['cropbox'][stack][[0,2]]
                s1, s2 = metadata_cache['section_limits'][stack]
                crop_zmin_rel2uncropped = int(np.floor(np.mean(DataManager.convert_section_to_z(stack=stack, sec=s1, downsample=32, z_begin=0))))
                origin_loadedResol = np.array([crop_xmin_rel2uncropped, crop_ymin_rel2uncropped, crop_zmin_rel2uncropped])
            elif domain == 'brainstemXYFullNoMargin':
                origin_loadedResol = np.loadtxt(DataManager.get_intensity_volume_bbox_filepath_v2(stack='MD589', prep_id=4, downscale=32)).astype(np.int)[[0,2,4]]
            else:
                raise "Domain %s is not recognized.\n" % domain

        origin_outResol = origin_loadedResol * loaded_cropbox_resolution_um / out_resolution_um

        return origin_outResol

    @staticmethod
    def load_cropbox_v2_relative(stack, prep_id, wrt_prep_id, out_resolution):

        alignedBrainstemCrop_xmin_down32, alignedBrainstemCrop_xmax_down32, \
        alignedBrainstemCrop_ymin_down32, alignedBrainstemCrop_ymax_down32 = DataManager.load_cropbox_v2(stack=stack, prep_id=prep_id, only_2d=True)

        alignedWithMargin_xmin_down32, alignedWithMargin_xmax_down32,\
        alignedWithMargin_ymin_down32, alignedWithMargin_ymax_down32 = DataManager.load_cropbox_v2(stack=stack, anchor_fn=None,
                                                                prep_id=wrt_prep_id,
                                                               return_dict=False, only_2d=True)

        alignedBrainstemCrop_xmin_wrt_alignedWithMargin_down32 = alignedBrainstemCrop_xmin_down32 - alignedWithMargin_xmin_down32
        alignedBrainstemCrop_xmax_wrt_alignedWithMargin_down32 = alignedBrainstemCrop_xmax_down32 - alignedWithMargin_xmin_down32
        alignedBrainstemCrop_ymin_wrt_alignedWithMargin_down32 = alignedBrainstemCrop_ymin_down32 - alignedWithMargin_ymin_down32
        alignedBrainstemCrop_ymax_wrt_alignedWithMargin_down32 = alignedBrainstemCrop_ymax_down32 - alignedWithMargin_ymin_down32

        scale_factor = convert_resolution_string_to_um('down32', stack) / convert_resolution_string_to_um(out_resolution, stack)

        return np.round([alignedBrainstemCrop_xmin_wrt_alignedWithMargin_down32 * scale_factor,
                         alignedBrainstemCrop_xmax_wrt_alignedWithMargin_down32 * scale_factor,
                         alignedBrainstemCrop_ymin_wrt_alignedWithMargin_down32 * scale_factor,
                         alignedBrainstemCrop_ymax_wrt_alignedWithMargin_down32 * scale_factor]).astype(np.int)


    @staticmethod
    def load_cropbox_v2(stack, anchor_fn=None, convert_section_to_z=False, prep_id=2,
                        return_origin_instead_of_bbox=False,
                       return_dict=False, only_2d=True):
        """
        Loads the cropping box for the given crop.

        Args:
            convert_section_to_z (bool): If true, return (xmin,xmax,ymin,ymax,zmin,zmax) where z=0 is section #1; if false, return (xmin,xmax,ymin,ymax,secmin,secmax)
            prep_id (int)
        """

        if isinstance(prep_id, str):
            fp = DataManager.get_cropbox_filename_v2(stack=stack, anchor_fn=anchor_fn, prep_id=prep_id)
        elif isinstance(prep_id, int):
            # fp = DataManager.get_cropbox_filename(stack=stack, anchor_fn=anchor_fn, prep_id=prep_id)
            fp = DataManager.get_cropbox_filename_v2(stack=stack, anchor_fn=anchor_fn, prep_id=prep_id)
        else:
            raise Exception("prep_id %s must be either str or int" % prep_id)

        # download_from_s3(fp, local_root=THUMBNAIL_DATA_ROOTDIR)

        if fp.endswith('.txt'):
            xmin, xmax, ymin, ymax, secmin, secmax = load_data(fp).astype(np.int)

            if convert_section_to_z:
                zmin = int(DataManager.convert_section_to_z(stack=stack, sec=secmin, downsample=32, z_begin=0, mid=True))
                zmax = int(DataManager.convert_section_to_z(stack=stack, sec=secmax, downsample=32, z_begin=0, mid=True))

        elif fp.endswith('.json'):
            cropbox_dict = load_data(fp)
            xmin = cropbox_dict['rostral_limit']
            xmax = cropbox_dict['caudal_limit']
            ymin = cropbox_dict['dorsal_limit']
            ymax = cropbox_dict['ventral_limit']

            if 'left_limit_section_number' in cropbox_dict:
                secmin = cropbox_dict['left_limit_section_number']
            else:
                secmin = None

            if 'right_limit_section_number' in cropbox_dict:
                secmax = cropbox_dict['right_limit_section_number']
            else:
                secmax = None

            if 'left_limit' in cropbox_dict:
                zmin = cropbox_dict['left_limit']
            else:
                zmin = None

            if 'right_limit' in cropbox_dict:
                zmax = cropbox_dict['right_limit']
            else:
                zmax = None

        if return_dict:
            if convert_section_to_z:
                cropbox_dict = {'rostral_limit': xmin,
                'caudal_limit': xmax,
                'dorsal_limit': ymin,
                'ventral_limit': ymax,
                'left_limit': zmin,
                'right_limit': zmax}
            else:
                cropbox_dict = {'rostral_limit': xmin,
                'caudal_limit': xmax,
                'dorsal_limit': ymin,
                'ventral_limit': ymax,
                'left_limit_section_number': secmin,
                'right_limit_section_number': secmax}
            return cropbox_dict

        else:
            if convert_section_to_z:
                cropbox = np.array((xmin, xmax, ymin, ymax, zmin, zmax))
                if return_origin_instead_of_bbox:
                    return cropbox[[0,2,4]].astype(np.int)
                else:
                    if only_2d:
                        return cropbox[:4].astype(np.int)
                    else:
                        return cropbox.astype(np.int)
            else:
                assert not return_origin_instead_of_bbox
                cropbox = np.array((xmin, xmax, ymin, ymax, secmin, secmax))
                if only_2d:
                    return cropbox[:4].astype(np.int)
                else:
                    return cropbox.astype(np.int)

    @staticmethod
    def load_cropbox(stack, anchor_fn=None, convert_section_to_z=False, prep_id=2, return_origin_instead_of_bbox=False):
        """
        Loads the crop box for brainstem.

        Args:
            convert_section_to_z (bool): If true, return (xmin,xmax,ymin,ymax,zmin,zmax) where z=0 is section #1; if false, return (xmin,xmax,ymin,ymax,secmin,secmax)
            prep_id (int)
        """

        fp = DataManager.get_cropbox_filename(stack=stack, anchor_fn=anchor_fn, prep_id=prep_id)
        # download_from_s3(fp, local_root=THUMBNAIL_DATA_ROOTDIR)

        if convert_section_to_z:
            xmin, xmax, ymin, ymax, secmin, secmax = np.loadtxt(fp).astype(np.int)
            zmin = int(np.mean(DataManager.convert_section_to_z(stack=stack, sec=secmin, downsample=32, z_begin=0)))
            zmax = int(np.mean(DataManager.convert_section_to_z(stack=stack, sec=secmax, downsample=32, z_begin=0)))
            cropbox = np.array((xmin, xmax, ymin, ymax, zmin, zmax))
        else:
            cropbox = np.loadtxt(fp).astype(np.int)

        if return_origin_instead_of_bbox:
            return cropbox[[0,2,4]]
        else:
            return cropbox

    # @staticmethod
    # def load_cropbox_thalamus(stack, anchor_fn=None, convert_section_to_z=False):
    #     """
    #     Loads the crop box for thalamus.
    #
    #     Args:
    #         convert_section_to_z (bool): If true, return (xmin,xmax,ymin,ymax,zmin,zmax); if false, return (xmin,xmax,ymin,ymax,secmin,secmax)
    #     """
    #
    #     fp = DataManager.get_cropbox_thalamus_filename(stack=stack, anchor_fn=anchor_fn)
    #     download_from_s3(fp, local_root=THUMBNAIL_DATA_ROOTDIR)
    #
    #     if convert_section_to_z:
    #         xmin, xmax, ymin, ymax, secmin, secmax = np.loadtxt(fp).astype(np.int)
    #         zmin = int(np.mean(DataManager.convert_section_to_z(stack=stack, sec=secmin, downsample=32, z_begin=0)))
    #         zmax = int(np.mean(DataManager.convert_section_to_z(stack=stack, sec=secmax, downsample=32, z_begin=0)))
    #         cropbox = np.array((xmin, xmax, ymin, ymax, zmin, zmax))
    #     else:
    #         cropbox = np.loadtxt(fp).astype(np.int)
    #     return cropbox

    @staticmethod
    def get_sorted_filenames_filename(stack):
        fn = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_sorted_filenames.txt')
        return fn

    @staticmethod
    def load_sorted_filenames(stack=None, fp=None, redownload=False):
        """
        Get the mapping between section index and image filename.

        Returns:
            Two dicts: filename_to_section, section_to_filename
        """

        if fp is None:
            fp = DataManager.get_sorted_filenames_filename(stack)

        # download_from_s3(fp, local_root=THUMBNAIL_DATA_ROOTDIR, redownload=redownload)
        filename_to_section, section_to_filename = DataManager.load_data(fp, filetype='file_section_map')
        if 'Placeholder' in filename_to_section:
            filename_to_section.pop('Placeholder')
        return filename_to_section, section_to_filename

    @staticmethod
    def get_transforms_filename(stack, anchor_fn=None):
        if anchor_fn is None:
            anchor_fn = metadata_cache['anchor_fn'][stack]
        fn = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_transformsTo_%s.pkl' % anchor_fn)
        return fn


    @staticmethod
    def load_consecutive_section_transform(stack, moving_fn, fixed_fn, elastix_output_dir):

        from preprocess_utilities import parse_elastix_parameter_file

        custom_tf_fp = os.path.join(DATA_DIR, stack, stack + '_custom_transforms', moving_fn + '_to_' + fixed_fn, moving_fn + '_to_' + fixed_fn + '_customTransform.txt')

        custom_tf_fp2 = os.path.join(DATA_DIR, stack, stack + '_custom_transforms', moving_fn + '_to_' + fixed_fn, 'TransformParameters.0.txt')

        if os.path.exists(custom_tf_fp):
            # if custom transform is provided
            sys.stderr.write('Load custom transform: %s\n' % custom_tf_fp)
            with open(custom_tf_fp, 'r') as f:
                t11, t12, t13, t21, t22, t23 = map(float, f.readline().split())
            transformation_to_previous_sec = np.linalg.inv(np.array([[t11, t12, t13], [t21, t22, t23], [0,0,1]]))
        elif os.path.exists(custom_tf_fp2):
            sys.stderr.write('Load custom transform: %s\n' % custom_tf_fp2)
            transformation_to_previous_sec = parse_elastix_parameter_file(custom_tf_fp2)
        else:
            # otherwise, load elastix output
            param_fp = os.path.join(elastix_output_dir, moving_fn + '_to_' + fixed_fn, 'TransformParameters.0.txt')
            sys.stderr.write('Load elastix-computed transform: %s\n' % param_fp)
            if not os.path.exists(param_fp):
                raise Exception('Transform file does not exist: %s to %s, %s' % (moving_fn, fixed_fn, param_fp))
            transformation_to_previous_sec = parse_elastix_parameter_file(param_fp)

        return transformation_to_previous_sec

    @staticmethod
    def get_elastix_parameters_dir():
        return os.path.join(REPO_DIR, 'preprocess', 'parameters')

    @staticmethod
    def load_image_filepath_warped_to_adjacent_section(stack, moving_fn, fixed_fn):
        # generated by manually specifying landmark points
        custom_aligned_image_fn = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_custom_transforms', moving_fn + '_to_' + fixed_fn, moving_fn + '_alignedTo_' + fixed_fn + '.tif')

        # generated by running elastix
        custom_aligned_image_fn2 = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_custom_transforms', moving_fn + '_to_' + fixed_fn, 'result.0.tif')

        if os.path.exists(custom_aligned_image_fn):
            sys.stderr.write('Load custom transform image. %s\n' % custom_aligned_image_fn)
            return custom_aligned_image_fn
        elif os.path.exists(custom_aligned_image_fn2):
            sys.stderr.write('Load custom transform image. %s\n' % custom_aligned_image_fn2)
            return custom_aligned_image_fn2
        else:
            fp = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_elastix_output', moving_fn + '_to_' + fixed_fn, 'result.0.tif')
            return fp

#     @staticmethod
#     def load_consecutive_section_transform(stack, moving_fn, fixed_fn):

#         custom_tf_fn = os.path.join(DATA_DIR, stack, stack + '_custom_transforms', moving_fn + '_to_' + fixed_fn, moving_fn + '_to_' + fixed_fn + '_customTransform.txt')
#         custom_tf_fn2 = os.path.join(DATA_DIR, stack, stack + '_custom_transforms', moving_fn + '_to_' + fixed_fn, 'TransformParameters.0.txt')
#         if os.path.exists(custom_tf_fn):
#             # if custom transform is provided
#             sys.stderr.write('Load custom transform: %s\n' % custom_tf_fn)
#             with open(custom_tf_fn, 'r') as f:
#                 t11, t12, t13, t21, t22, t23 = map(float, f.readline().split())
#             transformation_to_previous_sec = np.linalg.inv(np.array([[t11, t12, t13], [t21, t22, t23], [0,0,1]]))

#         elif os.path.exists(custom_tf_fn2):
#             sys.stderr.write('Load custom transform: %s\n' % custom_tf_fn2)
#             transformation_to_previous_sec = parse_elastix_parameter_file(custom_tf_fn2)
#         else:
#             # otherwise, load elastix output
#             sys.stderr.write('Load elastix-computed transform: %s\n' % custom_tf_fn2)
#             param_fn = os.path.join(elastix_output_dir, moving_fn + '_to_' + fixed_fn, 'TransformParameters.0.txt')
#             if not os.path.exists(param_fn):
#                 raise Exception('Transform file does not exist: %s to %s, %s' % (moving_fn, fixed_fn, param_fn))
#             transformation_to_previous_sec = parse_elastix_parameter_file(param_fn)

#         return transformation_to_previous_sec


    @staticmethod
    def load_transforms_v2(stack, in_image_resolution, out_image_resolution, use_inverse=True, anchor_fn=None):
        """
        Args:
            use_inverse (bool): If True, load the 2-d rigid transforms that when multiplied
                                to coordinates on the raw image space converts it to on aligned space.
                                In preprocessing, set to False, which means simply parse the transform files as they are.
            in_image_resolution (str): resolution of the image that the loaded transforms are derived from.
            out_image_resolution (str): resolution of the image that the output transform will be applied to.
        """

        rescale_in_resol_to_1um = convert_resolution_string_to_um(stack=stack, resolution=in_image_resolution)
        rescale_1um_to_out_resol = convert_resolution_string_to_um(stack=stack, resolution=out_image_resolution)

        Ts_anchor_to_individual_section_image_resol = DataManager.load_transforms(stack=stack, resolution='1um', use_inverse=True, anchor_fn=anchor_fn)

        Ts = {}

        for fn, T in Ts_anchor_to_individual_section_image_resol.iteritems():

            if use_inverse:
                T = np.linalg.inv(T)

            T_rescale_1um_to_out_resol = np.diag([1./rescale_1um_to_out_resol, 1./rescale_1um_to_out_resol, 1.])
            T_rescale_in_resol_to_1um = np.diag([rescale_in_resol_to_1um, rescale_in_resol_to_1um, 1.])

            T_overall = np.dot(T_rescale_1um_to_out_resol, np.dot(T, T_rescale_in_resol_to_1um))
            Ts[fn] = T_overall

        return Ts


    @staticmethod
    def load_transforms(stack, downsample_factor=None, resolution=None, use_inverse=True, anchor_fn=None):
        """
        Args:
            use_inverse (bool): If True, load the 2-d rigid transforms that when multiplied
                                to a point on original space converts it to on aligned space.
                                In preprocessing, set to False, which means simply parse the transform files as they are.
            downsample_factor (float): the downsample factor of images that the output transform will be applied to.
            resolution (str): resolution of the image that the output transform will be applied to.
        """

        if resolution is None:
            assert downsample_factor is not None
            resolution = 'down%d' % downsample_factor

        fp = DataManager.get_transforms_filename(stack, anchor_fn=anchor_fn)
        # download_from_s3(fp, local_root=THUMBNAIL_DATA_ROOTDIR)
        Ts_down32 = DataManager.load_data(fp, filetype='pickle')

        if use_inverse:
            Ts_inv_rescaled = {}
            for fn, T_down32 in sorted(Ts_down32.items()):
                T_rescaled = T_down32.copy()
                T_rescaled[:2, 2] = T_down32[:2, 2] * 32. * planar_resolution[stack] / convert_resolution_string_to_voxel_size(stack=stack, resolution=resolution)
                T_rescaled_inv = np.linalg.inv(T_rescaled)
                Ts_inv_rescaled[fn] = T_rescaled_inv
            return Ts_inv_rescaled
        else:
            Ts_rescaled = {}
            for fn, T_down32 in sorted(Ts_down32.items()):
                T_rescaled = T_down32.copy()
                T_rescaled[:2, 2] = T_down32[:2, 2] * 32. * planar_resolution[stack] / convert_resolution_string_to_voxel_size(stack=stack, resolution=resolution)
                Ts_rescaled[fn] = T_rescaled

            return Ts_rescaled

    ################
    # Registration #
    ################

    # @staticmethod
    # def get_original_volume_basename(stack, classifier_setting=None, downscale=32, volume_type='score', **kwargs):
    #     return DataManager.get_warped_volume_basename(stack_m=stack, classifier_setting_m=classifier_setting,
    #     downscale=downscale, type_m=volume_type)

    @staticmethod
    def get_original_volume_basename(stack, prep_id=None, detector_id=None, resolution=None, downscale=None, structure=None, volume_type='score', **kwargs):
        """
        Args:
            resolution (str): down32 or 10.0um
        """

        components = []
        if prep_id is not None:
            components.append('prep%(prep)d' % {'prep':prep_id})
        if detector_id is not None:
            components.append('detector%(detector_id)d' % {'detector_id':detector_id})

        if resolution is None:
            if downscale is not None:
                resolution = 'down%d' % downscale

        if resolution is not None:
            components.append('%(outres)s' % {'outres':resolution})

        tmp_str = '_'.join(components)
        basename = '%(stack)s_%(tmp_str)s_%(volstr)s' % \
            {'stack':stack, 'tmp_str':tmp_str, 'volstr':volume_type_to_str(volume_type)}
        if structure is not None:
            basename += '_' + structure
        return basename

    @staticmethod
    def get_original_volume_basename_v2(stack_spec):
        """
        Args:
            stack_spec (dict):
                - prep_id
                - detector_id
                - vol_type
                - structure (str or list)
                - name
                - resolution
        """

        if 'prep_id' in stack_spec:
            prep_id = stack_spec['prep_id']
        else:
            prep_id = None

        if 'detector_id' in stack_spec:
            detector_id = stack_spec['detector_id']
        else:
            detector_id = None

        if 'vol_type' in stack_spec:
            volume_type = stack_spec['vol_type']
        else:
            volume_type = None

        if 'structure' in stack_spec:
            structure = stack_spec['structure']
        else:
            structure = None

        assert 'name' in stack_spec, stack_spec
        stack = stack_spec['name']

        if 'resolution' in stack_spec:
            resolution = stack_spec['resolution']
        else:
            resolution = None

        components = []
        if prep_id is not None:
            if isinstance(prep_id, str):
                components.append(prep_id)
            elif isinstance(prep_id, int):
                components.append('prep%(prep)d' % {'prep':prep_id})
        if detector_id is not None:
            components.append('detector%(detector_id)d' % {'detector_id':detector_id})
        if resolution is not None:
            components.append(resolution)

        tmp_str = '_'.join(components)
        basename = '%(stack)s_%(tmp_str)s%(volstr)s' % \
            {'stack':stack, 'tmp_str': (tmp_str+'_') if tmp_str != '' else '', 'volstr':volume_type_to_str(volume_type)}
        if structure is not None:
            if isinstance(structure, str):
                basename += '_' + structure
            elif isinstance(structure, list):
                basename += '_' + '_'.join(sorted(structure))
            else:
                raise

        return basename

    # OBSOLETE
    @staticmethod
    def get_warped_volume_basename(stack_m,
                                   stack_f=None,
                                   warp_setting=None,
                                   prep_id_m=None,
                                   prep_id_f=None,
                                   detector_id_m=None,
                                   detector_id_f=None,
                                   downscale=32,
                                   structure_m=None,
                                   structure_f=None,
                                   vol_type_m='score',
                                   vol_type_f='score',
                                   trial_idx=None,
                                   **kwargs):

        basename_m = DataManager.get_original_volume_basename(stack=stack_m, prep_id=prep_id_m, detector_id=detector_id_m,
                                                  resolution='down%d'%downscale, volume_type=vol_type_m, structure=structure_m)

        if stack_f is None:
            assert warp_setting is None
            vol_name = basename_m
        else:
            basename_f = DataManager.get_original_volume_basename(stack=stack_f, prep_id=prep_id_f, detector_id=detector_id_f,
                                                  resolution='down%d'%downscale, volume_type=vol_type_f, structure=structure_f)
            vol_name = basename_m + '_warp%(warp)d_' % {'warp':warp_setting} + basename_f

        if trial_idx is not None:
            vol_name += '_trial_%d' % trial_idx

        return vol_name

    @staticmethod
    def get_warped_volume_basename_v2(alignment_spec, trial_idx=None):
        """
        Args:
            alignment_spec (dict): must have these keys warp_setting, stack_m and stack_f
        """

        warp_setting = alignment_spec['warp_setting']
        basename_m = DataManager.get_original_volume_basename_v2(alignment_spec['stack_m'])
        basename_f = DataManager.get_original_volume_basename_v2(alignment_spec['stack_f'])
        vol_name = basename_m + '_warp%(warp)d_' % {'warp':warp_setting} + basename_f

        if trial_idx is not None:
            vol_name += '_trial_%d' % trial_idx

        return vol_name

    # OBSOLETE
    @staticmethod
    def get_alignment_parameters_filepath(stack_f, stack_m,
                                          warp_setting,
                                          prep_id_m=None, prep_id_f=None,
                                          detector_id_m=None, detector_id_f=None,
                                          structure_f=None, structure_m=None,
                                          vol_type_f='score', vol_type_m='score',
                                          downscale=32,
                                          trial_idx=None):
        basename = DataManager.get_warped_volume_basename(**locals())
        fp = os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, '%(stack_m)s',
                              '%(basename)s',
                              '%(basename)s_parameters.txt') % {'stack_m': stack_m, 'basename':basename}
        return fp

    # OBSOLETE
    @staticmethod
    def load_alignment_parameters(stack_f, stack_m, warp_setting,
                                  prep_id_m=None, prep_id_f=None,
                                  detector_id_m=None, detector_id_f=None,
                                  structure_f=None, structure_m=None,
                                  vol_type_f='score', vol_type_m='score',
                                  downscale=32, trial_idx=None):
        """
        Returns
            (flattened parameters, centroid_m, centroid_f, xdim_m, ydim_m, zdim_m, xdim_f, ydim_f, zdim_f)
        """
        params_fp = DataManager.get_alignment_parameters_filepath(**locals())
        # download_from_s3(params_fp, redownload=True)
        # download_from_s3(params_fp, redownload=False)
        return DataManager.load_data(params_fp, 'transform_params')

    # @staticmethod
    # def load_alignment_parameters_v2(stack_f, stack_m, warp_setting,
    #                               prep_id_m=None, prep_id_f=None,
    #                               detector_id_m=None, detector_id_f=None,
    #                               structure_f=None, structure_m=None,
    #                               vol_type_f='score', vol_type_m='score',
    #                               downscale=32, trial_idx=None):
    #     what = 'parameters'
    #     tf_param_fp = DataManager.get_alignment_result_filepath_v2(**locals())
    #     download_from_s3(tf_param_fp)
    #     return load_json(tf_param_fp)

    # @staticmethod
    # def load_alignment_parameters_v3(alignment_spec, reg_root_dir):
    #     tf_param_fp = DataManager.get_alignment_result_filepath_v3(alignment_spec=alignment_spec, what='parameters', reg_root_dir=reg_root_dir)
    #     download_from_s3(tf_param_fp)
    #     tf_param = load_json(tf_param_fp)
    #     return {k: np.array(v) if isinstance(v, list) else v for k, v in tf_param.iteritems()}

    @staticmethod
    def load_alignment_results_v3(alignment_spec, what, reg_root_dir=REGISTRATION_PARAMETERS_ROOTDIR, out_form='dict'):
        """
        Args:
            what (str): any of parameters, scoreHistory, scoreEvolution or trajectory
        """
        from registration_utilities import convert_transform_forms
        res = load_data(DataManager.get_alignment_result_filepath_v3(alignment_spec=alignment_spec, what=what, reg_root_dir=reg_root_dir))
        if what == 'parameters':
            tf_out = convert_transform_forms(transform=res, out_form=out_form)
            return tf_out
        else:
            return res


    @staticmethod
    def save_alignment_results_v3(transform_parameters=None, score_traj=None, parameter_traj=None,
                                  alignment_spec=None,
                                  aligner=None, select_best='last_value',
                                  reg_root_dir=REGISTRATION_PARAMETERS_ROOTDIR,
                                 upload_s3=True):
        """
        Save the following alignment results:
        - `parameters`: eventual parameters
        - `scoreHistory`: score trajectory
        - `scoreEvolution`: a plot of score trajectory, exported as PNG
        - `trajectory`: parameter trajectory

        Must provide `alignment_spec`

        Args:
            transform_parameters:
            score_traj ((Ti,) array): score trajectory
            parameter_traj ((Ti, 12) array): parameter trajectory
            select_best (str): last_value or max_value
            alignment_spec (dict)
        """

        if aligner is not None:
            score_traj = aligner.scores
            parameter_traj = aligner.Ts

            if select_best == 'last_value':
                transform_parameters = dict(parameters=parameter_traj[-1],
                centroid_m_wrt_wholebrain=aligner.centroid_m,
                centroid_f_wrt_wholebrain=aligner.centroid_f)
            elif select_best == 'max_value':
                transform_parameters = dict(parameters=parameter_traj[np.argmax(score_traj)],
                centroid_m_wrt_wholebrain=aligner.centroid_m,
                centroid_f_wrt_wholebrain=aligner.centroid_f)
            else:
                raise Exception("select_best %s is not recognize." % select_best)

        # Save parameters
        params_fp = DataManager.get_alignment_result_filepath_v3(alignment_spec=alignment_spec, what='parameters', reg_root_dir=reg_root_dir)
        create_if_not_exists(os.path.dirname(params_fp))
        save_json(transform_parameters, params_fp)
        if upload_s3:
            upload_to_s3(params_fp)

        # Save score history
        history_fp = DataManager.get_alignment_result_filepath_v3(alignment_spec=alignment_spec, what='scoreHistory', reg_root_dir=reg_root_dir)
        bp.pack_ndarray_file(np.array(score_traj), history_fp)
        if upload_s3:
            upload_to_s3(history_fp)

        # Save score plot
        score_plot_fp = \
        history_fp = DataManager.get_alignment_result_filepath_v3(alignment_spec=alignment_spec, what='scoreEvolution', reg_root_dir=reg_root_dir)
        fig = plt.figure();
        plt.plot(score_traj);
        plt.savefig(score_plot_fp, bbox_inches='tight')
        plt.close(fig)
        if upload_s3:
            upload_to_s3(score_plot_fp)

        # Save trajectory
        trajectory_fp = DataManager.get_alignment_result_filepath_v3(alignment_spec=alignment_spec, what='trajectory', reg_root_dir=reg_root_dir)
        bp.pack_ndarray_file(np.array(parameter_traj), trajectory_fp)
        if upload_s3:
            upload_to_s3(trajectory_fp)


    @staticmethod
    def get_alignment_result_filepath_v3(alignment_spec, what, reg_root_dir=REGISTRATION_PARAMETERS_ROOTDIR):
        """
        Args:
            what (str): any of parameters, scoreHistory/trajectory, scoreEvolution, parametersWeightedAverage
        """
        warp_basename = DataManager.get_warped_volume_basename_v2(alignment_spec=alignment_spec)
        if what == 'parameters':
            ext = 'json'
        elif what == 'scoreHistory' or what == 'trajectory':
            ext = 'bp'
        elif what == 'scoreEvolution':
            ext = 'png'
        elif what == 'parametersWeightedAverage':
            ext = 'pkl'
        else:
            raise

        fp = os.path.join(reg_root_dir, alignment_spec['stack_m']['name'],
                          warp_basename, warp_basename + '_' + what + '.' + ext)
        return fp

    @staticmethod
    def get_best_trial_index_filepath(stack_f, stack_m, warp_setting,
    classifier_setting_m=None, classifier_setting_f=None,
    type_f='score', type_m='score', downscale=32, param_suffix=None):
        basename = DataManager.get_warped_volume_basename(**locals())
        if param_suffix is None:
            fp = os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m, basename + '_bestTrial', basename + '_bestTrial.txt')
        else:
            fp = os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m, basename + '_bestTrial', basename + '_bestTrial_%(param_suffix)s.txt' % \
                             {'param_suffix':param_suffix})
        return fp

    @staticmethod
    def load_best_trial_index(stack_f, stack_m, warp_setting,
    classifier_setting_m=None, classifier_setting_f=None,
    type_f='score', type_m='score', downscale=32, param_suffix=None):
        fp = DataManager.get_best_trial_index_filepath(**locals())
        # download_from_s3(fp)
        with open(fp, 'r') as f:
            best_trial_index = int(f.readline())
        return best_trial_index

    @staticmethod
    def load_best_trial_index_all_structures(stack_f, stack_m, warp_setting,
    classifier_setting_m=None, classifier_setting_f=None,
    type_f='score', type_m='score', downscale=32):
        input_kwargs = locals()
        best_trials = {}
        for structure in all_known_structures_sided:
            try:
                best_trials[structure] = DataManager.load_best_trial_index(param_suffix=structure, **input_kwargs)
            except Exception as e:
                sys.stderr.write(str(e) + '\n')
                sys.stderr.write("Best trial file for structure %s is not found.\n" % structure)
        return best_trials


    @staticmethod
    def get_alignment_viz_filepath(stack_m, stack_f,
                                   warp_setting,
                                    section,
                                   prep_id_m=None, prep_id_f=None,
                                   detector_id_m=None, detector_id_f=None,
                                    vol_type_m='score', vol_type_f='score',
                                    downscale=32,
                                    trial_idx=None,
                                  out_downscale=32):
        """
        Args:
            downscale (int): downscale of both volumes (must be consistent).
            out_downsample (int): downscale of the output visualization images.
        """

        reg_basename = DataManager.get_warped_volume_basename(**locals())
        return os.path.join(REGISTRATION_VIZ_ROOTDIR, stack_m, reg_basename, 'down'+str(out_downscale), reg_basename + '_%04d_down%d.jpg' % (section, out_downscale))

    @staticmethod
    def load_confidence(stack_m, stack_f,
                        warp_setting, what,
                        detector_id_m=None,
                                detector_id_f=None,
                        prep_id_m=None,
                        prep_id_f=None,
                        structure_f=None,
                        structure_m=None,
                            type_m='score', type_f='score',
                            trial_idx=None):
        fp = DataManager.get_confidence_filepath(**locals())
        # download_from_s3(fp)
        return load_pickle(fp)

    @staticmethod
    def get_confidence_filepath(stack_m, stack_f,
                                warp_setting, what,
                                detector_id_m=None,
                                detector_id_f=None,
                                prep_id_m=None,
                                prep_id_f=None,
                                structure_f=None,
                        structure_m=None,
                            type_m='score', type_f='score', param_suffix=None,
                            trial_idx=None):
        basename = DataManager.get_warped_volume_basename(**locals())

        if param_suffix is None:
            fn = basename + '_parameters' % {'param_suffix':param_suffix}
        else:
            fn = basename + '_parameters_%(param_suffix)s' % {'param_suffix':param_suffix}

        if what == 'hessians':
            return os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m, basename + '_hessians', fn + '_hessians.pkl')
        elif what == 'hessiansZscoreBased':
            return os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m, basename + '_hessiansZscoreBased', fn + '_hessiansZscoreBased.pkl')
        elif what == 'zscores':
            return os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m, basename + '_zscores', fn + '_zscores.pkl')
        elif what == 'score_landscape':
            return os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m, basename + '_scoreLandscape', fn + '_scoreLandscape.png')
        elif what == 'score_landscape_rotations':
            return os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m, basename + '_scoreLandscapeRotations', fn + '_scoreLandscapeRotations.png')
        elif what == 'peak_width':
            return os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m, basename + '_peakWidth', fn + '_peakWidth.pkl')
        elif what == 'peak_radius':
            return os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m, basename + '_peakRadius', fn + '_peakRadius.pkl')
        elif what == 'peak_radius_rotations':
            return os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m, basename + '_peakRadiusRotations', fn + '_peakRadiusRotations.pkl')
        elif what == 'hessians_rotations':
            return os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m, basename + '_hessiansRotations', fn + '_hessiansRotations.pkl')
        elif what == 'zscores_rotations':
            return os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m, basename + '_zscoresRotations', fn + '_zscoresRotations.pkl')

        raise Exception("Unrecognized confidence type %s" % what)

    @staticmethod
    def get_classifier_filepath(structure, classifier_id):
        clf_fp = os.path.join(CLF_ROOTDIR, 'setting_%(setting)s', 'classifiers', '%(structure)s_clf_setting_%(setting)d.dump') % {'structure': structure, 'setting':classifier_id}
        return clf_fp

    @staticmethod
    def load_classifiers(classifier_id, structures=all_known_structures):

        from sklearn.externals import joblib

        clf_allClasses = {}
        for structure in structures:
            clf_fp = DataManager.get_classifier_filepath(structure=structure, classifier_id=classifier_id)
            # download_from_s3(clf_fp)
            if os.path.exists(clf_fp):
                clf_allClasses[structure] = joblib.load(clf_fp)
            else:
                sys.stderr.write('Setting %d: No classifier found for %s.\n' % (classifier_id, structure))

        return clf_allClasses

#     @staticmethod
#     def load_sparse_scores(stack, structure, classifier_id, sec=None, fn=None, anchor_fn=None):

#         if fn is None:
#             fn = metadata_cache['sections_to_filenames'][stack][sec]

#         if anchor_fn is None:
#             anchor_fn = metadata_cache['anchor_fn'][stack]

#         sparse_scores_fn = DataManager.get_sparse_scores_filepath(stack=stack, structure=structure,
#                                             classifier_id=classifier_id, fn=fn, anchor_fn=anchor_fn)
#         download_from_s3(sparse_scores_fn)
#         return DataManager.load_data(sparse_scores_fn, filetype='bp')

    @staticmethod
    def load_sparse_scores(stack, structure, detector_id, prep_id=2, sec=None, fn=None):

        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][sec]

        sparse_scores_fp = DataManager.get_sparse_scores_filepath(**locals())
        # download_from_s3(sparse_scores_fp)
        return DataManager.load_data(sparse_scores_fp, filetype='bp')

    @staticmethod
    def get_sparse_scores_filepath(stack, structure, detector_id, prep_id=2, sec=None, fn=None):
        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][sec]

        return os.path.join(SPARSE_SCORES_ROOTDIR, stack,
                            fn + '_prep%d'%prep_id,
                            'detector%d'%detector_id,
                            fn + '_prep%d'%prep_id + '_detector%d'%detector_id + '_' + structure + '_sparseScores.bp')


#     @staticmethod
#     def get_sparse_scores_filepath(stack, structure, classifier_id, sec=None, fn=None, anchor_fn=None):
#         if fn is None:
#             fn = metadata_cache['sections_to_filenames'][stack][sec]

#         if anchor_fn is None:
#             anchor_fn = metadata_cache['anchor_fn'][stack]

#         return os.path.join(SPARSE_SCORES_ROOTDIR, stack, '%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped', \
#                 '%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_%(structure)s_sparseScores_setting_%(classifier_id)s.hdf') % \
#                 {'fn': fn, 'anchor_fn': anchor_fn, 'structure':structure, 'classifier_id': classifier_id}

    @staticmethod
    def load_intensity_volume(stack, downscale=32):
        fn = DataManager.get_intensity_volume_filepath(stack=stack, downscale=downscale)
        return DataManager.load_data(fn, filetype='bp')

    @staticmethod
    def load_intensity_volume_v2(stack, downscale=32, prep_id=2):
        """
        v2 adds argument `prep_id`.
        """
        fn = DataManager.get_intensity_volume_filepath_v2(stack=stack, downscale=downscale, prep_id=prep_id)
        # download_from_s3(fn)
        return DataManager.load_data(fn, filetype='bp')

    @staticmethod
    def load_intensity_volume_v3(stack, prep_id=2, downscale=32, return_origin_instead_of_bbox=False):
        """
        Returns:
            (3d volume of uint8, bbox_wrt_wholebrain)
        """

        fn = DataManager.get_intensity_volume_filepath_v2(stack=stack, prep_id=prep_id, downscale=downscale)
        # download_from_s3(fn)
        vol = DataManager.load_data(fn, filetype='bp')

        bbox_fp = DataManager.get_intensity_volume_bbox_filepath_v2(stack=stack, prep_id=prep_id, downscale=downscale)
        bbox_wrt_wholebrain = np.loadtxt(bbox_fp, dtype=np.int)

        if return_origin_instead_of_bbox:
            return (vol, np.array(bbox_wrt_wholebrain)[[0,2,4]])
        else:
            return (vol, bbox_wrt_wholebrain)

    @staticmethod
    def get_intensity_volume_filepath(stack, downscale=32):
        basename = DataManager.get_original_volume_basename(stack=stack, volume_type='intensity', downscale=downscale)
        vol_fn = os.path.join(VOLUME_ROOTDIR, stack, basename, basename + '.bp')
        return vol_fn

    @staticmethod
    def get_intensity_volume_filepath_v2(stack, downscale=32, prep_id=2):
        basename = DataManager.get_original_volume_basename(stack=stack, volume_type='intensity', downscale=downscale, prep_id=prep_id)
        vol_fn = os.path.join(VOLUME_ROOTDIR, stack, basename, basename + '.bp')
        return vol_fn

    @staticmethod
    def get_intensity_volume_bbox_filepath_v2(stack, downscale=32, prep_id=2):
        basename = DataManager.get_original_volume_basename(volume_type='intensity', **locals())
        return os.path.join(VOLUME_ROOTDIR, stack, basename, basename + '_bbox.txt')

    @staticmethod
    def get_intensity_volume_bbox_filepath(stack, downscale=32):
        basename = DataManager.get_original_volume_basename(volume_type='intensity', **locals())
        return os.path.join(VOLUME_ROOTDIR, stack, basename, basename + '_bbox.txt')

    # @staticmethod
    # def get_intensity_volume_bbox_filepath(stack, downscale=32):
    #     basename = DataManager.get_original_volume_basename(volume_type='intensity', **locals())
    #     return os.path.join(VOLUME_ROOTDIR, stack, basename, basename + '_bbox.txt')

    @staticmethod
    def get_intensity_volume_metaimage_filepath(stack, downscale=32):
        """
        Returns:
            (header *.mhd filepath, data *.raw filepath)
        """
        basename = DataManager.get_original_volume_basename(stack=stack, volume_type='intensity_metaimage', downscale=downscale)
        vol_mhd_fp = os.path.join(VOLUME_ROOTDIR, stack, basename, basename + '.mhd')
        vol_raw_fp = os.path.join(VOLUME_ROOTDIR, stack, basename, basename + '.raw')
        return vol_mhd_fp, vol_raw_fp

    @staticmethod
    def get_intensity_volume_mask_metaimage_filepath(stack, downscale=32):
        """
        Returns:
            (header *.mhd filepath, data *.raw filepath)
        """
        basename = DataManager.get_original_volume_basename(stack=stack, volume_type='intensity_metaimage', downscale=downscale)
        vol_mhd_fp = os.path.join(VOLUME_ROOTDIR, stack, basename, basename + '_mask.mhd')
        vol_raw_fp = os.path.join(VOLUME_ROOTDIR, stack, basename, basename + '_mask.raw')
        return vol_mhd_fp, vol_raw_fp

    @staticmethod
    def load_annotation_as_score_volume(stack, downscale, structure):
        fn = DataManager.get_annotation_as_score_volume_filepath(**locals())
        return DataManager.load_data(fn, filetype='bp')

    @staticmethod
    def get_annotation_as_score_volume_filepath(stack, downscale, structure):
        basename = DataManager.get_original_volume_basename(volume_type='annotation_as_score', **locals())
        vol_fn = os.path.join(VOLUME_ROOTDIR, stack, basename, 'score_volumes', basename + '.bp')
        return vol_fn

    @staticmethod
    def load_annotation_volume(stack, downscale):
        fp = DataManager.get_annotation_volume_filepath(**locals())
        # download_from_s3(fp)
        return DataManager.load_data(fp, filetype='bp')

    @staticmethod
    def get_annotation_volume_filepath(stack, downscale):
        basename = DataManager.get_original_volume_basename(volume_type='annotation', **locals())
        vol_fn = os.path.join(VOLUME_ROOTDIR, stack, basename, basename + '.bp')
        return vol_fn

    @staticmethod
    def get_annotation_volume_bbox_filepath(stack, downscale=32):
        basename = DataManager.get_original_volume_basename(volume_type='annotation', **locals())
        return os.path.join(VOLUME_ROOTDIR, stack, basename, basename + '_bbox.txt')

    @staticmethod
    def get_annotation_volume_label_to_name_filepath(stack):
        basename = DataManager.get_original_volume_basename(volume_type='annotation', **locals())
        fn = os.path.join(VOLUME_ROOTDIR, stack, basename, basename + '_nameToLabel.txt')
        # fn = os.path.join(volume_dir, stack, stack+'_down32_annotationVolume_nameToLabel.txt')
        return fn

    @staticmethod
    def load_annotation_volume_label_to_name(stack):
        fn = DataManager.get_annotation_volume_label_to_name_filepath(stack)
        # download_from_s3(fn)
        label_to_name, name_to_label = DataManager.load_data(fn, filetype='label_name_map')
        return label_to_name, name_to_label

    ################
    # Mesh related #
    ################

    # @staticmethod
    # def load_shell_mesh(stack, downscale, return_polydata_only=True):
    #     shell_mesh_fn = DataManager.get_shell_mesh_filepath(stack, downscale)
    #     return load_mesh_stl(shell_mesh_fn, return_polydata_only=return_polydata_only)
    #
    # @staticmethod
    # def get_shell_mesh_filepath(stack, downscale):
    #     basename = DataManager.get_original_volume_basename(stack=stack, downscale=downscale, volume_type='outer_contour')
    #     shell_mesh_fn = os.path.join(MESH_ROOTDIR, stack, basename, basename + "_smoothed.stl")
    #     return shell_mesh_fn

    # @staticmethod
    # def get_mesh_filepath(stack_m,
    #                         structure,
    #                         detector_id_m=None,
    #                       prep_id_f=None,
    #                         detector_id_f=None,
    #                         warp_setting=None,
    #                         stack_f=None,
    #                         downscale=32,
    #                         vol_type_m='score',
    #                       vol_type_f='score',
    #                         trial_idx=None, **kwargs):
    #     basename = DataManager.get_warped_volume_basename(**locals())
    #     fn = basename + '_%s' % structure
    #     return os.path.join(MESH_ROOTDIR, stack_m, basename, fn + '.stl')

    @staticmethod
    def get_mesh_filepath_v2(brain_spec, structure=None, resolution=None, level=None):

        if 'stack_f' in brain_spec: # warped
            basename = DataManager.get_warped_volume_basename_v2(alignment_spec=brain_spec, structure=structure, resolution=resolution)
            return os.path.join(MESH_ROOTDIR, stack_m, basename, fn + '.stl')
        else:
            basename = DataManager.get_original_volume_basename_v2(stack_spec=brain_spec)
            if structure is None:
                structure = brain_spec['structure']
            assert structure is not None, 'Must specify structure'

            if level is None:
                mesh_fp = os.path.join(MESH_ROOTDIR, '%(stack)s',
                          '%(basename)s',
                         '%(basename)s_%(struct)s.stl') % \
    {'stack':brain_spec['name'], 'basename':basename, 'struct':structure}
            else:
                mesh_fp = os.path.join(MESH_ROOTDIR, '%(stack)s',
                          '%(basename)s',
                         '%(basename)s_%(struct)s_l%(level).1f.stl') % \
    {'stack':brain_spec['name'], 'basename':basename, 'struct':structure, 'level': level}

            return mesh_fp

    @staticmethod
    def load_mesh_v2(brain_spec, structure=None, resolution=None, return_polydata_only=True, level=None):
        from vis3d_utilities import load_mesh_stl

        mesh_fp = DataManager.get_mesh_filepath_v2(brain_spec=brain_spec, structure=structure, resolution=resolution, level=level)
        mesh = load_mesh_stl(mesh_fp, return_polydata_only=return_polydata_only)
        if mesh is None:
            raise Exception('Mesh is empty: %s.' % structure)
        return mesh

    @staticmethod
    def load_meshes_v2(brain_spec,
                    structures=None,
                       resolution=None,
                    sided=True,
                    return_polydata_only=True,
                   include_surround=False,
                      levels=.5):
        """
        Args:
            levels (list of float): levels to load
        """

        kwargs = locals()

        if structures is None:
            if sided:
                if include_surround:
                    structures = all_known_structures_sided + [convert_to_surround_name(s, margin='200um') for s in all_known_structures_sided]
                else:
                    structures = all_known_structures_sided
            else:
                structures = all_known_structures

        if isinstance(levels, float) or levels is None:
            meshes = {}
            for structure in structures:
                try:
                    meshes[structure] = DataManager.load_mesh_v2(brain_spec=brain_spec,
                                                                 structure=structure,
                                                                 resolution=resolution,
                                                                 return_polydata_only=return_polydata_only,
                                                                level=levels)
                except Exception as e:
                    sys.stderr.write('Error loading mesh for %s: %s\n' % (structure, e))
            return meshes

        else:
            meshes_all_levels_all_structures = defaultdict(dict)
            for structure in structures:
                for level in levels:
                    try:
                        meshes[structure][level] = DataManager.load_mesh_v2(brain_spec=brain_spec,
                                                                     structure=structure,
                                                                     resolution=resolution,
                                                                     return_polydata_only=return_polydata_only,
                                                                    level=level)
                    except Exception as e:
                        raise e
                        sys.stderr.write('Error loading mesh for %s: %s\n' % (structure, e))
            meshes_all_levels_all_structures.default_factory = None

            return meshes_all_levels_all_structures

    @staticmethod
    def get_atlas_canonical_centroid_filepath(atlas_name, **kwargs):
        """
        Filepath of the atlas canonical centroid data.
        The centroid is with respect to the wholebrain frame of the fixed brain used to build atlas (MD589).
        """
        return os.path.join(MESH_ROOTDIR, atlas_name, atlas_name + '_canonicalCentroid_wrt_fixedWholebrain.txt')

    @staticmethod
    def get_atlas_canonical_normal_filepath(atlas_name, **kwargs):
        """
        Filepath of the atlas canonical centroid data.
        """
        return os.path.join(MESH_ROOTDIR, atlas_name, atlas_name + '_canonicalNormal.txt')

    @staticmethod
    def get_structure_mean_positions_filepath(atlas_name, resolution, **kwargs):
        """
        Filepath of the structure mean positions.
        """
        return os.path.join(MESH_ROOTDIR, atlas_name, atlas_name + '_' + resolution + '_meanPositions.pkl')

    @staticmethod
    def get_instance_centroids_filepath(atlas_name, **kwargs):
        """
        Filepath of the structure mean positions.
        """
        return os.path.join(MESH_ROOTDIR, atlas_name, atlas_name + '_instanceCentroids.pkl')

    @staticmethod
    def get_structure_viz_filepath(atlas_name, structure, suffix, **kwargs):
        """
        """
        return os.path.join(MESH_ROOTDIR, atlas_name, 'visualizations', structure, atlas_name + '_' + structure + '_' + suffix + '.png')

    @staticmethod
    def load_mean_shape(atlas_name, structure, resolution):
        """
        Returns:
            (volume, origin_wrt_meanShapeCentroid)
        """
        vol = load_data(DataManager.get_mean_shape_filepath(atlas_name=atlas_name, structure=structure, what='volume', resolution=resolution))
        ori_wrt_meanShapeCentroid = load_data(DataManager.get_mean_shape_filepath(atlas_name=atlas_name, structure=structure, what='origin_wrt_meanShapeCentroid', resolution=resolution))
        return vol, ori_wrt_meanShapeCentroid

    @staticmethod
    def get_mean_shape_filepath(atlas_name, structure, what, resolution, level=None, **kwargs):
        """
        Args:
            structure (str): unsided structure name
            what (str): any of volume, origin_wrt_meanShapeCentroid and mesh
            level (float): required if `what` = "mesh".
        """

        if what == 'volume':
            return os.path.join(MESH_ROOTDIR, atlas_name, 'mean_shapes', atlas_name + '_' + resolution + '_' + structure + '_meanShape_volume.bp')
        elif what == 'origin_wrt_meanShapeCentroid':
            return os.path.join(MESH_ROOTDIR, atlas_name, 'mean_shapes', atlas_name + '_' + resolution + '_' + structure + '_meanShape_origin_wrt_meanShapeCentroid.txt')
        elif what == 'mesh':
            return os.path.join(MESH_ROOTDIR, atlas_name, 'mean_shapes', atlas_name + '_' + resolution + '_' + structure + '_meanShape_mesh_level%.1f.stl' % level)
        else:
            raise

    # @staticmethod
    # def save_mean_shape(volume, origin, atlas_name, structure, resolution):
    #     """
    #     Save (volume, origin_wrt_meanShapeCentroid)
    #     """
    #     save_data(volume, DataManager.get_mean_shape_filepath(atlas_name=atlas_name, structure=structure, what='volume', resolution=resolution))
    #     save_data(origin, DataManager.get_mean_shape_filepath(atlas_name=atlas_name, structure=structure, what='origin_wrt_meanShapeCentroid', resolution=resolution))

    # @staticmethod
    # def get_structure_mean_shape_origin_filepath(atlas_name, structure, **kwargs):
    #     """
    #     Mean shape origin, relative to the template instance's centroid.
    #     """
    #     return os.path.join(MESH_ROOTDIR, atlas_name, 'mean_shapes', atlas_name + '_' + structure + '_meanShapeOrigin.txt')


    # @staticmethod
    # def get_structure_mean_shape_filepath(atlas_name, structure, **kwargs):
    #     """
    #     """
    #     return os.path.join(MESH_ROOTDIR, atlas_name, 'mean_shapes', atlas_name + '_' + structure + '_meanShape.bp')

    # @staticmethod
    # def get_structure_mean_shape_origin_filepath(atlas_name, structure, **kwargs):
    #     """
    #     Mean shape origin, relative to the template instance's centroid.
    #     """
    #     return os.path.join(MESH_ROOTDIR, atlas_name, 'mean_shapes', atlas_name + '_' + structure + '_meanShapeOrigin.txt')

    # @staticmethod
    # def get_structure_mean_mesh_filepath(atlas_name, structure, **kwargs):
    #     """
    #     Structure mean mesh, relative to the template instance's centroid.
    #     """
    #     return os.path.join(MESH_ROOTDIR, atlas_name, 'mean_shapes', atlas_name + '_' + structure + '_meanMesh.stl')

    @staticmethod
    def get_instance_mesh_filepath(atlas_name, structure, index, resolution=None, **kwargs):
        """
        Filepath of the instance mesh to derive mean shapes in atlas.

        Args:
            index (int): the index of the instance. The template instance is at index 0.
        """

        if resolution is None:
            return os.path.join(MESH_ROOTDIR, atlas_name, 'aligned_instance_meshes', atlas_name + '_' + structure + '_' + str(index) + '.stl')
        else:
            return os.path.join(MESH_ROOTDIR, atlas_name, 'aligned_instance_meshes', atlas_name + '_' + resolution + '_' + structure + '_' + str(index) + '.stl')


    @staticmethod
    def get_instance_sources_filepath(atlas_name, structure, **kwargs):
        """
        Path to the instance mesh sources file.
        """
        return os.path.join(MESH_ROOTDIR, atlas_name, 'instance_sources', atlas_name + '_' + structure + '_sources.pkl')

    @staticmethod
    def get_prior_covariance_matrix_filepath(atlas_name, structure):
        """
        Path to the covariance matrix files.
        """
        return os.path.join(MESH_ROOTDIR, atlas_name, 'covariance_matrices', atlas_name + '_' + structure + '_convariance.bp')

    @staticmethod
    def load_prior_covariance_matrix(atlas_name, structure):
        """
        Load the covariance matrix defined in atlas for the given structure.
        """
        return bp.unpack_ndarray_file(DataManager.get_prior_covariance_matrix_filepath(**locals()))


    ###############
    # Volumes I/O #
    ###############

    # @staticmethod
    # def load_volume_all_known_structures(stack_m, stack_f,
    #                                     warp_setting,
    #                                     classifier_setting_m=None,
    #                                     classifier_setting_f=None,
    #                                     type_m='score',
    #                                     type_f='score',
    #                                     downscale=32,
    #                                     structures=None,
    #                                     trial_idx=None,
    #                                     sided=True,
    #                                     include_surround=False):
    #     if stack_f is not None:
    #         return DataManager.load_transformed_volume_all_known_structures(**locals())
    #     else:
    #         raise Exception('Not implemented.')


    @staticmethod
    def save_transformed_volume_v2(volume, alignment_spec, structure=None, wrt='wholebrain', upload_s3=True):
        vol, ori = convert_volume_forms(volume=volume, out_form=("volume", "origin"))
        save_data(vol, DataManager.get_transformed_volume_filepath_v2(alignment_spec=alignment_spec, structure=structure), upload_s3=upload_s3)
        save_data(ori, DataManager.get_transformed_volume_origin_filepath(alignment_spec=alignment_spec, structure=structure, wrt='fixedWholebrain'), upload_s3=upload_s3)

    @staticmethod
    def save_transformed_volume(volume, bbox, alignment_spec, resolution=None, structure=None):
        """
        Save volume array as bp file and bounding box as txt file.

        Args:
            resolution (str):
            bbox ((3,)-array): wrt fixedWholebrain
        """

        if resolution is None:
            resolution = alignment_spec['stack_m']['resolution']

        ######### Save volume ##########
        volume_m_warped_fp = \
        DataManager.get_transformed_volume_filepath_v2(alignment_spec=alignment_spec, structure=structure,
                                                       resolution=resolution)
        create_parent_dir_if_not_exists(volume_m_warped_fp)
        bp.pack_ndarray_file(volume, volume_m_warped_fp)
        upload_to_s3(volume_m_warped_fp)

        ############### Save bbox #############
        volume_m_warped_bbox_fp = \
        DataManager.get_transformed_volume_bbox_filepath_v2(alignment_spec=alignment_spec, structure=structure,
                                                           resolution=resolution, wrt='fixedWholebrain')
        create_parent_dir_if_not_exists(volume_m_warped_bbox_fp)
        np.savetxt(volume_m_warped_bbox_fp, bbox[:,None], fmt='%d')
        upload_to_s3(volume_m_warped_bbox_fp)


    @staticmethod
    def load_transformed_volume_v2(alignment_spec, resolution=None, structure=None, trial_idx=None,
                                   return_origin_instead_of_bbox=False, legacy=False):
        """
        Args:
            alignment_spec (dict): specify stack_m, stack_f, warp_setting.
            resolution (str): resolution of the output volume.
            legacy (bool): if legacy, resolution can only be down32.

        Returns:
            (2-tuple): (volume, bounding box wrt "wholebrain" domain of the fixed stack)

        """
        kwargs = locals()

        if legacy:
            stack_m = alignment_spec['stack_m']['name']
            stack_f = alignment_spec['stack_f']['name']
            detector_id_f = alignment_spec['stack_f']['detector_id']
            warp = alignment_spec['warp_setting']

            origin_outResol = DataManager.get_domain_origin(stack=stack_f, domain='brainstem', resolution=resolution).astype(np.int)

            vol_down32 = DataManager.load_transformed_volume(stack_m=stack_m, stack_f=stack_f,
                                                      warp_setting=warp, detector_id_m=None,
                                                      detector_id_f=detector_id_f,
                                                      prep_id_m=None, prep_id_f=2,
                                                        vol_type_m='score', vol_type_f='score', downscale=32,
                                                        structure=structure)

            vol_outResol = rescale_by_resampling(vol_down32,
                                  scaling=convert_resolution_string_to_um(resolution='down32', stack=stack_f)/convert_resolution_string_to_um(resolution=resolution, stack=stack_f))

            bbox_outResol = np.array((origin_outResol[0],
                             origin_outResol[0]+vol_outResol.shape[1],
                             origin_outResol[1],
                             origin_outResol[1]+vol_outResol.shape[0],
                             origin_outResol[2],
                             origin_outResol[2]+vol_outResol.shape[2])).astype(np.int)

            origin_outResol = bbox_outResol[[0,2,4]].astype(np.int)
            if return_origin_instead_of_bbox:
                return (vol_outResol, origin_outResol)
            else:
                return (vol_outResol, bbox_outResol)

        else:

            vol = load_data(DataManager.get_transformed_volume_filepath_v2(alignment_spec=alignment_spec,
                                                                resolution=resolution,
                                                                structure=structure))
            # download_from_s3(fp)
            # vol = DataManager.load_data(fp, filetype='bp')

            # bbox_fp = DataManager.get_transformed_volume_bbox_filepath_v2(wrt='fixedWholebrain',
            #                                                              alignment_spec=alignment_spec,
            #                                                     resolution=resolution,
            #                                                     structure=structure)
            # download_from_s3(bbox_fp)
            # bbox = np.loadtxt(bbox_fp)
            # origin = bbox[[0,2,4]]

            origin = load_data(DataManager.get_transformed_volume_origin_filepath(wrt='fixedWholebrain',
                                                                         alignment_spec=alignment_spec,
                                                                resolution=resolution,
                                                                structure=structure))
            if return_origin_instead_of_bbox:
                return (vol, origin)
            else:
                return convert_volume_forms((vol, origin), out_form=('volume','bbox'))


    @staticmethod
    def load_transformed_volume(stack_m, stack_f,
                                warp_setting,
                                detector_id_m=None,
                                detector_id_f=None,
                                prep_id_m=None,
                                prep_id_f=None,
                                structure_f=None,
                                structure_m=None,
                                vol_type_m='score',
                                vol_type_f='score',
                                structure=None,
                                downscale=32,
                                trial_idx=None):
        fp = DataManager.get_transformed_volume_filepath(**locals())
        # download_from_s3(fp)
        return DataManager.load_data(fp, filetype='bp')


    @staticmethod
    def load_transformed_volume_all_known_structures_v3(alignment_spec,
                                                        resolution,
                                                    structures=None,
                                                    sided=True,
                                                    include_surround=False,
                                                    surround_margin='200um',
                                                     trial_idx=None,
                                                     return_label_mappings=False,
                                                     name_or_index_as_key='name',
                                                     common_shape=False,
                                                        return_origin_instead_of_bbox=False,
                                                        legacy=False,
):
        """
        Load transformed volumes for all structures and normalize them into a common shape.

        Args:
            alignment_spec (dict):
            trial_idx: could be int (for global transform) or dict {sided structure name: best trial index} (for local transform).
            common_shape (bool): If true, volumes are normalized to the same shape.

        Returns:
            If `common_shape` is True:
                if return_label_mappings is True, returns (volumes, common_bbox, structure_to_label, label_to_structure), volumes is dict.
                else, returns (volumes, common_bbox).
                By default, `common_bbox` is wrt fixed stack's wholebrain domain.

            If `common_shape` is False:
                if return_label_mappings is True, returns (dict of volume_bbox_tuples, structure_to_label, label_to_structure).
                else, returns volume_bbox_tuples.
        """

        if structures is None:
            if sided:
                if include_surround:
                    structures = structures + [convert_to_surround_name(s, margin=surround_margin) for s in structures]
                else:
                    structures = all_known_structures_sided
            else:
                structures = all_known_structures

        loaded = False
        sys.stderr.write('Prior structure/index map not found. Generating a new one.\n')

        volumes = {}
        if not loaded:
            structure_to_label = {}
            label_to_structure = {}
            index = 1

        for structure in structures:
            # try:

            if loaded:
                index = structure_to_label[structure]

            if trial_idx is None or isinstance(trial_idx, int):
                trial_idx_ = trial_idx
            else:
                trial_idx_ = trial_idx[convert_to_nonsurround_label(structure)]

            assert return_origin_instead_of_bbox

            v, b = DataManager.load_transformed_volume_v2(alignment_spec=alignment_spec,
                                                          structure=structure,
                                                          trial_idx=trial_idx_,
                                                         resolution=resolution,
                                                         return_origin_instead_of_bbox=False,
                                                         legacy=legacy)

            if name_or_index_as_key == 'name':
                volumes[structure] = (v,b)
            else:
                volumes[index] = (v,b)

            if not loaded:
                structure_to_label[structure] = index
                label_to_structure[index] = structure
                index += 1

            # except Exception as e:
                # raise e
                # sys.stderr.write('%s\n' % e)
                # sys.stderr.write('Cannot load score volume for %s.\n' % structure)

        if common_shape:
            volumes_normalized, common_bbox = convert_vol_bbox_dict_to_overall_vol(vol_bbox_dict=volumes)

            if return_label_mappings:
                if return_origin_instead_of_bbox:
                    return volumes_normalized, common_bbox[[0,2,4]], structure_to_label, label_to_structure
                else:
                    return volumes_normalized, common_bbox, structure_to_label, label_to_structure

            else:
                if return_origin_instead_of_bbox:
                    return volumes_normalized, common_bbox[[0,2,4]]
                else:
                    return volumes_normalized, common_bbox
        else:
            if return_origin_instead_of_bbox:
                volumes = {k: (v, b[[0,2,4]]) for k, (v,b) in volumes.iteritems()}

            if return_label_mappings:
                return volumes, structure_to_label, label_to_structure
            else:
                return volumes



#     @staticmethod
#     def load_transformed_volume_all_known_structures_v2(stack_m,
#                                                      stack_f,
#                                                     warp_setting,
#                                                     detector_id_m=None,
#                                                     detector_id_f=None,
#                                                      prep_id_m=None,
#                                                      prep_id_f=None,
#                                                     vol_type_m='score',
#                                                     vol_type_f='score',
#                                                     downscale=32,
#                                                     structures=None,
#                                                     sided=True,
#                                                     include_surround=False,
#                                                      trial_idx=None,
#                                                      return_label_mappings=False,
#                                                      name_or_index_as_key='name',
#                                                      common_shape=True
# ):
#         """
#         Load transformed volumes for all structures and normalize them into a common shape.
#
#         Args:
#             trial_idx: could be int (for global transform) or dict {sided structure name: best trial index} (for local transform).
#             common_shape (bool): If true, volumes are normalized to the same shape.
#
#         Returns:
#             If `common_shape` is True:
#                 if return_label_mappings is True, returns (volumes, common_bbox, structure_to_label, label_to_structure), volumes is dict.
#                 else, returns (volumes, common_bbox).
#             If `common_shape` is False:
#                 if return_label_mappings is True, returns (dict of volume_bbox_tuples, structure_to_label, label_to_structure).
#                 else, returns volume_bbox_tuples.
#         """
#
#         if structures is None:
#             if sided:
#                 if include_surround:
#                     structures = all_known_structures_sided_with_surround_200um
#                 else:
#                     structures = all_known_structures_sided
#             else:
#                 structures = all_known_structures
#
#         loaded = False
#         sys.stderr.write('Prior structure/index map not found. Generating a new one.\n')
#
#         volumes = {}
#         if not loaded:
#             structure_to_label = {}
#             label_to_structure = {}
#             index = 1
#         for structure in structures:
#             try:
#
#                 if loaded:
#                     index = structure_to_label[structure]
#
#                 if trial_idx is None or isinstance(trial_idx, int):
#                     trial_idx_ = trial_idx
#                 else:
#                     trial_idx_ = trial_idx[convert_to_nonsurround_label(structure)]
#
#                 v = DataManager.load_transformed_volume(stack_m=stack_m, vol_type_m=vol_type_m,
#                                                         stack_f=stack_f, vol_type_f=vol_type_f,
#                                                         downscale=downscale,
#                                                         prep_id_m=prep_id_m,
#                                                         prep_id_f=prep_id_f,
#                                                         detector_id_m=detector_id_m,
#                                                         detector_id_f=detector_id_f,
#                                                         warp_setting=warp_setting,
#                                                         structure=structure,
#                                                         trial_idx=trial_idx_)
#
#                 b = DataManager.load_transformed_volume_bbox(stack_m=stack_m, vol_type_m=vol_type_m,
#                                                         stack_f=stack_f, vol_type_f=vol_type_f,
#                                                         downscale=downscale,
#                                                         prep_id_m=prep_id_m,
#                                                         prep_id_f=prep_id_f,
#                                                         detector_id_m=detector_id_m,
#                                                         detector_id_f=detector_id_f,
#                                                         warp_setting=warp_setting,
#                                                         structure=structure,
#                                                         trial_idx=trial_idx_)
#
#                 if name_or_index_as_key == 'name':
#                     volumes[structure] = (v,b)
#                 else:
#                     volumes[index] = (v,b)
#
#                 if not loaded:
#                     structure_to_label[structure] = index
#                     label_to_structure[index] = structure
#                     index += 1
#
#             except Exception as e:
#                 sys.stderr.write('%s\n' % e)
#                 sys.stderr.write('Score volume for %s does not exist.\n' % structure)
#
#         if common_shape:
#             volumes_normalized, common_bbox = convert_vol_bbox_dict_to_overall_vol(vol_bbox_dict=volumes)
#
#             if return_label_mappings:
#                 return volumes_normalized, common_bbox, structure_to_label, label_to_structure
#             else:
#                 return volumes_normalized, common_bbox
#         else:
#             if return_label_mappings:
#                 return volumes, structure_to_label, label_to_structure
#             else:
#                 return volumes
#
#     @staticmethod
#     def load_transformed_volume_all_known_structures(stack_m,
#                                                      stack_f,
#                                                     warp_setting,
#                                                     detector_id_m=None,
#                                                     detector_id_f=None,
#                                                      prep_id_m=None,
#                                                      prep_id_f=None,
#                                                     vol_type_m='score',
#                                                     vol_type_f='score',
#                                                     downscale=32,
#                                                     structures=None,
#                                                     sided=True,
#                                                     include_surround=False,
#                                                      trial_idx=None,
#                                                      return_label_mappings=False,
#                                                      name_or_index_as_key='name'
# ):
#         """
#         Load transformed volumes for all structures.
#
#         Args:
#             trial_idx: could be int (for global transform) or dict {sided structure name: best trial index} (for local transform).
#             structures: Default is None - using all structures.
#
#         Returns:
#             if return_label_mappings is True, returns (volumes, structure_to_label, label_to_structure), volumes is dict.
#             else, returns volumes.
#         """
#
#         if structures is None:
#             if sided:
#                 if include_surround:
#                     structures = all_known_structures_sided_with_surround_200um
#                 else:
#                     structures = all_known_structures_sided
#             else:
#                 structures = all_known_structures
#
#         loaded = False
#         sys.stderr.write('Prior structure/index map not found. Generating a new one.\n')
#
#         volumes = {}
#         if not loaded:
#             structure_to_label = {}
#             label_to_structure = {}
#             index = 1
#         for structure in structures:
#             try:
#
#                 if loaded:
#                     index = structure_to_label[structure]
#
#                 if trial_idx is None or isinstance(trial_idx, int):
#                     v = DataManager.load_transformed_volume(stack_m=stack_m, vol_type_m=vol_type_m,
#                                                             stack_f=stack_f, vol_type_f=vol_type_f,
#                                                             downscale=downscale,
#                                                             prep_id_m=prep_id_m,
#                                                             prep_id_f=prep_id_f,
#                                                             detector_id_m=detector_id_m,
#                                                             detector_id_f=detector_id_f,
#                                                             warp_setting=warp_setting,
#                                                             structure=structure,
#                                                             trial_idx=trial_idx)
#
#                 else:
#                     v = DataManager.load_transformed_volume(stack_m=stack_m, vol_type_m=vol_type_m,
#                                                             stack_f=stack_f, vol_type_f=vol_type_f,
#                                                             downscale=downscale,
#                                                             prep_id_m=prep_id_m,
#                                                             prep_id_f=prep_id_f,
#                                                             detector_id_m=detector_id_m,
#                                                             detector_id_f=detector_id_f,
#                                                             warp_setting=warp_setting,
#                                                             structure=structure,
#                                                             trial_idx=trial_idx[convert_to_nonsurround_label(structure)])
#
#                 if name_or_index_as_key == 'name':
#                     volumes[structure] = v
#                 else:
#                     volumes[index] = v
#
#                 if not loaded:
#                     structure_to_label[structure] = index
#                     label_to_structure[index] = structure
#                     index += 1
#
#             except Exception as e:
#                 sys.stderr.write('%s\n' % e)
#                 sys.stderr.write('Score volume for %s does not exist.\n' % structure)
#
#         if return_label_mappings:
#             return volumes, structure_to_label, label_to_structure
#         else:
#             return volumes

    @staticmethod
    def get_transformed_volume_filepath_v2(alignment_spec, resolution=None, trial_idx=None, structure=None):

        if resolution is None:
            if 'resolution' in alignment_spec['stack_m']:
                resolution = alignment_spec['stack_m']['resolution']

        if structure is None:
            if 'structure' in alignment_spec['stack_m']:
                structure = alignment_spec['stack_m']['structure']

        warp_basename = DataManager.get_warped_volume_basename_v2(alignment_spec=alignment_spec,
                                                             trial_idx=trial_idx)
        vol_basename = warp_basename + '_' + resolution
        vol_basename_with_structure_suffix = vol_basename + ('_' + structure) if structure is not None else ''

        return os.path.join(VOLUME_ROOTDIR, alignment_spec['stack_m']['name'],
                            vol_basename, 'score_volumes', vol_basename_with_structure_suffix + '.bp')

    # OBSOLETE
    @staticmethod
    def get_transformed_volume_filepath(stack_m, stack_f,
                                        warp_setting,
                                        prep_id_m=None,
                                        prep_id_f=None,
                                        detector_id_m=None,
                                        detector_id_f=None,
                                        structure_m=None,
                                        structure_f=None,
                                        downscale=32,
                                        vol_type_m='score',
                                        vol_type_f='score',
                                        structure=None,
                                        trial_idx=None):

        basename = DataManager.get_warped_volume_basename(**locals())
        if structure is not None:
            fn = basename + '_%s' % structure
        else:
            fn = basename
        return os.path.join(VOLUME_ROOTDIR, stack_m, basename, 'score_volumes', fn + '.bp')

    # @staticmethod
    # def load_transformed_volume_bbox(stack_m, stack_f,
    #                                     warp_setting,
    #                                     prep_id_m=None,
    #                                     prep_id_f=None,
    #                                     detector_id_m=None,
    #                                     detector_id_f=None,
    #                                     structure_m=None,
    #                                     structure_f=None,
    #                                     downscale=32,
    #                                     vol_type_m='score',
    #                                     vol_type_f='score',
    #                                     structure=None,
    #                                     trial_idx=None):
    #     fp = DataManager.get_transformed_volume_bbox_filepath(**locals())
    #     download_from_s3(fp)
    #     return np.loadtxt(fp)


    @staticmethod
    def get_transformed_volume_origin_filepath(alignment_spec, structure=None, wrt='wholebrain', resolution=None):
        """
        Args:
            alignment_spec (dict): specifies the multi-map.
            wrt (str): specify which domain is the bounding box relative to.
            resolution (str): specifies the resolution of the multi-map.
            structure (str): specifies one map of the multi-map.
        """

        if resolution is None:
            if 'resolution' in alignment_spec['stack_m']:
                resolution = alignment_spec['stack_m']['resolution']

        if structure is None:
            if 'structure' in alignment_spec['stack_m']:
                structure = alignment_spec['stack_m']['structure']

        warp_basename = DataManager.get_warped_volume_basename_v2(alignment_spec=alignment_spec, trial_idx=None)
        vol_basename = warp_basename + '_' + resolution
        vol_basename_with_structure_suffix = vol_basename + ('_' + structure if structure is not None else '')

        return os.path.join(VOLUME_ROOTDIR, alignment_spec['stack_m']['name'],
                            vol_basename, 'score_volumes', vol_basename_with_structure_suffix + '_origin_wrt_' + wrt + '.txt')

        # volume_type = stack_spec['vol_type']
        #
        # if 'resolution' not in stack_spec or stack_spec['resolution'] is None:
        #     assert resolution is not None
        #     stack_spec['resolution'] = resolution
        #
        # if 'structure' not in stack_spec or stack_spec['structure'] is None:
        #     vol_basename = DataManager.get_original_volume_basename_v2(stack_spec=stack_spec)
        # else:
        #     stack_spec_no_structure = stack_spec.copy()
        #     stack_spec_no_structure['structure'] = None
        #     vol_basename = DataManager.get_original_volume_basename_v2(stack_spec=stack_spec_no_structure)
        #
        # if volume_type == 'score' or volume_type == 'annotationAsScore':
        #     origin_fp = os.path.join(VOLUME_ROOTDIR, '%(stack)s',
        #                   '%(basename)s',
        #                   'score_volumes',
        #                  '%(basename)s_%(struct)s_origin' + ('_wrt_'+wrt if wrt is not None else '') + '.txt') % \
        #     {'stack':stack_spec['name'], 'basename':vol_basename, 'struct':structure}
        #
        # elif volume_type == 'intensity':
        #     origin_fp = os.path.join(VOLUME_ROOTDIR, stack_spec['name'], vol_basename, vol_basename + '_origin' + ('_wrt_'+wrt if wrt is not None else '') + '.txt')
        # else:
        #     raise Exception("vol_type of %s is not recognized." % stack_spec['vol_type'])
        #
        # return origin_fp

    # @staticmethod
    # def get_transformed_volume_bbox_filepath_v2(alignment_spec,
    #                                             wrt,
    #                                             resolution=None,
    #                                     structure=None,
    #                                     trial_idx=None):
    #     """
    #     Args:
    #         alignment_spec (dict): specifies the multi-map.
    #         wrt (str): specify which domain is the bounding box relative to.
    #         resolution (str): specifies the resolution of the multi-map.
    #         structure (str): specifies one map of the multi-map.
    #     """
    #
    #     if resolution is None:
    #         resolution = alignment_spec['stack_m']['resolution']
    #
    #     warp_basename = DataManager.get_warped_volume_basename_v2(alignment_spec=alignment_spec, trial_idx=trial_idx)
    #     vol_basename = warp_basename + '_' + resolution
    #     vol_basename_with_structure_suffix = vol_basename + ('_' + structure if structure is not None else '')
    #
    #     return os.path.join(VOLUME_ROOTDIR, alignment_spec['stack_m']['name'],
    #                         vol_basename, 'score_volumes', vol_basename_with_structure_suffix + '_bbox_wrt_' + wrt + '.txt')

    # @staticmethod
    # def get_transformed_volume_bbox_filepath(stack_m, stack_f,
    #                                     warp_setting,
    #                                     prep_id_m=None,
    #                                     prep_id_f=None,
    #                                     detector_id_m=None,
    #                                     detector_id_f=None,
    #                                     structure_m=None,
    #                                     structure_f=None,
    #                                     downscale=32,
    #                                     vol_type_m='score',
    #                                     vol_type_f='score',
    #                                     structure=None,
    #                                     trial_idx=None):
    #
    #     basename = DataManager.get_warped_volume_basename(**locals())
    #     if structure is not None:
    #         fn = basename + '_%s' % structure
    #     else:
    #         fn = basename
    #     return os.path.join(VOLUME_ROOTDIR, stack_m, basename, 'score_volumes', fn + '_bbox.txt')



    ##########################
    ## Probabilistic Shape  ##
    ##########################

    @staticmethod
    def load_prob_shapes(stack_m, stack_f=None,
            classifier_setting_m=None,
            classifier_setting_f=None,
            warp_setting=None,
            downscale=32,
            type_m='score', type_f='score',
            trial_idx=0,
            structures=None,
            sided=True,
            return_polydata_only=True):

        kwargs = locals()

        if structures is None:
            if sided:
                structures = all_known_structures_sided
            else:
                structures = all_known_structures

        prob_shapes = {}
        for structure in structures:
            try:
                vol_fp = DataManager.get_prob_shape_volume_filepath(structure=structure, **kwargs)
                # download_from_s3(vol_fp)
                vol = bp.unpack_ndarray_file(vol_fp)

                origin_fp = DataManager.get_prob_shape_origin_filepath(structure=structure, **kwargs)
                # download_from_s3(origin_fp)
                origin = np.loadtxt(origin_fp)

                prob_shapes[structure] = (vol, origin)
            except Exception as e:
                sys.stderr.write('Error loading probablistic shape for %s: %s\n' % (structure, e))

        return prob_shapes

#     @staticmethod
#     def get_prob_shape_viz_filepath(stack_m, structure,
#                                     stack_f=None,
#                                         warp_setting=None,
#                                         classifier_setting_m=None,
#                                         classifier_setting_f=None,
#                                         downscale=32,
#                                          type_m='score',
#                                          type_f='score',
#                                         trial_idx=0,
#                                         suffix=None,
#                                         **kwargs):
#         """
#         Return prob. shape volume filepath.
#         """

#         basename = DataManager.get_warped_volume_basename(**locals())
#         assert structure is not None
#         fn = basename + '_' + structure + '_' + suffix
#         return os.path.join(, stack_m, basename, 'probabilistic_shape_viz', structure, fn + '.png')

#     @staticmethod
#     def get_prob_shape_volume_filepath(stack_m, stack_f=None,
#                                         warp_setting=None,
#                                         classifier_setting_m=None,
#                                         classifier_setting_f=None,
#                                         downscale=32,
#                                          type_m='score',
#                                          type_f='score',
#                                         structure=None,
#                                         trial_idx=0, **kwargs):
#         """
#         Return prob. shape volume filepath.
#         """

#         basename = DataManager.get_warped_volume_basename(**locals())
#         if structure is not None:
#             fn = basename + '_' + structure

#         return os.path.join(VOLUME_ROOTDIR, stack_m, basename, 'probabilistic_shapes', fn + '.bp')

#     @staticmethod
#     def get_prob_shape_origin_filepath(stack_m, stack_f=None,
#                                         warp_setting=None,
#                                         classifier_setting_m=None,
#                                         classifier_setting_f=None,
#                                         downscale=32,
#                                          type_m='score',
#                                          type_f='score',
#                                         structure=None,
#                                         trial_idx=0, **kwargs):
#         """
#         Return prob. shape volume origin filepath.

#         Note that these origins are with respect to

#         """

#         basename = DataManager.get_warped_volume_basename(**locals())
#         if structure is not None:
#             fn = basename + '_' + structure
#         return os.path.join(VOLUME_ROOTDIR, stack_m, basename, 'probabilistic_shapes', fn + '_origin.txt')

    # @staticmethod
    # def get_volume_filepath(stack_m, stack_f=None,
    #                                     warp_setting=None,
    #                                     classifier_setting_m=None,
    #                                     classifier_setting_f=None,
    #                                     downscale=32,
    #                                      type_m='score',
    #                                       type_f='score',
    #                                     structure=None,
    #                                     trial_idx=None):
    #
    #     basename = DataManager.get_warped_volume_basename(**locals())
    #
    #     if structure is not None:
    #         fn = basename + '_' + structure
    #
    #     if type_m == 'score':
    #         return DataManager.get_score_volume_filepath(stack=stack_m, structure=structure, downscale=downscale)
    #     else:
    #         raise

    # @staticmethod
    # def get_score_volume_filepath(stack, structure, volume_type='score', prep_id=None, detector_id=None, downscale=32):
    #     basename = DataManager.get_original_volume_basename(stack=stack, detector_id=detector_id, prep_id=prep_id, volume_type=volume_type, downscale=downscale)
    #     vol_fp = os.path.join(VOLUME_ROOTDIR, '%(stack)s',
    #                           '%(basename)s',
    #                           'score_volumes',
    #                          '%(basename)s_%(struct)s.bp') % \
    #     {'stack':stack, 'basename':basename, 'struct':structure}
    #     return vol_fp
    #
    # @staticmethod
    # def get_score_volume_filepath_v2(stack, structure, resolution, volume_type='score', prep_id=None, detector_id=None):
    #     basename = DataManager.get_original_volume_basename(stack=stack, detector_id=detector_id, prep_id=prep_id, volume_type=volume_type, resolution=resolution)
    #     vol_fp = os.path.join(VOLUME_ROOTDIR, '%(stack)s',
    #                           '%(basename)s',
    #                           'score_volumes',
    #                          '%(basename)s_%(struct)s.bp') % \
    #     {'stack':stack, 'basename':basename, 'struct':structure}
    #     return vol_fp

    @staticmethod
    def get_score_volume_filepath_v3(stack_spec, structure):
        basename = DataManager.get_original_volume_basename_v2(stack_spec=stack_spec)
        vol_fp = os.path.join(VOLUME_ROOTDIR, '%(stack)s',
                              '%(basename)s',
                              'score_volumes',
                             '%(basename)s_%(struct)s.bp') % \
        {'stack':stack_spec['name'], 'basename':basename, 'struct':structure}
        return vol_fp


    # @staticmethod
    # def get_score_volume_bbox_filepath(stack, structure, downscale, detector_id, prep_id=2, volume_type='score', **kwargs):
    #     basename = DataManager.get_original_volume_basename(stack=stack, detector_id=detector_id, prep_id=prep_id, volume_type=volume_type, downscale=downscale)
    #     fp = os.path.join(VOLUME_ROOTDIR, '%(stack)s',
    #                           '%(basename)s',
    #                           'score_volumes',
    #                          '%(basename)s_%(struct)s_bbox.txt') % \
    #     {'stack':stack, 'basename':basename, 'struct':structure}
    #     return fp
    #
    # @staticmethod
    # def get_score_volume_bbox_filepath_v2(stack, structure, resolution, detector_id, prep_id=2, volume_type='score', **kwargs):
    #     basename = DataManager.get_original_volume_basename(stack=stack, detector_id=detector_id, prep_id=prep_id, volume_type=volume_type, resolution=resolution)
    #     fp = os.path.join(VOLUME_ROOTDIR, '%(stack)s',
    #                           '%(basename)s',
    #                           'score_volumes',
    #                          '%(basename)s_%(struct)s_bbox.txt') % \
    #     {'stack':stack, 'basename':basename, 'struct':structure}
        # return fp


    @staticmethod
    def get_score_volume_origin_filepath_v3(stack_spec, structure, wrt='wholebrain'):

        if 'structure' not in stack_spec or stack_spec['structure'] is None:
            vol_basename = DataManager.get_original_volume_basename_v2(stack_spec=stack_spec)
        else:
            stack_spec_no_structure = stack_spec.copy()
            stack_spec_no_structure['structure'] = None
            vol_basename = DataManager.get_original_volume_basename_v2(stack_spec=stack_spec_no_structure)

        fp = os.path.join(VOLUME_ROOTDIR, '%(stack)s',
                          '%(basename)s',
                          'score_volumes',
                         '%(basename)s_%(struct)s_origin' + ('_wrt_'+wrt if wrt is not None else '') + '.txt') % \
        {'stack':stack_spec['name'], 'basename':vol_basename, 'struct':structure}
        return fp

    @staticmethod
    def get_score_volume_bbox_filepath_v3(stack_spec, structure, wrt='wholebrain'):

        if 'structure' not in stack_spec or stack_spec['structure'] is None:
            vol_basename = DataManager.get_original_volume_basename_v2(stack_spec=stack_spec)
        else:
            stack_spec_no_structure = stack_spec.copy()
            stack_spec_no_structure['structure'] = None
            vol_basename = DataManager.get_original_volume_basename_v2(stack_spec=stack_spec_no_structure)

        fp = os.path.join(VOLUME_ROOTDIR, '%(stack)s',
                          '%(basename)s',
                          'score_volumes',
                         '%(basename)s_%(struct)s_bbox' + ('_wrt_'+wrt if wrt is not None else '') + '.txt') % \
        {'stack':stack_spec['name'], 'basename':vol_basename, 'struct':structure}
        return fp

    # @staticmethod
    # def get_volume_gradient_filepath_template(stack, structure, prep_id=None, detector_id=None, downscale=32, volume_type='score', **kwargs):
    #     basename = DataManager.get_original_volume_basename(stack=stack, prep_id=prep_id, detector_id=detector_id, downscale=downscale, volume_type=volume_type, **kwargs)
    #     grad_fp = os.path.join(VOLUME_ROOTDIR, '%(stack)s',
    #                           '%(basename)s',
    #                           'score_volume_gradients',
    #                          '%(basename)s_%(struct)s_%%(suffix)s.bp') % \
    #     {'stack':stack, 'basename':basename, 'struct':structure}
    #     return grad_fp
    #
    # @staticmethod
    # def get_volume_gradient_filepath_template_v2(stack, structure, out_resolution_um=10.,
    #                                              prep_id=None, detector_id=None, volume_type='score', **kwargs):
    #     basename = DataManager.get_original_volume_basename(stack=stack, prep_id=prep_id, detector_id=detector_id, out_resolution_um=out_resolution_um, volume_type=volume_type, **kwargs)
    #     grad_fp = os.path.join(VOLUME_ROOTDIR, '%(stack)s',
    #                           '%(basename)s',
    #                           'score_volume_gradients',
    #                          '%(basename)s_%(struct)s_%%(suffix)s.bp') % \
    #     {'stack':stack, 'basename':basename, 'struct':structure}
    #     return grad_fp

    @staticmethod
    def get_volume_gradient_filepath_template_v3(stack_spec, structure, **kwargs):

        if 'structure' not in stack_spec or stack_spec['structure'] is None:
            vol_basename = DataManager.get_original_volume_basename_v2(stack_spec=stack_spec)
        else:
            stack_spec_no_structure = stack_spec.copy()
            stack_spec_no_structure['structure'] = None
            vol_basename = DataManager.get_original_volume_basename_v2(stack_spec=stack_spec_no_structure)

        grad_fp = os.path.join(VOLUME_ROOTDIR, '%(stack)s',
                              '%(basename)s',
                              'score_volume_gradients',
                             '%(basename)s_%(struct)s_%%(suffix)s.bp') % \
        {'stack':stack_spec['name'], 'basename':vol_basename, 'struct':structure}
        return grad_fp

#     @staticmethod
#     def get_volume_gradient_origin_filepath(stack_spec, structure, wrt='wholebrain'):

#         if 'structure' not in stack_spec or stack_spec['structure'] is None:
#             vol_basename = DataManager.get_original_volume_basename_v2(stack_spec=stack_spec)
#         else:
#             stack_spec_no_structure = stack_spec.copy()
#             stack_spec_no_structure['structure'] = None
#             vol_basename = DataManager.get_original_volume_basename_v2(stack_spec=stack_spec_no_structure)

#         grad_fp = os.path.join(VOLUME_ROOTDIR, '%(stack)s',
#                               '%(basename)s',
#                               'score_volume_gradients',
#                              '%(basename)s_%(struct)s_origin' + ('_wrt_'+wrt if wrt is not None else '') + '.txt') % \
#         {'stack':stack_spec['name'], 'basename':vol_basename, 'struct':structure}
#         return grad_fp

    @staticmethod
    def get_volume_gradient_filepath_v3(stack_spec, structure, suffix=None):
        if suffix is None:
            if 'structure' not in stack_spec or stack_spec['structure'] is None:
                vol_basename = DataManager.get_original_volume_basename_v2(stack_spec=stack_spec)
            else:
                stack_spec_no_structure = stack_spec.copy()
                stack_spec_no_structure['structure'] = None
                vol_basename = DataManager.get_original_volume_basename_v2(stack_spec=stack_spec_no_structure)

            grad_fp = os.path.join(VOLUME_ROOTDIR, '%(stack)s',
                                  '%(basename)s',
                                  'score_volume_gradients',
                                 '%(basename)s_%(struct)s_gradients.bp') % \
            {'stack':stack_spec['name'], 'basename':vol_basename, 'struct':structure}
        else:
            grad_fp = DataManager.get_volume_gradient_filepath_template_v3(stack_spec=stack_spec, structure=structure)  % {'suffix': suffix}

        return grad_fp

    @staticmethod
    def get_volume_gradient_origin_filepath_v3(stack_spec, structure, wrt='wholebrain'):
        if 'structure' not in stack_spec or stack_spec['structure'] is None:
            vol_basename = DataManager.get_original_volume_basename_v2(stack_spec=stack_spec)
        else:
            stack_spec_no_structure = stack_spec.copy()
            stack_spec_no_structure['structure'] = None
            vol_basename = DataManager.get_original_volume_basename_v2(stack_spec=stack_spec_no_structure)

        grad_fp = os.path.join(VOLUME_ROOTDIR, '%(stack)s',
                              '%(basename)s',
                              'score_volume_gradients',
                             '%(basename)s_%(struct)s_origin' + ('_wrt_'+wrt if wrt is not None else '') + '.txt') % \
        {'stack':stack_spec['name'], 'basename':vol_basename, 'struct':structure}
        return grad_fp

    @staticmethod
    def save_volume_gradients(gradients, stack_spec, structure):
        """
        Args:
            gradients: tuple ((gx, gy, gz), origin)
        """

        assert isinstance(gradients, tuple)
        save_data(gradients[0], DataManager.get_volume_gradient_filepath_v3(stack_spec=stack_spec, structure=structure))
        save_data(gradients[1], DataManager.get_volume_gradient_origin_filepath_v3(stack_spec=stack_spec, structure=structure))

    @staticmethod
    def load_volume_gradients(stack_spec, structure):
        """
        Returns:
            tuple ((gx, gy, gz), origin)
        """

        gradients = load_data(DataManager.get_volume_gradient_filepath_v3(stack_spec=stack_spec, structure=structure))
        origin = load_data(DataManager.get_volume_gradient_origin_filepath_v3(stack_spec=stack_spec, structure=structure))
        return (gradients, origin)

#     @staticmethod
#     def load_original_volume_all_known_structures(stack, downscale=32, detector_id=None, prep_id=None,
#     structures=None, sided=True, volume_type='score', return_structure_index_mapping=True, include_surround=False):
#         """
#         Args:
#             return_structure_index_mapping (bool): if True, return both volumes and structure-label mapping. If False, return only volumes.

#         Returns:
#         """

#         if structures is None:
#             if sided:
#                 if include_surround:
#                     structures = all_known_structures_sided_with_surround
#                 else:
#                     structures = all_known_structures_sided
#             else:
#                 structures = all_known_structures

#         if return_structure_index_mapping:

#             try:
#                 label_to_structure, structure_to_label = DataManager.load_volume_label_to_name(stack=stack)
#                 loaded = True
#                 sys.stderr.write('Load structure/index map.\n')
#             except:
#                 loaded = False
#                 sys.stderr.write('Prior structure/index map not found. Generating a new one.\n')

#             volumes = {}
#             if not loaded:
#                 structure_to_label = {}
#                 label_to_structure = {}
#                 index = 1
#             for structure in sorted(structures):
#                 try:
#                     if loaded:
#                         index = structure_to_label[structure]

#                     volumes[index] = DataManager.load_original_volume(stack=stack, structure=structure,
#                                             downscale=downscale, detector_id=detector_id, prep_id=prep_id,
#                                             volume_type=volume_type)
#                     if not loaded:
#                         structure_to_label[structure] = index
#                         label_to_structure[index] = structure
#                         index += 1
#                 except Exception as e:
#                     sys.stderr.write('Score volume for %s does not exist: %s\n' % (structure, e))

#             # One volume at down=32 takes about 1MB of memory.

#             sys.stderr.write('Volume shape: (%d, %d, %d)\n' % volumes.values()[0].shape)
#             return volumes, structure_to_label, label_to_structure

#         else:
#             volumes = {}
#             for structure in structures:
#                 try:
#                     volumes[structure] = DataManager.load_original_volume(stack=stack, structure=structure,
#                                             downscale=downscale, detector_id=detector_id, prep_id=prep_id,
#                                             volume_type=volume_type)
#                 except:
#                     sys.stderr.write('Score volume for %s does not exist.\n' % structure)

#             sys.stderr.write('Volume shape: (%d, %d, %d)\n' % volumes.values()[0].shape)
#             return volumes

#     @staticmethod
#     def load_original_volume_all_known_structures(stack, downscale=32, detector_id=None, prep_id=None,
#     structures=None, sided=True, volume_type='score', return_structure_index_mapping=True, include_surround=False):
#         """
#         Args:
#             return_structure_index_mapping (bool): if True, return both volumes and structure-label mapping. If False, return only volumes.

#         Returns:
#         """

#         if structures is None:
#             if sided:
#                 if include_surround:
#                     structures = all_known_structures_sided_with_surround
#                 else:
#                     structures = all_known_structures_sided
#             else:
#                 structures = all_known_structures

#         if return_structure_index_mapping:

#             try:
#                 label_to_structure, structure_to_label = DataManager.load_volume_label_to_name(stack=stack)
#                 loaded = True
#                 sys.stderr.write('Load structure/index map.\n')
#             except:
#                 loaded = False
#                 sys.stderr.write('Prior structure/index map not found. Generating a new one.\n')

#             volumes = {}
#             if not loaded:
#                 structure_to_label = {}
#                 label_to_structure = {}
#                 index = 1
#             for structure in sorted(structures):
#                 try:
#                     if loaded:
#                         index = structure_to_label[structure]

#                     volumes[index] = DataManager.load_original_volume(stack=stack, structure=structure,
#                                             downscale=downscale, detector_id=detector_id, prep_id=prep_id,
#                                             volume_type=volume_type)
#                     if not loaded:
#                         structure_to_label[structure] = index
#                         label_to_structure[index] = structure
#                         index += 1
#                 except Exception as e:
#                     sys.stderr.write('Score volume for %s does not exist: %s\n' % (structure, e))

#             # One volume at down=32 takes about 1MB of memory.

#             sys.stderr.write('Volume shape: (%d, %d, %d)\n' % volumes.values()[0].shape)
#             return volumes, structure_to_label, label_to_structure

#         else:
#             volumes = {}
#             for structure in structures:
#                 try:
#                     volumes[structure] = DataManager.load_original_volume(stack=stack, structure=structure,
#                                             downscale=downscale, detector_id=detector_id, prep_id=prep_id,
#                                             volume_type=volume_type)
#                 except:
#                     sys.stderr.write('Score volume for %s does not exist.\n' % structure)

#             sys.stderr.write('Volume shape: (%d, %d, %d)\n' % volumes.values()[0].shape)
#             return volumes


    # @staticmethod
    # def load_original_volume_all_known_structures_v2(stack,
    #                                                  in_bbox_wrt,
    #                                                  out_bbox_wrt='wholebrain',
    #                                                 detector_id=None,
    #                                                  prep_id=None,
    #                                                 volume_type='score',
    #                                                 downscale=32,
    #                                                 structures=None,
    #                                                 sided=True,
    #                                                 include_surround=False,
    #                                                  return_label_mappings=False,
    #                                                  name_or_index_as_key='name',
    #                                                  common_shape=True
    #                                                 ):
    #     """
    #     Load original (un-transformed) volumes for all structures and optionally pad them into a common shape.
    #
    #     Args:
    #         common_shape (bool): If true, volumes are padded to the same shape.
    #
    #     Returns:
    #         If `common_shape` is True:
    #             if return_label_mappings is True, returns (volumes, common_bbox, structure_to_label, label_to_structure), volumes is dict.
    #             else, returns (volumes, common_bbox).
    #         Note that `common_bbox` is relative to the same origin the individual volumes' bounding boxes are (which, ideally, one can infer from the bbox filenames (TODO: systematic renaming)).
    #         If `common_shape` is False:
    #             if return_label_mappings is True, returns (dict of volume_bbox_tuples, structure_to_label, label_to_structure).
    #             else, returns volume_bbox_tuples.
    #     """
    #
    #     if structures is None:
    #         if sided:
    #             if include_surround:
    #                 structures = all_known_structures_sided_with_surround
    #             else:
    #                 structures = all_known_structures_sided
    #         else:
    #             structures = all_known_structures
    #
    #     loaded = False
    #     sys.stderr.write('Prior structure/index map not found. Generating a new one.\n')
    #
    #     volumes = {}
    #     if not loaded:
    #         structure_to_label = {}
    #         label_to_structure = {}
    #         index = 1
    #
    #     for structure in structures:
    #         try:
    #
    #             if loaded:
    #                 index = structure_to_label[structure]
    #
    #             v = DataManager.load_original_volume(stack=stack, volume_type=volume_type,
    #                                                     downscale=downscale,
    #                                                     prep_id=prep_id,
    #                                                     detector_id=detector_id,
    #                                                     structure=structure)
    #
    #             b = DataManager.load_original_volume_bbox(stack=stack, volume_type=volume_type,
    #                                                     downscale=downscale,
    #                                                     prep_id=prep_id,
    #                                                     detector_id=detector_id,
    #                                                       structure=structure)
    #
    #             in_bbox_origin_wrt_wholebrain = DataManager.get_domain_origin(stack=stack, domain=in_bbox_wrt)
    #             b = b + in_bbox_origin_wrt_wholebrain[[0,0,1,1,2,2]]
    #
    #             if name_or_index_as_key == 'name':
    #                 volumes[structure] = (v,b)
    #             else:
    #                 volumes[index] = (v,b)
    #
    #             if not loaded:
    #                 structure_to_label[structure] = index
    #                 label_to_structure[index] = structure
    #                 index += 1
    #
    #         except Exception as e:
    #             # raise e
    #             sys.stderr.write('Error loading score volume for %s: %s.\n' % (structure, e))
    #
    #     if common_shape:
    #         volumes_normalized, common_bbox = convert_vol_bbox_dict_to_overall_vol(vol_bbox_dict=volumes)
    #
    #         if return_label_mappings:
    #             return volumes_normalized, common_bbox, structure_to_label, label_to_structure
    #         else:
    #             return volumes_normalized, common_bbox
    #     else:
    #         if return_label_mappings:
    #             return volumes, structure_to_label, label_to_structure
    #         else:
    #             return volumes


    @staticmethod
    def load_original_volume_all_known_structures_v3(stack_spec,
                                                     in_bbox_wrt,
                                                     out_bbox_wrt='wholebrain',
                                                    structures=None,
                                                    sided=True,
                                                    include_surround=False,
                                                    surround_margin='200um',
                                                     return_label_mappings=False,
                                                     name_or_index_as_key='name',
                                                     common_shape=False,
                                                     return_origin_instead_of_bbox=True):
        """
        Load original (un-transformed) volumes for all structures and optionally pad them into a common shape.

        Args:
            common_shape (bool): If true, volumes are padded to the same shape.
            in_bbox_wrt (str): the bbox origin for the bbox files currently stored.
            loaded_cropbox_resolution (str): resolution in which the loaded cropbox is defined on.

        Returns:
            If `common_shape` is True:
                if return_label_mappings is True, returns (volumes, common_bbox, structure_to_label, label_to_structure), volumes is dict.
                else, returns (volumes, common_bbox).
            Note that `common_bbox` is relative to the same origin the individual volumes' bounding boxes are (which, ideally, one can infer from the bbox filenames (TODO: systematic renaming)).
            If `common_shape` is False:
                if return_label_mappings is True, returns (dict of volume_bbox_tuples, structure_to_label, label_to_structure).
                else, returns volume_bbox_tuples.
        """

        if structures is None:
            if sided:
                if include_surround:
                    structures = structures + [convert_to_surround_name(s, margin=surround_margin) for s in structures]
                else:
                    structures = all_known_structures_sided
            else:
                structures = all_known_structures

        loaded = False
        sys.stderr.write('Prior structure/index map not found. Generating a new one.\n')

        volumes = {}

        if not loaded:
            structure_to_label = {}
            label_to_structure = {}
            index = 1

        for structure in structures:
            try:

                if loaded:
                    index = structure_to_label[structure]

                v, o = DataManager.load_original_volume_v2(stack_spec, structure=structure, bbox_wrt=in_bbox_wrt, resolution=stack_spec['resolution'])

                in_bbox_origin_wrt_wholebrain = DataManager.get_domain_origin(stack=stack_spec['name'], domain=in_bbox_wrt,
                                                                             resolution=stack_spec['resolution'],
                                                                             loaded_cropbox_resolution=stack_spec['resolution'])
                o = o + in_bbox_origin_wrt_wholebrain

                if name_or_index_as_key == 'name':
                    volumes[structure] = (v,o)
                else:
                    volumes[index] = (v,o)

                if not loaded:
                    structure_to_label[structure] = index
                    label_to_structure[index] = structure
                    index += 1

            except Exception as e:
                # raise e
                sys.stderr.write('%s\n' % e)
                sys.stderr.write('Score volume for %s does not exist.\n' % structure)

        if common_shape:
            volumes_normalized, common_bbox = convert_vol_bbox_dict_to_overall_vol(vol_bbox_dict=volumes)

            if return_label_mappings:
                return volumes_normalized, common_bbox, structure_to_label, label_to_structure
            else:
                return volumes_normalized, common_bbox
        else:
            if return_label_mappings:
                return {k: crop_volume_to_minimal(vol=v, origin=o,
                            return_origin_instead_of_bbox=return_origin_instead_of_bbox)
                        for k, (v, o) in volumes.iteritems()}, structure_to_label, label_to_structure

            else:
                return {k: crop_volume_to_minimal(vol=v, origin=o,
                            return_origin_instead_of_bbox=return_origin_instead_of_bbox)
                        for k, (v, o) in volumes.iteritems()}



    @staticmethod
    def get_original_volume_origin_filepath_v3(stack_spec, structure, wrt='wholebrain', resolution=None):

        volume_type = stack_spec['vol_type']

        if 'resolution' not in stack_spec or stack_spec['resolution'] is None:
            assert resolution is not None
            stack_spec['resolution'] = resolution

        if 'structure' not in stack_spec or stack_spec['structure'] is None:
            vol_basename = DataManager.get_original_volume_basename_v2(stack_spec=stack_spec)
        else:
            stack_spec_no_structure = stack_spec.copy()
            stack_spec_no_structure['structure'] = None
            vol_basename = DataManager.get_original_volume_basename_v2(stack_spec=stack_spec_no_structure)

        if volume_type == 'score' or volume_type == 'annotationAsScore':
            origin_fp = os.path.join(VOLUME_ROOTDIR, '%(stack)s',
                          '%(basename)s',
                          'score_volumes',
                         '%(basename)s_%(struct)s_origin' + ('_wrt_'+wrt if wrt is not None else '') + '.txt') % \
            {'stack':stack_spec['name'], 'basename':vol_basename, 'struct':structure}

        elif volume_type == 'intensity':
            origin_fp = os.path.join(VOLUME_ROOTDIR, stack_spec['name'], vol_basename, vol_basename + '_origin' + ('_wrt_'+wrt if wrt is not None else '') + '.txt')
        else:
            raise Exception("vol_type of %s is not recognized." % stack_spec['vol_type'])

        return origin_fp


    @staticmethod
    def get_original_volume_filepath_v2(stack_spec, structure=None, resolution=None):
        """
        Args:
            stack_spec (dict): keys are:
                                - name
                                - resolution
                                - prep_id (optional)
                                - detector_id (optional)
                                - structure (optional)
                                - vol_type
        """

        if 'resolution' not in stack_spec or stack_spec['resolution'] is None:
            assert resolution is not None
            stack_spec['resolution'] = resolution

        if 'structure' not in stack_spec or stack_spec['structure'] is None:
            vol_basename = DataManager.get_original_volume_basename_v2(stack_spec=stack_spec)
        else:
            stack_spec_no_structure = stack_spec.copy()
            stack_spec_no_structure['structure'] = None
            vol_basename = DataManager.get_original_volume_basename_v2(stack_spec=stack_spec_no_structure)

        vol_basename_with_structure_suffix = vol_basename + ('_' + structure) if structure is not None else ''

        if stack_spec['vol_type'] == 'score':
            return os.path.join(VOLUME_ROOTDIR, stack_spec['name'], vol_basename, 'score_volumes', vol_basename_with_structure_suffix + '.bp')
        elif stack_spec['vol_type'] == 'annotationAsScore':
            return os.path.join(VOLUME_ROOTDIR, stack_spec['name'], vol_basename, 'score_volumes', vol_basename_with_structure_suffix + '.bp')
        elif stack_spec['vol_type'] == 'intensity':
            return os.path.join(VOLUME_ROOTDIR, stack_spec['name'], vol_basename, vol_basename + '.bp')
        else:
            raise Exception("vol_type of %s is not recognized." % stack_spec['vol_type'])

    # @staticmethod
    # def get_original_volume_filepath(stack, structure, prep_id=None, detector_id=None, volume_type='score', downscale=32):
    #     if volume_type == 'score':
    #         fp = DataManager.get_score_volume_filepath(**locals())
    #     elif volume_type == 'annotation':
    #         fp = DataManager.get_annotation_volume_filepath(stack=stack, downscale=downscale)
    #     elif volume_type == 'annotationAsScore':
    #         fp = DataManager.get_score_volume_filepath(**locals())
    #     elif volume_type == 'intensity':
    #         fp = DataManager.get_intensity_volume_filepath(stack=stack, downscale=downscale)
    #     elif volume_type == 'intensity_mhd':
    #         fp = DataManager.get_intensity_volume_mhd_filepath(stack=stack, downscale=downscale)
    #     else:
    #         raise Exception("Volume type must be one of score, annotation, annotationAsScore or intensity.")
    #     return fp


    @staticmethod
    def load_original_volume_v2(stack_spec, structure=None, resolution=None, bbox_wrt='wholebrain',
                                return_origin_instead_of_bbox=True,
                                crop_to_minimal=False):
        """
        Args:

        Returns:
            (3d-array, (6,)-tuple): (volume, bounding box wrt wholebrain)
        """

        vol_fp = DataManager.get_original_volume_filepath_v2(stack_spec=stack_spec, structure=structure, resolution=resolution)
        # download_from_s3(vol_fp, is_dir=False)
        volume = DataManager.load_data(vol_fp, filetype='bp')

        # bbox_fp = DataManager.get_original_volume_bbox_filepath_v2(stack_spec=stack_spec, structure=structure,
        #                                                            resolution=resolution, wrt=bbox_wrt)
        # download_from_s3(bbox_fp)
        # volume_bbox = DataManager.load_data(bbox_fp, filetype='bbox')

        origin = load_data(DataManager.get_original_volume_origin_filepath_v3(stack_spec=stack_spec, structure=structure, wrt=bbox_wrt, resolution=resolution))

        if crop_to_minimal:
            volume, origin = crop_volume_to_minimal(vol=volume, origin=origin, return_origin_instead_of_bbox=True)

        # if return_origin_instead_of_bbox:
        return volume, origin
        # else:
        #     convert_frame
        #     return volume, volume_bbox


    # @staticmethod
    # def load_original_volume_v2(stack, structure, downscale, prep_id=None, detector_id=None, volume_type='score'):
    #     """
    #     Args:
    #
    #     Returns:
    #         (3d-array, (6,)-tuple): (volume, bounding box with respect to coordinates origin of the contours)
    #     """
    #     vol_fp = DataManager.get_original_volume_filepath(**locals())
    #     download_from_s3(vol_fp, is_dir=False)
    #     volume = DataManager.load_data(vol_fp, filetype='bp')
    #     if volume_type == 'annotationAsScore':
    #         volume = volume.astype(np.float32)
    #
    #     bbox_fp = DataManager.get_original_volume_bbox_filepath(**locals())
    #     download_from_s3(bbox_fp)
    #     volume_bbox = DataManager.load_data(bbox_fp, filetype='bbox')
    #
    #     return volume, volume_bbox

    # @staticmethod
    # def load_original_volume(stack, structure, downscale, prep_id=None, detector_id=None, volume_type='score'):
    #     """
    #     Args:
    #     """
    #     vol_fp = DataManager.get_original_volume_filepath(**locals())
    #     download_from_s3(vol_fp, is_dir=False)
    #     volume = DataManager.load_data(vol_fp, filetype='bp')
    #     if volume_type == 'annotationAsScore':
    #         volume = volume.astype(np.float32)
    #     return volume

    # OBSOLETE
    @staticmethod
    def load_original_volume_bbox(stack, volume_type, prep_id=None, detector_id=None, structure=None, downscale=32,
                                 relative_to_uncropped=False):
        """
        This returns the 3D bounding box of the volume.
        (?) Bounding box coordinates are with respect to coordinates origin of the contours. (?)

        Args:
            volume_type (str): score or annotationAsScore.
            relative_to_uncropped (bool): if True, the returned bounding box is with respect to "wholebrain"; if False, wrt "wholebrainXYcropped". Default is False.

        Returns:
            (6-tuple): bounding box of the volume (xmin, xmax, ymin, ymax, zmin, zmax).
        """

        bbox_fp = DataManager.get_original_volume_bbox_filepath(**locals())
        # download_from_s3(bbox_fp)
        volume_bbox_wrt_wholebrainXYcropped = DataManager.load_data(bbox_fp, filetype='bbox')
        # for volume type "score" or "thumbnail", bbox of the loaded volume wrt "wholebrainXYcropped".
        # for volume type "annotationAsScore", bbox on file is wrt wholebrain.

        if relative_to_uncropped:
            if volume_type == 'score' or volume_type == 'thumbnail':
                # bbox of "brainstem" wrt "wholebrain"
                brainstem_bbox_wrt_wholebrain = DataManager.get_crop_bbox_rel2uncropped(stack=stack)
                volume_bbox_wrt_wholebrain = np.r_[volume_bbox_wrt_wholebrainXYcropped[:4] + brainstem_bbox_wrt_wholebrain[[0,0,2,2]], brainstem_bbox_wrt_wholebrain[4:]]
                return volume_bbox_wrt_wholebrain
            # else:
            #     continue
                # raise

        return volume_bbox_wrt_wholebrainXYcropped

    # @staticmethod
    # def load_original_volume_bbox_v2(stack_spec, structure=None, wrt='wholebrain', **kwargs):
    #     """
    #     """
    #
    #     bbox_fp = DataManager.get_original_volume_bbox_filepath_v2(**locals())
    #     download_from_s3(bbox_fp)
    #     return np.loadtxt(bbox_fp)

#     @staticmethod
#     def get_original_volume_origin_filepath_v2(stack_spec, structure=None, wrt='wholebrain', **kwargs):
#         """
#         """

#         volume_type = stack_spec['vol_type']
#         if volume_type == 'score':
#             origin_fp = DataManager.get_score_volume_origin_filepath_v3(stack_spec=stack_spec, structure=structure, wrt=wrt)
#         elif volume_type == 'annotationAsScore':
#             origin_fp = DataManager.get_score_volume_origin_filepath_v3(stack_spec=stack_spec, structure=structure, wrt=wrt)
#         elif volume_type == 'intensity':

#             if 'resolution' not in stack_spec or stack_spec['resolution'] is None:
#                 assert resolution is not None
#                 stack_spec['resolution'] = resolution

#             if 'structure' not in stack_spec or stack_spec['structure'] is None:
#                 vol_basename = DataManager.get_original_volume_basename_v2(stack_spec=stack_spec)
#             else:
#                 stack_spec_no_structure = stack_spec.copy()
#                 stack_spec_no_structure['structure'] = None
#                 vol_basename = DataManager.get_original_volume_basename_v2(stack_spec=stack_spec_no_structure)

#             origin_fp = os.path.join(VOLUME_ROOTDIR, stack_spec['name'], vol_basename + '_origin' + + ('_wrt_'+wrt if wrt is not None else '') + '.txt')
#         # elif volume_type == 'shell':
#         #     raise
#         # elif volume_type == 'thumbnail':
#         #     raise
#         else:
#             raise Exception("vol_type of %s is not recognized." % stack_spec['vol_type'])

#         return origin_fp

    @staticmethod
    def save_original_volume(volume, stack_spec, structure=None, wrt='wholebrain', **kwargs):
        """
        Args:
            volume: any representation
        """

        vol, ori = convert_volume_forms(volume=volume, out_form=("volume", "origin"))

        save_data(vol, DataManager.get_original_volume_filepath_v2(stack_spec=stack_spec, structure=structure))
        save_data(ori, DataManager.get_original_volume_origin_filepath_v3(stack_spec=stack_spec, structure=structure, wrt=wrt))

    # @staticmethod
    # def get_original_volume_bbox_filepath_v2(stack_spec, structure=None, wrt='wholebrain', **kwargs):
    #     volume_type = stack_spec['vol_type']
    #     if volume_type == 'annotation':
    #         raise
    #     elif volume_type == 'score':
    #         bbox_fn = DataManager.get_score_volume_bbox_filepath_v3(stack_spec=stack_spec, structure=structure, wrt=wrt)
    #     elif volume_type == 'annotationAsScore':
    #         bbox_fn = DataManager.get_score_volume_bbox_filepath_v3(stack_spec=stack_spec, structure=structure, wrt=wrt)
    #     elif volume_type == 'shell':
    #         raise
    #     elif volume_type == 'thumbnail':
    #         raise
    #     else:
    #         raise Exception('Type must be annotation, score, shell or thumbnail.')
    #
    #     return bbox_fn

    # OBSOLETE
    @staticmethod
    def get_original_volume_bbox_filepath(stack,
                                detector_id=None,
                                          prep_id=None,
                                downscale=32,
                                 volume_type='score',
                                structure=None, **kwargs):
        if volume_type == 'annotation':
            bbox_fn = DataManager.get_annotation_volume_bbox_filepath(stack=stack)
        elif volume_type == 'score':
            bbox_fn = DataManager.get_score_volume_bbox_filepath(**locals())
        elif volume_type == 'annotationAsScore':
            bbox_fn = DataManager.get_score_volume_bbox_filepath(**locals())
        elif volume_type == 'shell':
            bbox_fn = DataManager.get_shell_bbox_filepath(stack, structure, downscale)
        elif volume_type == 'thumbnail':
            bbox_fn = DataManager.get_score_volume_bbox_filepath(stack=stack, structure='7N', downscale=downscale,
            detector_id=detector_id)
        else:
            raise Exception('Type must be annotation, score, shell or thumbnail.')

        return bbox_fn

    @staticmethod
    def get_shell_bbox_filepath(stack, label, downscale):
        bbox_filepath = VOLUME_ROOTDIR + '/%(stack)s/%(stack)s_down%(ds)d_outerContourVolume_bbox.txt' % \
                        dict(stack=stack, ds=downscale)
        return bbox_filepath


    #########################
    ###     Score map     ###
    #########################

    @staticmethod
    def get_image_version_str(stack, version, resolution='lossless', downscale=None, anchor_fn=None):

        if resolution == 'thumbnail':
            downscale = 32

        if anchor_fn is None:
            anchor_fn = metadata_cache['anchor_fn'][stack]

        basename = resolution + '_alignedTo_' + anchor_fn + '_' + version + '_down' + str(downscale)
        return basename

    @staticmethod
    def get_scoremap_viz_filepath(stack, downscale, detector_id, prep_id=2, section=None, fn=None, structure=None):

        if section is not None:
            fn = metadata_cache['sections_to_filenames'][stack][section]
            if is_invalid(fn):
                raise Exception('Section is invalid: %s.' % fn)

        scoremap_viz_fp = os.path.join(SCOREMAP_VIZ_ROOTDIR, 'down%(smdown)d',
                                       '%(struct)s', '%(stack)s', 'detector%(detector_id)d',
                                       'prep%(prep)s', '%(fn)s_prep%(prep)d_down%(smdown)d_%(struct)s_detector%(detector_id)s_scoremapViz.jpg') % {'stack':stack, 'struct':structure, 'smdown':downscale, 'prep':prep_id, 'fn':fn, 'detector_id':detector_id}

        return scoremap_viz_fp

    @staticmethod
    def get_scoremap_viz_filepath_v2(stack, out_resolution, detector_id, prep_id=2,
                                     section=None, fn=None, structure=None):
        """
        Args:
            out_resolution (str): e.g. 10.0um or down32
        """

        if section is not None:
            fn = metadata_cache['sections_to_filenames'][stack][section]
            if is_invalid(fn):
                raise Exception('Section is invalid: %s.' % fn)

        scoremap_viz_fp = os.path.join(SCOREMAP_VIZ_ROOTDIR, '%(outres)s',
                                       '%(struct)s', '%(stack)s', 'detector%(detector_id)d',
                                       'prep%(prep)s', '%(fn)s_prep%(prep)d_%(outres)s_%(struct)s_detector%(detector_id)s_scoremapViz.jpg') % {'stack':stack, 'struct':structure, 'outres':out_resolution, 'prep':prep_id, 'fn':fn, 'detector_id':detector_id}

        return scoremap_viz_fp

    @staticmethod
    def load_scoremap_viz_v2(stack, out_resolution, detector_id, prep_id=2,
                                     section=None, fn=None, structure=None):
        """
        Args:
            out_resolution (str): e.g. 10.0um or down32
        """
        viz_fp = DataManager.get_scoremap_viz_filepath_v2(**locals())
        # download_from_s3(viz_fp)
        viz = imread(viz_fp)
        return viz

    @staticmethod
    def get_downscaled_scoremap_filepath(stack, structure, detector_id,
                                         out_resolution_um=None, downscale=None,
                                         prep_id=2, section=None, fn=None):
        """
        Args:
            out_resolution_um (float):
        """

        if isinstance(prep_id, str):
            prep_id = prep_str_to_id_2d[prep_id]

        if section is not None:
            fn = metadata_cache['sections_to_filenames'][stack][section]
            if is_invalid(fn):
                raise Exception('Section is invalid: %s.' % fn)

        if downscale is not None:
            scoremap_bp_filepath = os.path.join(SCOREMAP_ROOTDIR, 'down%(smdown)d',
                                            '%(stack)s',
                                            '%(stack)s_prep%(prep)d_down%(smdown)d_detector%(detector_id)d',
                                           '%(fn)s_prep%(prep)d_down%(smdown)d_detector%(detector_id)d',
                                           '%(fn)s_prep%(prep)d_down%(smdown)d_detector%(detector_id)d_%(structure)s_scoremap.bp') % {'stack':stack, 'prep':prep_id, 'fn': fn, 'smdown':downscale, 'detector_id': detector_id, 'structure':structure}
        elif out_resolution_um is not None:
            scoremap_bp_filepath = os.path.join(SCOREMAP_ROOTDIR, '%(outres).1fum', '%(stack)s',
                                                '%(stack)s_prep%(prep)d_%(outres).1fum_detector%(detector_id)d',
                                                '%(fn)s_prep%(prep)d_%(outres).1fum_detector%(detector_id)d',
                                                '%(fn)s_prep%(prep)d_%(outres).1fum_detector%(detector_id)d_%(structure)s_scoremap.bp') % {'stack':stack, 'prep':prep_id, 'fn': fn, 'outres':out_resolution_um, 'detector_id': detector_id, 'structure':structure}

        return scoremap_bp_filepath

    @staticmethod
    def load_downscaled_scoremap(stack, structure, detector_id,
                                 out_resolution_um=None, downscale=None,
                                 prep_id=2, section=None, fn=None):
        """
        Return scoremap as bp file.
        """

        scoremap_bp_filepath = DataManager.get_downscaled_scoremap_filepath(**locals())
        # download_from_s3(scoremap_bp_filepath)

        if not os.path.exists(scoremap_bp_filepath):
            raise Exception('No scoremap for image %s (section %d) for label %s\n' % \
            (metadata_cache['sections_to_filenames'][stack][section], section, structure))

        scoremap_downscaled = DataManager.load_data(scoremap_bp_filepath, filetype='bp')
        return scoremap_downscaled

    @staticmethod
    def load_scoremap(stack, structure, detector_id, downscale, prep_id=2, section=None, fn=None):
        """
        Return scoremap as bp file.
        """
        return DataManager.load_downscaled_scoremap(**locals())


#     @staticmethod
#     def load_downscaled_scoremap(stack, structure, classifier_id, section=None, fn=None, anchor_fn=None, downscale=32):
#         """
#         Return scoremaps as bp files.
#         """

#         # Load scoremap
#         scoremap_bp_filepath = DataManager.get_downscaled_scoremap_filepath(stack, section=section, \
#                         fn=fn, anchor_fn=anchor_fn, structure=structure, classifier_id=classifier_id,
#                         downscale=downscale)

#         download_from_s3(scoremap_bp_filepath)

#         if not os.path.exists(scoremap_bp_filepath):
#             raise Exception('No scoremap for image %s (section %d) for label %s\n' % \
#             (metadata_cache['sections_to_filenames'][stack][section], section, structure))

#         scoremap_downscaled = DataManager.load_data(scoremap_bp_filepath, filetype='bp')
#         return scoremap_downscaled

#     @staticmethod
#     def get_scoremap_filepath(stack, structure, classifier_id, section=None, fn=None, anchor_fn=None, return_bbox_fp=False):

#         if section is not None:
#             fn = metadata_cache['sections_to_filenames'][stack][section]
#             if is_invalid(fn):
#                 raise Exception('Section is invalid: %s.' % fn)

#         if anchor_fn is None:
#             anchor_fn = metadata_cache['anchor_fn'][stack]

#         scoremap_bp_filepath = os.path.join(SCOREMAPS_ROOTDIR, stack, \
#         '%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_%(structure)s_denseScoreMap_setting_%(classifier_id)d.hdf') \
#         % dict(stack=stack, fn=fn, structure=structure, anchor_fn=anchor_fn, classifier_id=classifier_id)

#         scoremap_bbox_filepath = os.path.join(SCOREMAPS_ROOTDIR, stack, \
#         '%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_%(structure)s_denseScoreMap_interpBox.txt') \
#             % dict(stack=stack, fn=fn, structure=structure, anchor_fn=anchor_fn)

#         if return_bbox_fp:
#             return scoremap_bp_filepath, scoremap_bbox_filepath
#         else:
#             return scoremap_bp_filepath

#     @staticmethod
#     def load_scoremap(stack, structure, classifier_id, section=None, fn=None, anchor_fn=None, downscale=1):
#         """
#         Return scoremaps.
#         """

#         # Load scoremap
#         scoremap_bp_filepath, scoremap_bbox_filepath = DataManager.get_scoremap_filepath(stack, section=section, \
#                                     fn=fn, anchor_fn=anchor_fn, structure=structure, return_bbox_fp=True, classifier_id=classifier_id)
#         if not os.path.exists(scoremap_bp_filepath):
#             raise Exception('No scoremap for image %s (section %d) for label %s\n' % \
#             (metadata_cache['sections_to_filenames'][stack][section], section, structure))

#         scoremap = DataManager.load_data(scoremap_bp_filepath, filetype='hdf')

#         # Load interpolation box
#         xmin, xmax, ymin, ymax = DataManager.load_data(scoremap_bbox_filepath, filetype='bbox')
#         ymin_downscaled = ymin / downscale
#         xmin_downscaled = xmin / downscale

#         full_width, full_height = metadata_cache['image_shape'][stack]
#         scoremap_downscaled = np.zeros((full_height/downscale, full_width/downscale), np.float32)

#         # To conserve memory, it is important to make a copy of the sub-scoremap and delete the original scoremap
#         scoremap_roi_downscaled = scoremap[::downscale, ::downscale].copy()
#         del scoremap

#         h_downscaled, w_downscaled = scoremap_roi_downscaled.shape

#         scoremap_downscaled[ymin_downscaled : ymin_downscaled + h_downscaled,
#                             xmin_downscaled : xmin_downscaled + w_downscaled] = scoremap_roi_downscaled

#         return scoremap_downscaled

    ###########################
    ######  CNN Features ######
    ###########################

    # @staticmethod
    # def load_dnn_feature_locations(stack, model_name, section=None, fn=None, anchor_fn=None, input_img_version='cropped_gray'):
    #     fp = DataManager.get_dnn_feature_locations_filepath(stack=stack, model_name=model_name, section=section, fn=fn, anchor_fn=anchor_fn, input_img_version=input_img_version)
    #     download_from_s3(fp)
    #     locs = np.loadtxt(fp).astype(np.int)
    #     indices = locs[:, 0]
    #     locations = locs[:, 1:]
    #     return indices, locations

    # @staticmethod
    # def load_dnn_feature_locations(stack, model_name, section=None, fn=None, prep_id=2, win=1, input_img_version='gray'):
    #     fp = DataManager.get_dnn_feature_locations_filepath(stack=stack, model_name=model_name, section=section, fn=fn, prep_id=prep_id, input_img_version=input_img_version, win=win)
    #     download_from_s3(fp)
    #     locs = np.loadtxt(fp).astype(np.int)
    #     indices = locs[:, 0]
    #     locations = locs[:, 1:]
    #     return indices, locations

    @staticmethod
    def load_patch_locations(stack, win, section=None, fn=None, prep_id=2, input_img_version='gray'):
        fp = DataManager.get_patch_locations_filepath(**locals())
        download_from_s3(fp)
        locs = np.loadtxt(fp).astype(np.int)
        indices = locs[:, 0]
        locations = locs[:, 1:]
        return indices, locations

#     @staticmethod
#     def get_dnn_feature_locations_filepath(stack, model_name, section=None, fn=None, anchor_fn=None, input_img_version='cropped_gray'):

#         if fn is None:
#             fn = metadata_cache['sections_to_filenames'][stack][section]

#         if anchor_fn is None:
#             anchor_fn = metadata_cache['anchor_fn'][stack]

#         image_version_basename = DataManager.get_image_version_basename(stack=stack, resol='lossless', version=input_img_version)
#         image_basename = DataManager.get_image_basename(stack=stack, fn=fn, resol='lossless', version=input_img_version)

#         # feature_locs_fn = os.path.join(PATCH_FEATURES_ROOTDIR, model_name, stack, \
#         # '%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_patch_locations.txt' % \
#         # dict(fn=fn, anchor_fn=anchor_fn))

#         feature_locs_fn = os.path.join(PATCH_FEATURES_ROOTDIR, model_name, stack, image_version_basename,
#                                        image_basename + '_patch_locations.txt')
#         return feature_locs_fn

#     @staticmethod
#     def get_dnn_feature_locations_filepath(stack, model_name, section=None, fn=None, prep_id=2, input_img_version='gray', win=1):

#         if fn is None:
#             fn = metadata_cache['sections_to_filenames'][stack][section]
#         feature_locs_fp = os.path.join(PATCH_FEATURES_ROOTDIR, model_name, stack,
#                                        stack+'_prep%(prep)d'%{'prep':prep_id}+'_'+input_img_version+'_win%(win)d'%{'win':win},
#                                        fn+'_prep%(prep)d'%{'prep':prep_id}+'_'+input_img_version+'_win%(win)d'%{'win':win}+'_'+model_name+'_patch_locations.txt')
#         return feature_locs_fp

#     @staticmethod
#     def get_dnn_feature_locations_filepath_v2(stack, section=None, fn=None, prep_id=2, input_img_version='gray', win=1):

#         if fn is None:
#             fn = metadata_cache['sections_to_filenames'][stack][section]
#         feature_locs_fp = os.path.join(PATCH_FEATURES_ROOTDIR, stack,
#                                        stack+'_prep%(prep)d'%{'prep':prep_id}+'_'+input_img_version+'_win%(win)d'%{'win':win},
#                                        fn+'_prep%(prep)d'%{'prep':prep_id}+'_'+input_img_version+'_win%(win)d'%{'win':win}+'_patchLocations.txt')
#         return feature_locs_fp

    @staticmethod
    def get_patch_locations_filepath(stack, win, section=None, fn=None, prep_id=2, input_img_version='gray'):

        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][section]
        feature_locs_fp = os.path.join(PATCH_LOCATIONS_ROOTDIR, stack,
                                       stack+'_prep%(prep)d'%{'prep':prep_id}+'_'+input_img_version+'_win%(win)d'%{'win':win},
                                       fn+'_prep%(prep)d'%{'prep':prep_id}+'_'+input_img_version+'_win%(win)d'%{'win':win}+'_patchLocations.txt')
        return feature_locs_fp

#     @staticmethod
#     def get_patch_locations_filepath_v2(stack, win, section=None, fn=None, prep_id=2, input_img_version='gray'):

#         if fn is None:
#             fn = metadata_cache['sections_to_filenames'][stack][section]
#         feature_locs_fp = os.path.join(PATCH_LOCATIONS_ROOTDIR, stack,
#                                        stack+'_prep%(prep)d'%{'prep':prep_id}+'_'+input_img_version+'_win%(win)d'%{'win':win},
#                                        fn+'_prep%(prep)d'%{'prep':prep_id}+'_'+input_img_version+'_win%(win)d'%{'win':win}+'_patchLocations.txt')
#         return feature_locs_fp

#     @staticmethod
#     def get_dnn_features_filepath(stack, model_name, section=None, fn=None, anchor_fn=None, input_img_version='cropped_gray'):
#         """
#         Args:
#             version (str): default is cropped_gray.
#         """

#         if fn is None:
#             fn = metadata_cache['sections_to_filenames'][stack][section]

#         if anchor_fn is None:
#             anchor_fn = metadata_cache['anchor_fn'][stack]

#         image_version_basename = DataManager.get_image_version_basename(stack=stack, resol='lossless', version=input_img_version)
#         image_basename = DataManager.get_image_basename(stack=stack, fn=fn, resol='lossless', version=input_img_version)

#         feature_fn = os.path.join(PATCH_FEATURES_ROOTDIR, model_name, stack, image_version_basename, image_basename + '_features.bp')

#         return feature_fn


    @staticmethod
    def get_dnn_features_filepath_v2(stack, prep_id, win_id,
                              normalization_scheme,
                                             model_name, what='features',
                                    sec=None, fn=None, timestamp=None):
        """
        Args:
            what (str): "features" or "locations"
        """

        if timestamp == 'now':
            timestamp = datetime.now().strftime("%m%d%Y%H%M%S")

        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][sec]

        prep_str = 'prep%(prep)d' % {'prep':prep_id}
        win_str = 'win%(win)d' % {'win':win_id}

        feature_fp = os.path.join(PATCH_FEATURES_ROOTDIR, model_name, stack,
                    stack + '_' + prep_str + '_' + normalization_scheme + '_' + win_str,
            fn + '_' + prep_str + '_' + normalization_scheme + '_' + win_str + '_' + model_name + '_' + what + ('_%s'%timestamp if timestamp is not None else '') + '.bp')

        return feature_fp

    @staticmethod
    def load_dnn_features_v2(stack, prep_id, win_id,
                              normalization_scheme,
                             model_name, sec=None, fn=None):
        """
        Args:
            win (int): the spacing/size scheme

        Returns:
            (features, patch center locations wrt prep=2 images)

        Note: `mean_img` is assumed to be the default provided by mxnet.
        """

        features_fp = DataManager.get_dnn_features_filepath_v2(stack=stack, sec=sec, fn=fn, prep_id=prep_id, win_id=win_id,
                              normalization_scheme=normalization_scheme,
                                             model_name=model_name, what='features')
        # download_from_s3(features_fp, local_root=DATA_ROOTDIR)
        if not os.path.exists(features_fp):
            raise Exception("Features for %s, %s/%s does not exist." % (stack, sec, fn))

        features = bp.unpack_ndarray_file(features_fp)

        locations_fp = DataManager.get_dnn_features_filepath_v2(stack=stack, sec=sec, fn=fn, prep_id=prep_id, win_id=win_id,
                              normalization_scheme=normalization_scheme,
                                             model_name=model_name, what='locations')
        # download_from_s3(locations_fp)
        locations = np.loadtxt(locations_fp).astype(np.int)

        return features, locations


    @staticmethod
    def save_dnn_features_v2(features, locations, stack, prep_id,
                             win_id, normalization_scheme, model_name, sec=None, fn=None, timestamp=None):
        """
        Args:
            features ((n,1024) array of float):
            locations ((n,2) array of int): list of (x,y) coordinates relative to prep=2 image. This matches the features list.
        """

        features_fp = DataManager.get_dnn_features_filepath_v2(stack=stack, sec=sec, fn=fn, prep_id=prep_id, win_id=win_id,
                              normalization_scheme=normalization_scheme,
                                             model_name=model_name, what='features', timestamp=timestamp)
        create_parent_dir_if_not_exists(features_fp)
        bp.pack_ndarray_file(features, features_fp)
        upload_to_s3(features_fp)

        locations_fp = DataManager.get_dnn_features_filepath_v2(stack=stack, sec=sec, fn=fn, prep_id=prep_id, win_id=win_id,
                              normalization_scheme=normalization_scheme,
                                             model_name=model_name, what='locations', timestamp=timestamp)
        np.savetxt(locations_fp, locations, fmt='%d')
        upload_to_s3(locations_fp)

#     @staticmethod
#     def get_dnn_features_filepath(stack, model_name, win, section=None, fn=None, prep_id=2, input_img_version='gray', suffix=None):
#         """
#         Args:
#             input_img_version (str): default is gray.
#         """
#         if fn is None:
#             fn = metadata_cache['sections_to_filenames'][stack][section]

#         if suffix is None:
#             feature_fp = os.path.join(PATCH_FEATURES_ROOTDIR, model_name, stack,
#                                        stack+'_prep%(prep)d'%{'prep':prep_id}+'_'+input_img_version+'_win%(win)d'%{'win':win},
#                                        fn+'_prep%(prep)d'%{'prep':prep_id}+'_'+input_img_version+'_win%(win)d'%{'win':win}+'_'+model_name+'_features.bp')
#         else:
#             feature_fp = os.path.join(PATCH_FEATURES_ROOTDIR, model_name, stack,
#                                        stack+'_prep%(prep)d'%{'prep':prep_id}+'_'+input_img_version+'_win%(win)d'%{'win':win},
#                                        fn+'_prep%(prep)d'%{'prep':prep_id}+'_'+input_img_version+'_win%(win)d'%{'win':win}+'_'+model_name+'_features_' + suffix + '.bp')

#         return feature_fp

#     @staticmethod
#     def load_dnn_features(stack, model_name, win, section=None, fn=None, input_img_version='gray', prep_id=2, suffix=None):
#         """
#         Args:
#             input_img_version (str): default is gray.
#             win (int): the spacing/size scheme
#         """

#         features_fp = DataManager.get_dnn_features_filepath(**locals())
#         download_from_s3(features_fp, local_root=DATA_ROOTDIR)

#         try:
#             return load_hdf(features_fp)
#         except:
#             pass

#         try:
#             return load_hdf_v2(features_fp)
#         except:
#             pass

#         return bp.unpack_ndarray_file(features_fp)

#     @staticmethod
#     def load_dnn_features(stack, model_name, section=None, fn=None, anchor_fn=None, input_img_version='cropped_gray'):
#         """
#         Args:
#             version (str): default is cropped_gray.
#         """

#         features_fp = DataManager.get_dnn_features_filepath(stack=stack, model_name=model_name, section=section, fn=fn, anchor_fn=anchor_fn, input_img_version=input_img_version)
#         download_from_s3(features_fp)

#         try:
#             return load_hdf(features_fp)
#         except:
#             pass

#         try:
#             return load_hdf_v2(features_fp)
#         except:
#             pass

#         return bp.unpack_ndarray_file(features_fp)




    ##################
    ##### Image ######
    ##################

    @staticmethod
    def get_image_version_basename(stack, version, resol='lossless', anchor_fn=None):

        if anchor_fn is None:
            anchor_fn = metadata_cache['anchor_fn'][stack]

        if resol == 'lossless' and (version == 'cropped' or version == 'cropped_tif'):
            image_version_basename = stack + '_' + resol + '_alignedTo_' + anchor_fn + '_cropped'
        else:
            image_version_basename = stack + '_' + resol + '_alignedTo_' + anchor_fn + '_' + version

        return image_version_basename

    @staticmethod
    def get_image_basename(stack, version, resol='lossless', anchor_fn=None, fn=None, section=None):

        if anchor_fn is None:
            anchor_fn = metadata_cache['anchor_fn'][stack]

        if section is not None:
            fn = metadata_cache['sections_to_filenames'][stack][section]
            assert is_invalid(fn=fn), 'Section is invalid: %s.' % fn

        if resol == 'lossless' and (version == 'cropped' or version == 'cropped_tif'):
            image_basename = fn + '_' + resol + '_alignedTo_' + anchor_fn + '_cropped'
        else:
            image_basename = fn + '_' + resol + '_alignedTo_' + anchor_fn + '_' + version

        return image_basename


#     @staticmethod
#     def get_image_basename_v2(stack, version, resol='lossless', anchor_fn=None, fn=None, section=None):

#         if anchor_fn is None:
#             anchor_fn = metadata_cache['anchor_fn'][stack]

#         if section is not None:
#             fn = metadata_cache['sections_to_filenames'][stack][section]
#             assert is_invalid(fn=fn), 'Section is invalid: %s.' % fn

#         if resol == 'lossless' and (version == 'cropped' or version == 'cropped_tif'):
#             image_basename = fn + '_' + resol + '_alignedTo_' + anchor_fn + '_cropped'
#         else:
#             image_basename = fn + '_' + resol + '_alignedTo_' + anchor_fn + '_' + version

#         return image_basename

    @staticmethod
    def get_image_dir_v2(stack, prep_id=None, version=None, resol='lossless',
                      data_dir=DATA_DIR, raw_data_dir=RAW_DATA_DIR, thumbnail_data_dir=THUMBNAIL_DATA_DIR):
        """
        Args:
            version (str): version string
            data_dir: This by default is DATA_DIR, but one can change this ad-hoc when calling the function

        Returns:
            Absolute path of the image directory.
        """

        if prep_id is not None and isinstance(prep_id, str):
            prep_id = prep_str_to_id_2d[prep_id]

        if version is None:
            if resol == 'thumbnail' or resol == 'down64':
                image_dir = os.path.join(thumbnail_data_dir, stack, stack + ('_prep%d' % prep_id if prep_id is not None else '') + '_%s' % resol)
            else:
                image_dir = os.path.join(data_dir, stack, stack + ('_prep%d' % prep_id if prep_id is not None else '') + '_%s' % resol)
        else:
            if resol == 'thumbnail' or resol == 'down64':
                image_dir = os.path.join(thumbnail_data_dir, stack, stack + ('_prep%d' % prep_id if prep_id is not None else '') + '_%s' % resol + '_' + version)
            else:
                image_dir = os.path.join(data_dir, stack, stack + ('_prep%d' % prep_id if prep_id is not None else '') + '_%s' % resol + '_' + version)

        return image_dir


    # @staticmethod
    # def get_image_dir(stack, version, resol='lossless', anchor_fn=None, modality=None,
    #                   data_dir=DATA_DIR, raw_data_dir=RAW_DATA_DIR, thumbnail_data_dir=THUMBNAIL_DATA_DIR):
    #     """
    #     Args:
    #         data_dir: This by default is DATA_DIR, but one can change this ad-hoc when calling the function
    #         resol: can be either lossless or thumbnail
    #         version: TODO - Write a list of options
    #         modality: can be either nissl or fluorescent. If not specified, it is inferred.
    #
    #     Returns:
    #         Absolute path of the image directory.
    #     """
    #
    #     if anchor_fn is None:
    #         anchor_fn = DataManager.load_anchor_filename(stack)
    #
    #     if resol == 'lossless' and version == 'original_jp2':
    #         image_dir = os.path.join(raw_data_dir, stack)
    #     elif resol == 'lossless' and version == 'jpeg':
    #         assert modality == 'nissl'
    #         image_dir = os.path.join(data_dir, stack, stack+'_'+resol+'_alignedTo_%(anchor_fn)s_cropped_compressed' % {'anchor_fn':anchor_fn})
    #     elif resol == 'lossless' and version == 'uncropped_tif':
    #         image_dir = os.path.join(data_dir, stack, stack + '_' + resol + '_tif')
    #     elif resol == 'lossless' and version == 'cropped_16bit':
    #         image_dir = os.path.join(data_dir, stack, stack+'_'+resol+'_alignedTo_%(anchor_fn)s_cropped' % {'anchor_fn':anchor_fn})
    #     elif resol == 'thumbnail' and (version == 'cropped' or version == 'cropped_tif'):
    #         image_dir = os.path.join(thumbnail_data_dir, stack, stack+'_'+resol+'_alignedTo_%(anchor_fn)s_cropped' % {'anchor_fn':anchor_fn})
    #     elif (resol == 'thumbnail' and version == 'aligned') or (resol == 'thumbnail' and version == 'aligned_tif'):
    #         image_dir = os.path.join(thumbnail_data_dir, stack, stack+'_'+resol+'_alignedTo_%(anchor_fn)s' % {'anchor_fn':anchor_fn})
    #     elif resol == 'thumbnail' and version == 'original_png':
    #         image_dir = os.path.join(raw_data_dir, stack)
    #     else:
    #         # sys.stderr.write('No special rule for (%s, %s). So using the default image directory composition rule.\n' % (version, resol))
    #         image_dir = os.path.join(data_dir, stack, stack + '_' + resol + '_alignedTo_' + anchor_fn + '_' + version)
    #
    #     return image_dir

    @staticmethod
    def load_image_v2(stack, prep_id, resol='raw', version=None, section=None, fn=None, data_dir=DATA_DIR, ext=None, thumbnail_data_dir=THUMBNAIL_DATA_DIR):

        img_fp = DataManager.get_image_filepath_v2(stack=stack, prep_id=prep_id, resol=resol, version=version,
                                                       section=section, fn=fn, data_dir=data_dir, ext=ext,
                                                      thumbnail_data_dir=thumbnail_data_dir)

        sys.stderr.write("Trying to load %s\n" % img_fp)

        if not os.path.exists(img_fp):
            sys.stderr.write('File not on local disk. Download from S3.\n')
            # if resol == 'lossless' or resol == 'raw' or resol == 'down8':
            #     download_from_s3(img_fp, local_root=DATA_ROOTDIR)
            # else:
            #     download_from_s3(img_fp, local_root=THUMBNAIL_DATA_ROOTDIR)

        global use_image_cache
        if use_image_cache:
            args_tuple = tuple(locals().values())
            if args_tuple in image_cache:
                sys.stderr.write("Loaded image from image_cache.\n")
                img = image_cache[args_tuple]
            else:
                img = cv2.imread(img_fp, -1)
                if img is None:
                    img = imread(img_fp, -1)
                    # if img is None:
                    #     raise Exception("Image loading returns None: %s" % img_fp)

                if img is not None:
                    image_cache[args_tuple] = img
                    sys.stderr.write("Image %s is now cached.\n" % os.path.basename(img_fp))
        else:
            # sys.stderr.write("Not using image_cache.\n")
            img = cv2.imread(img_fp, -1)
            if img is None:
                sys.stderr.write("cv2.imread fails to load. Try skimage.imread.\n")
                try:
                    img = imread(img_fp, -1)
                except:
                    sys.stderr.write("skimage.imread fails to load.\n")
                    img = None

        if img is None:
            sys.stderr.write("Image fails to load. Trying to convert from other resol/versions.\n")

            if resol != 'raw':
                try:
                    sys.stderr.write("Resolution %s is not available. Instead, try loading raw and then downscale...\n" % resol)
                    img = DataManager.load_image_v2(stack=stack, prep_id=prep_id, resol='raw', version=version, section=section, fn=fn, data_dir=data_dir, ext=ext, thumbnail_data_dir=thumbnail_data_dir)

                    downscale_factor = convert_resolution_string_to_um(resolution='raw', stack=stack)/convert_resolution_string_to_um(resolution=resol, stack=stack)
                    img = rescale_by_resampling(img, downscale_factor)
                except:
                    sys.stderr.write('Cannot load raw either.')

                    if version == 'blue':
                        img = DataManager.load_image_v2(stack=stack, prep_id=prep_id, resol=resol, version=None, section=section, fn=fn, data_dir=data_dir, ext=ext, thumbnail_data_dir=thumbnail_data_dir)[..., 2]
                    elif version == 'grayJpeg':
                        sys.stderr.write("Version %s is not available. Instead, load raw RGB JPEG and convert to uint8 grayscale...\n" % version)
                        img = DataManager.load_image_v2(stack=stack, prep_id=prep_id, resol=resol, version='jpeg', section=section, fn=fn, data_dir=data_dir, ext=ext, thumbnail_data_dir=thumbnail_data_dir)
                        img = img_as_ubyte(rgb2gray(img))
                    elif version == 'gray':
                        sys.stderr.write("Version %s is not available. Instead, load raw RGB and convert to uint8 grayscale...\n" % version)
                        img = DataManager.load_image_v2(stack=stack, prep_id=prep_id, resol=resol, version=None, section=section, fn=fn, data_dir=data_dir, ext=ext, thumbnail_data_dir=thumbnail_data_dir)
                        img = img_as_ubyte(rgb2gray(img))
                    elif version == 'Ntb':
                        sys.stderr.write("Version %s is not available. Instead, load lossless and take the blue channel...\n" % version)
                        img = DataManager.load_image_v2(stack=stack, prep_id=prep_id, resol=resol, version=None, section=section, fn=fn, data_dir=data_dir, ext=ext, thumbnail_data_dir=thumbnail_data_dir)
                        img = img[..., 2]
                    elif version == 'mask' and (resol == 'down32' or resol == 'thumbnail'):
                        if isinstance(prep_id, str):
                            prep_id = prep_str_to_id_2d[prep_id]

                        if prep_id == 5:
                            sys.stderr.write('Cannot load mask %s, section=%s, fn=%s, prep=%s\n' % (stack, section, fn, prep_id))
                            sys.stderr.write('Try finding prep1 masks.\n')
                            mask_prep1 = DataManager.load_image_v2(stack=stack, section=section, fn=fn, prep_id=1, version='mask', resol='thumbnail')
                            xmin,xmax,ymin,ymax = DataManager.load_cropbox_v2(stack=stack, prep_id=prep_id, return_dict=False, only_2d=True)
                            mask_prep2 = mask_prep1[ymin:ymax+1, xmin:xmax+1].copy()
                            return mask_prep2.astype(np.bool)

                        elif prep_id == 2:
                            # get prep 2 masks directly from prep 5 masks.
                            try:
                                sys.stderr.write('Try finding prep5 masks.\n')
                                mask_prep5 = DataManager.load_image_v2(stack=stack, section=section, fn=fn, prep_id=5, version='mask', resol='thumbnail')
                                xmin,xmax,ymin,ymax = DataManager.load_cropbox_v2_relative(stack=stack, prep_id=prep_id, wrt_prep_id=5, out_resolution='down32')
                                mask_prep2 = mask_prep5[ymin:ymax+1, xmin:xmax+1].copy()
                                return mask_prep2.astype(np.bool)
                            except:
                                # get prep 2 masks directly from prep 1 masks.
                                sys.stderr.write('Cannot load mask %s, section=%s, fn=%s, prep=%s\n' % (stack, section, fn, prep_id))
                                sys.stderr.write('Try finding prep1 masks.\n')
                                mask_prep1 = DataManager.load_image_v2(stack=stack, section=section, fn=fn, prep_id=1, version='mask', resol='thumbnail')
                                xmin,xmax,ymin,ymax = DataManager.load_cropbox_v2(stack=stack, prep_id=prep_id, return_dict=False, only_2d=True)
                                mask_prep2 = mask_prep1[ymin:ymax+1, xmin:xmax+1].copy()
                                return mask_prep2.astype(np.bool)
                        else:
                            try:
                                mask = DataManager.load_image_v2(stack=stack, section=section, fn=fn, prep_id=prep_id, version='mask', resol='down32')
                                return mask.astype(np.bool)
                            except:
                                sys.stderr.write('Cannot load mask %s, section=%s, fn=%s, prep=%s\n' % (stack, section, fn, prep_id))
                    else:
                        sys.stderr.write('Cannot load stack=%s, section=%s, fn=%s, prep=%s, version=%s, resolution=%s\n' % (stack, section, fn, prep_id, version, resol))
                        raise Exception("Image loading failed.")

        if version == 'mask':
            img = img.astype(np.bool)

        if img.ndim == 3:
            return img[...,::-1] # cv2 load images in BGR, this converts it to RGB.
        else:
            return img

    @staticmethod
    def enable_image_cache():
        global use_image_cache
        use_image_cache = True

        DataManager.clear_image_cache()

    @staticmethod
    def disable_image_cache():
        global use_image_cache
        use_image_cache = False

        DataManager.clear_image_cache()

    @staticmethod
    def clear_image_cache():
        global image_cache
        image_cache = {}

    @staticmethod
    def load_image(stack, version, resol='lossless', section=None, fn=None, anchor_fn=None, modality=None, data_dir=DATA_DIR, ext=None):
        img_fp = DataManager.get_image_filepath(**locals())
        # download_from_s3(img_fp)
        return imread(img_fp)

    @staticmethod
    def get_image_filepath_v2(stack, prep_id, version=None, resol='raw',
                           data_dir=DATA_DIR, raw_data_dir=RAW_DATA_DIR, thumbnail_data_dir=THUMBNAIL_DATA_DIR,
                           section=None, fn=None, ext=None, sorted_filenames_fp=None):
        """
        Args:
            version (str): the version string.

        Returns:
            Absolute path of the image file.
        """

        if resol == 'lossless':
            if stack == 'CHATM2' or stack == 'CHATM3':
                resol = 'raw'
        elif resol == 'raw':
            if stack not in ['CHATM2', 'CHATM3', 'MD661', 'DEMO999']:
                resol = 'lossless'

        if section is not None:
            
            if sorted_filenames_fp is not None:
                _, sections_to_filenames = DataManager.load_sorted_filenames(fp=sorted_filenames_fp)                
                fn = sections_to_filenames[section]
            else:
                fn = metadata_cache['sections_to_filenames'][stack][section]
                
            if is_invalid(fn=fn):
                raise Exception('Section is invalid: %s.' % fn)
        else:
            assert fn is not None

        if prep_id is not None and isinstance(prep_id, str):
            prep_id = prep_str_to_id_2d[prep_id]

        image_dir = DataManager.get_image_dir_v2(stack=stack, prep_id=prep_id, resol=resol, version=version, data_dir=data_dir, thumbnail_data_dir=thumbnail_data_dir)


        if version is None:
            image_name = fn + ('_prep%d' % prep_id if prep_id is not None else '') + '_%s' % resol + '.' + 'tif'
        else:
            if ext is None:
                if version == 'mask':
                    ext = 'png'
                elif version == 'contrastStretched' or version.endswith('Jpeg') or version == 'jpeg':
                    ext = 'jpg'
                else:
                    ext = 'tif'
            image_name = fn + ('_prep%d' % prep_id if prep_id is not None else '') + '_' + resol + '_' + version + '.' + ext
        image_path = os.path.join(image_dir, image_name)

        return image_path

    @staticmethod
    def get_image_filepath(stack, version, resol='lossless',
                           data_dir=DATA_DIR, raw_data_dir=RAW_DATA_DIR, thumbnail_data_dir=THUMBNAIL_DATA_DIR,
                           section=None, fn=None, anchor_fn=None, modality=None, ext=None):
        """
        Args:
            data_dir: This by default is DATA_DIR, but one can change this ad-hoc when calling the function
            resol: can be either lossless or thumbnail
            version: TODO - Write a list of options
            modality: can be either nissl or fluorescent. If not specified, it is inferred.

        Returns:
            Absolute path of the image file.
        """

        image_name = None

        if section is not None:
            fn = metadata_cache['sections_to_filenames'][stack][section]
            if is_invalid(fn=fn):
                raise Exception('Section is invalid: %s.' % fn)
        else:
            assert fn is not None

        if anchor_fn is None:
            anchor_fn = DataManager.load_anchor_filename(stack)

        if modality is None:
            if (stack in all_alt_nissl_ntb_stacks or stack in all_alt_nissl_tracing_stacks) and fn.split('-')[1][0] == 'F':
                modality = 'fluorescent'
            else:
                modality = 'nissl'

        image_dir = DataManager.get_image_dir(stack=stack, version=version, resol=resol, modality=modality, data_dir=data_dir)

        if resol == 'thumbnail' and version == 'original_png':
            image_name = fn + '.png'
        elif resol == 'thumbnail' and (version == 'cropped' or version == 'cropped_tif'):
            image_name = '_'.join([fn, resol, 'alignedTo_' + anchor_fn + '_cropped.tif'])
        elif resol == 'thumbnail' and (version == 'aligned' or version == 'aligned_tif'):
            image_name = '_'.join([fn, resol, 'alignedTo_' + anchor_fn + '.tif'])
        elif resol == 'lossless' and version == 'jpeg':
            assert modality == 'nissl'
            image_name = '_'.join([fn, resol, 'alignedTo_%(anchor_fn)s_cropped_compressed.jpg' % {'anchor_fn':anchor_fn}])
        elif resol == 'lossless' and version == 'cropped':
            image_name = '_'.join([fn, resol, 'alignedTo_%(anchor_fn)s_cropped.tif' % {'anchor_fn':anchor_fn}])
        elif resol == 'lossless' and version == 'uncropped_tif':
            image_name = fn + '_lossless.tif'
        elif resol == 'lossless' and version == 'cropped_gray_jpeg':
            image_name = fn + '_' + resol + '_alignedTo_' + anchor_fn + '_cropped_gray.jpg'
        else:
            if ext is None:
                ext = 'tif'
            image_name = '_'.join([fn, resol, 'alignedTo_' + anchor_fn + '_' + version + '.' + ext])

        image_path = os.path.join(image_dir, image_name)

        return image_path


    @staticmethod
    def get_image_dimension(stack):
        """
        Returns:
            (image width, image height).
        """

        # first_sec, last_sec = DataManager.load_cropbox(stack)[4:]
        # anchor_fn = DataManager.load_anchor_filename(stack)
        # filename_to_section, section_to_filename = DataManager.load_sorted_filenames(stack)

        xmin, xmax, ymin, ymax = DataManager.load_cropbox_v2(stack=stack, prep_id=2)
        return (xmax - xmin + 1) * 32, (ymax - ymin + 1) * 32

        # for i in range(10, 13):
#         random_fn = metadata_cache['valid_filenames'][stack][0]
#         # random_fn = section_to_filename[i]
#         # fp = DataManager.get_image_filepath(stack=stack, resol='thumbnail', version='cropped', fn=random_fn, anchor_fn=anchor_fn)
#         # try:
#         try:
#             img = DataManager.load_image_v2(stack=stack, resol='thumbnail', prep_id=2, fn=random_fn)
#         except:
#             img = DataManager.load_image_v2(stack=stack, resol='thumbnail', prep_id=2, fn=random_fn, version='Ntb')
#             # break
#         # except:
#         #     pass

#         image_height, image_width = img.shape[:2]
#         image_height = image_height * 32
#         image_width = image_width * 32

        # return image_width, image_height

    #######################################################

    @staticmethod
    def get_intensity_normalization_result_filepath(what, stack, fn=None, section=None):

        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][section]
            if is_invalid(fn=fn):
                raise Exception('Section is invalid: %s.' % fn)

        if what == 'region_centers':
            fp = os.path.join(DATA_DIR, stack, stack + '_intensity_normalization_results', 'regionCenters',
                         stack + '_' + fn + '_raw_regionCenters.bp')
        elif what == 'mean_std_all_regions':
            fp = os.path.join(DATA_DIR, stack, stack + '_intensity_normalization_results', 'meanStdAllRegions',
                         stack + '_' + fn + '_raw_meanStdAllRegions.bp')
        elif what == 'mean_map':
            fp = os.path.join(DATA_DIR, stack, stack + '_intensity_normalization_results', 'meanMap',
                         stack + '_' + fn + '_raw_meanMap.bp')
        elif what == 'std_map':
            fp = os.path.join(DATA_DIR, stack, stack + '_intensity_normalization_results', 'stdMap',
                         stack + '_' + fn + '_raw_stdMap.bp')
        elif what == 'float_histogram_png':
            fp = os.path.join(DATA_DIR, stack, stack + '_intensity_normalization_results', 'floatHistogram',
                         stack + '_' + fn + '_raw_floatHistogram.png')
        elif what == 'normalized_float_map':
            fp = os.path.join(DATA_DIR, stack, stack + '_intensity_normalization_results', 'normalizedFloatMap',
                         stack + '_' + fn + '_raw_normalizedFloatMap.bp')
        elif what == 'float_percentiles':
            fp = os.path.join(DATA_DIR, stack, stack + '_intensity_normalization_results', 'floatPercentiles',
                         stack + '_' + fn + '_raw_floatPercentiles.bp')
        else:
            raise Exception("what = %s is not recognized." % what)

        return fp


    #######################################################

    @staticmethod
    # def convert_section_to_z(sec, downsample=None, resolution=None, stack=None, first_sec=None, mid=False):
    def convert_section_to_z(sec, downsample=None, resolution=None, stack=None, mid=False, z_begin=None, first_sec=None):
        """
        Voxel size is determined by `resolution`.

        z = sec * section_thickness_in_unit_of_cubic_voxel_size - z_begin

        Physical size of a cubic voxel depends on the downsample factor.

        Args:
            downsample/resolution: this determines the voxel size.
            z_begin (float): z-coordinate of an origin. The z-coordinate of a given section is relative to this value.
                Default is the z position of the `first_sec`. This must be consistent with `downsample`.
            first_sec (int): Index of the section that defines z=0.
                Default is the first brainstem section defined in ``cropbox".
                If `stack` is given, the default is the first section of the brainstem.
                If `stack` is not given, default = 1.
            mid (bool): If false, return the z-coordinates of the two sides of the section. If true, only return a single scalar = the average.

        Returns:
            z1, z2 (2-tuple of float): the z-levels of the beginning and end of the queried section, counted from `z_begin`.
        """

        if downsample is not None:
            resolution = 'down%d' % downsample

        voxel_size_um = convert_resolution_string_to_voxel_size(resolution=resolution, stack=stack)
        section_thickness_in_voxel = SECTION_THICKNESS / voxel_size_um # Voxel size in z direction in unit of x,y pixel.
        # if first_sec is None:
        #     # first_sec, _ = DataManager.load_cropbox(stack)[4:]
        #     if stack is not None:
        #         first_sec = metadata_cache['section_limits'][stack][0]
        #     else:
        #         first_sec = 1
        #

        if z_begin is None:
            if first_sec is not None:
                z_begin = (first_sec - 1) * section_thickness_in_voxel
            else:
                z_begin = 0

        z1 = (sec-1) * section_thickness_in_voxel
        z2 = sec * section_thickness_in_voxel
        # print "z1, z2 =", z1, z2

        if mid:
            return np.mean([z1-z_begin, z2-1-z_begin])
        else:
            return z1-z_begin, z2-1-z_begin

    @staticmethod
    def convert_z_to_section(z, downsample=None, resolution=None, z_first_sec=None, sec_z0=None, stack=None):
        """
        Convert z coordinate to section index.

        Args:
            resolution (str): planar resolution
            z_first_sec (int): z level of section index 1. Provide either this or `sec_z0`.
            sec_z0 (int): section index at z=0. Provide either this or `z_first_sec`.
        """

        if downsample is not None:
            resolution = 'down%d' % downsample

        voxel_size_um = convert_resolution_string_to_voxel_size(resolution=resolution, stack=stack)
        section_thickness_in_voxel = SECTION_THICKNESS / voxel_size_um

        if z_first_sec is not None:
            sec_float = np.float32((z - z_first_sec) / section_thickness_in_voxel) # if use np.float, will result in np.floor(98.0)=97
        elif sec_z0 is not None:
            sec_float = np.float32(z / section_thickness_in_voxel) + sec_z0
        else:
            sec_float = np.float32(z / section_thickness_in_voxel)

        # print "sec_float =", sec_float
        sec = int(np.ceil(sec_float))
        return sec

    @staticmethod
    def get_initial_snake_contours_filepath(stack):
        return os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_prep1_thumbnail_initSnakeContours.pkl')

    @staticmethod
    def get_anchor_initial_snake_contours_filepath(stack):
        return os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_prep1_thumbnail_anchorInitSnakeContours.pkl')

    @staticmethod
    def get_auto_submask_rootdir_filepath(stack):
        return os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_prep1_thumbnail_autoSubmasks')

    @staticmethod
    def get_auto_submask_dir_filepath(stack, fn=None, sec=None):
        submasks_dir = DataManager.get_auto_submask_rootdir_filepath(stack)
        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][sec]
        dir_path = os.path.join(submasks_dir, fn)
        return dir_path

    @staticmethod
    def get_auto_submask_filepath(stack, what, submask_ind=None, fn=None, sec=None):
        """
        Args:
            what (str): submask or decisions.
            submask_ind (int): if what is submask, must provide submask_ind.
        """
        dir_path = DataManager.get_auto_submask_dir_filepath(stack=stack, fn=fn, sec=sec)

        if what == 'submask':
            assert submask_ind is not None, "Must provide submask_ind."
            fp = os.path.join(dir_path, fn + '_prep1_thumbnail_autoSubmask_%d.png' % submask_ind)
        elif what == 'decisions':
            fp = os.path.join(dir_path, fn + '_prep1_thumbnail_autoSubmaskDecisions.csv')
        else:
            raise Exception("Input %s is not recognized." % what)

        return fp

    @staticmethod
    def get_user_modified_submask_rootdir_filepath(stack):
        dir_path = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_prep1_thumbnail_userModifiedSubmasks')
        return dir_path

    @staticmethod
    def get_user_modified_submask_dir_filepath(stack, fn=None, sec=None):
        submasks_dir = DataManager.get_user_modified_submask_rootdir_filepath(stack)
        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][sec]
        dir_path = os.path.join(submasks_dir, fn)
        return dir_path

    @staticmethod
    def get_user_modified_submask_filepath(stack, what, submask_ind=None, fn=None, sec=None):
        """
        Args:
            what (str): submask or decisions.
            submask_ind (int): if what is submask, must provide submask_ind.
        """
        dir_path = DataManager.get_user_modified_submask_dir_filepath(stack=stack, fn=fn, sec=sec)

        if what == 'submask':
            assert submask_ind is not None, "Must provide submask_ind."
            fp = os.path.join(dir_path, fn + '_prep1_thumbnail_userModifiedSubmask_%d.png' % submask_ind)
        elif what == 'decisions':
            fp = os.path.join(dir_path, fn + '_prep1_thumbnail_userModifiedSubmaskDecisions.csv')
        elif what == 'parameters':
            fp = os.path.join(dir_path, fn + '_prep1_thumbnail_userModifiedParameters.json')
        elif what == 'contour_vertices':
            fp = os.path.join(dir_path, fn + '_prep1_thumbnail_userModifiedSubmaskContourVertices.pkl')
        else:
            raise Exception("Input %s is not recognized." % what)

        return fp


    @staticmethod
    def get_thumbnail_mask_dir_v3(stack, prep_id):
        """
        Get directory path of thumbnail mask.
        """
        return os.path.join(THUMBNAIL_DATA_DIR, stack, stack + ('_prep%d_' % prep_id if prep_id is not None else '_') + 'thumbnail_mask')

    @staticmethod
    def get_thumbnail_mask_filename_v3(stack, prep_id, section=None, fn=None):
        """
        Get filepath of thumbnail mask.
        """

        if isinstance(prep_id, str):
            prep_id = prep_str_to_id_2d[prep_id]

        dir_path = DataManager.get_thumbnail_mask_dir_v3(stack, prep_id=prep_id)
        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][section]
        fp = os.path.join(dir_path, fn + ('_prep%d_' % prep_id if prep_id is not None else '_') + 'thumbnail_mask.png')
        return fp

    # @staticmethod
    # def get_thumbnail_mask_filename_v3(stack, section=None, fn=None, version='aligned_cropped'):
    #     """
    #     Get filepath of thumbnail mask.
    #
    #     Args:
    #         version (str): One of aligned, aligned_cropped, cropped.
    #     """
    #
    #     anchor_fn = metadata_cache['anchor_fn'][stack]
    #     dir_path = DataManager.get_thumbnail_mask_dir_v3(stack, version=version)
    #     if fn is None:
    #         fn = metadata_cache['sections_to_filenames'][stack][section]
    #
    #     if version == 'aligned':
    #         fp = os.path.join(dir_path, fn + '_alignedTo_' + anchor_fn + '_mask.png')
    #     elif version == 'aligned_cropped' or version == 'cropped':
    #         fp = os.path.join(dir_path, fn + '_alignedTo_' + anchor_fn + '_mask_cropped.png')
    #     else:
    #         raise Exception('version %s is not recognized.' % version)
    #     return fp

    @staticmethod
    def load_thumbnail_mask_v3(stack, prep_id, section=None, fn=None):
        """
        Args:
            prep_id (str or int)
        """

        try:
            fp = DataManager.get_thumbnail_mask_filename_v3(stack=stack, section=section, fn=fn, prep_id=prep_id)
            # download_from_s3(fp, local_root=THUMBNAIL_DATA_ROOTDIR)
            mask = imread(fp).astype(np.bool)
            return mask
        except:
            sys.stderr.write('Cannot load mask %s, section=%s, fn=%s, prep=%s\n' % (stack, section, fn, prep_id))

            if isinstance(prep_id, str):
                prep_id = prep_str_to_id_2d[prep_id]

            if prep_id == 2:
                # get prep 2 masks directly from prep 5 masks.
                try:
                    sys.stderr.write('Try finding prep5 masks.\n')
                    fp = DataManager.get_thumbnail_mask_filename_v3(stack=stack, section=section, fn=fn, prep_id=5)
                    # download_from_s3(fp, local_root=THUMBNAIL_DATA_ROOTDIR)
                    mask_prep5 = imread(fp).astype(np.bool)

                    xmin,xmax,ymin,ymax = DataManager.load_cropbox_v2_relative(stack=stack, prep_id=prep_id, wrt_prep_id=5, out_resolution='down32')
                    mask_prep2 = mask_prep5[ymin:ymax+1, xmin:xmax+1].copy()
                    return mask_prep2
                except:
                    # get prep 2 masks directly from prep 1 masks.
                    sys.stderr.write('Cannot load mask %s, section=%s, fn=%s, prep=%s\n' % (stack, section, fn, prep_id))
                    sys.stderr.write('Try finding prep1 masks.\n')
                    fp = DataManager.get_thumbnail_mask_filename_v3(stack=stack, section=section, fn=fn, prep_id=1)
                    # download_from_s3(fp, local_root=THUMBNAIL_DATA_ROOTDIR)
                    mask_prep1 = imread(fp).astype(np.bool)

                    xmin,xmax,ymin,ymax = DataManager.load_cropbox_v2(stack=stack, prep_id=prep_id, return_dict=False, only_2d=True)
                    mask_prep2 = mask_prep1[ymin:ymax+1, xmin:xmax+1].copy()
                    return mask_prep2


    # @staticmethod
    # def load_thumbnail_mask_v3(stack, prep_id, section=None, fn=None):
    #     if stack in ['MD589', 'MD585', 'MD594']:
    #         fp = DataManager.get_thumbnail_mask_filename_v2(stack=stack, section=section, fn=fn, prep_id=prep_id)
    #     else:
    #         fp = DataManager.get_thumbnail_mask_filename_v3(stack=stack, section=section, fn=fn, prep_id=prep_id)
    #     download_from_s3(fp)
    #     mask = imread(fp).astype(np.bool)
    #     return mask

    # @staticmethod
    # def load_thumbnail_mask_v3(stack, section=None, fn=None, version='aligned_cropped'):
    #     if stack in ['MD589', 'MD585', 'MD594']:
    #         fp = DataManager.get_thumbnail_mask_filename_v2(stack=stack, section=section, fn=fn, version=version)
    #     else:
    #         fp = DataManager.get_thumbnail_mask_filename_v3(stack=stack, section=section, fn=fn, version=version)
    #     download_from_s3(fp)
    #     mask = imread(fp).astype(np.bool)
    #     return mask

    # @staticmethod
    # def load_thumbnail_mask_v2(stack, section=None, fn=None, version='aligned_cropped'):
    #     fp = DataManager.get_thumbnail_mask_filename_v2(stack=stack, section=section, fn=fn, version=version)
    #     download_from_s3(fp, local_root=DATA_ROOTDIR)
    #     mask = DataManager.load_data(fp, filetype='image').astype(np.bool)
    #     return mask
    #
    # @staticmethod
    # def get_thumbnail_mask_dir_v2(stack, version='aligned_cropped'):
    #     anchor_fn = metadata_cache['anchor_fn'][stack]
    #     if version == 'aligned_cropped':
    #         mask_dir = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_masks_alignedTo_' + anchor_fn + '_cropped')
    #     elif version == 'aligned':
    #         mask_dir = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_masks_alignedTo_' + anchor_fn)
    #     else:
    #         raise Exception("version %s not recognized." % version)
    #     return mask_dir
    #
    # @staticmethod
    # def get_thumbnail_mask_filename_v2(stack, section=None, fn=None, version='aligned_cropped'):
    #     anchor_fn = metadata_cache['anchor_fn'][stack]
    #     sections_to_filenames = metadata_cache['sections_to_filenames'][stack]
    #     if fn is None:
    #         fn = sections_to_filenames[section]
    #     mask_dir = DataManager.get_thumbnail_mask_dir_v2(stack=stack, version=version)
    #     if version == 'aligned_cropped':
    #         fp = os.path.join(mask_dir, fn + '_mask_alignedTo_' + anchor_fn + '_cropped.png')
    #     elif version == 'aligned':
    #         fp = os.path.join(mask_dir, fn + '_mask_alignedTo_' + anchor_fn + '.png')
    #     else:
    #         raise Exception("version %s not recognized." % version)
    #     return fp

    ###################################

    @staticmethod
    def get_region_labels_filepath(stack, sec=None, fn=None):
        """
        Returns:
            dict {label: list of region indices}
        """
        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][sec]
        return os.path.join(CELL_FEATURES_CLF_ROOTDIR, 'region_indices_by_label', stack, fn + '_region_indices_by_label.hdf')

    @staticmethod
    def get_ntb_to_nissl_intensity_profile_mapping_filepath(stack=None, ntb_fn=None):
        """
        Args:
            stack (str): If None, read the a priori mapping.
        """
        if stack is None:
            fp = os.path.join(DATA_DIR, 'average_nissl_intensity_mapping.npy')
        elif stack == 'ChatCryoJane201710':
            fp = os.path.join(DATA_DIR, 'kleinfeld_neurotrace_to_nissl_intensity_mapping.npy')
        else:
            fp = os.path.join(DATA_DIR, stack, stack + '_intensity_mapping', '%s_intensity_mapping.npy' % (ntb_fn))

        return fp

    @staticmethod
    def get_dataset_dir(dataset_id):
        return os.path.join(CLF_ROOTDIR, 'datasets', 'dataset_%d' % dataset_id)

    @staticmethod
    def get_dataset_patches_filepath(dataset_id, structure=None):
        if structure is None:
            patch_images_fp = os.path.join(CLF_ROOTDIR, 'datasets', 'dataset_%d' % dataset_id, 'patch_images.hdf')
        else:
            patch_images_fp = os.path.join(CLF_ROOTDIR, 'datasets', 'dataset_%d' % dataset_id, 'patch_images_%s.hdf' % structure)
        return patch_images_fp

    @staticmethod
    def load_dataset_patches(dataset_id, structure=None):
        """
        FIXME: file extension is hdf but the format is actually bp.

        Returns:
            (n,224,224)-array: patches
        """
        fp = DataManager.get_dataset_patches_filepath(dataset_id=dataset_id, structure=structure)
        # download_from_s3(fp, local_root=os.path.dirname(CLF_ROOTDIR))
        return bp.unpack_ndarray_file(fp)

    @staticmethod
    def get_dataset_features_filepath(dataset_id, structure=None, ext='bp'):
        if structure is None:
            features_fp = os.path.join(CLF_ROOTDIR, 'datasets', 'dataset_%d' % dataset_id, 'patch_features.' + ext)
        else:
            features_fp = os.path.join(CLF_ROOTDIR, 'datasets', 'dataset_%d' % dataset_id, 'patch_features_%s.' % structure + ext)
        return features_fp

    @staticmethod
    def load_dataset_features(dataset_id, structure=None):
        fp = DataManager.get_dataset_features_filepath(dataset_id=dataset_id, structure=structure)
        # download_from_s3(fp, local_root=os.path.dirname(CLF_ROOTDIR))
        return bp.unpack_ndarray_file(fp)

    @staticmethod
    def get_dataset_addresses_filepath(dataset_id, structure=None):
        if structure is None:
            addresses_fp = os.path.join(CLF_ROOTDIR, 'datasets', 'dataset_%d' % dataset_id, 'patch_addresses.pkl')
        else:
            addresses_fp = os.path.join(CLF_ROOTDIR, 'datasets', 'dataset_%d' % dataset_id, 'patch_addresses_%s.pkl' % structure)
        return addresses_fp

    @staticmethod
    def load_dataset_addresses(dataset_id, structure=None):
        fp = DataManager.get_dataset_addresses_filepath(dataset_id=dataset_id, structure=structure)
        # download_from_s3(fp, local_root=os.path.dirname(CLF_ROOTDIR))
        return load_pickle(fp)

    @staticmethod
    def get_classifier_filepath(classifier_id, structure):
        classifier_id_dir = os.path.join(CLF_ROOTDIR, 'setting_%d' % classifier_id)
        classifier_dir = os.path.join(classifier_id_dir, 'classifiers')
        return os.path.join(classifier_dir, '%(structure)s_clf_setting_%(setting)d.dump' % \
                     dict(structure=structure, setting=classifier_id))

    ####### Fluorescent ########

    @staticmethod
    def get_labeled_neurons_filepath(stack, sec=None, fn=None):
        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][sec]
        return os.path.join(LABELED_NEURONS_ROOTDIR, stack, fn, fn + ".pkl")


    @staticmethod
    def load_labeled_neurons_filepath(stack, sec=None, fn=None):
        fp = DataManager.get_labeled_neurons_filepath(**locals())
        # download_from_s3(fp)
        return load_pickle(fp)

    @staticmethod
    def load_datasets_bp(dataset_ids, labels_to_sample=None, clf_rootdir=CLF_ROOTDIR):
        """
        Load multiple datasets, returns both features and addresses.
        Assume the features are stored as patch_features_<name>.bp; addresses are stored as patch_addresses_<name>.bp.

        Args:
            labels_to_sample (list of str): e.g. VCA_surround_500_VCP. If this is not given, use all labels in the associated dataset directory.

        Returns:
            (merged_features, merged_addresses)
        """

        merged_features = {}
        merged_addresses = {}

        for dataset_id in dataset_ids:

            if labels_to_sample is None:
                labels_to_sample = []
                for dataset_id in dataset_ids:
                    dataset_dir = DataManager.get_dataset_dir(dataset_id=dataset_id)
                    #download_from_s3(dataset_dir, is_dir=True)
                    for fn in os.listdir(dataset_dir):
                        g = re.match('patch_features_(.*).bp', fn).groups()
                        if len(g) > 0:
                            labels_to_sample.append(g[0])

            for label in labels_to_sample:
                try:
                    # Load training features

                    # features_fp = DataManager.get_dataset_features_filepath(dataset_id=dataset_id, structure=label)
                    #download_from_s3(features_fp)
                    # features = bp.unpack_ndarray_file(features_fp)
                    features = DataManager.load_dataset_features(dataset_id=dataset_id, structure=label)

                    # load training addresses

                    # addresses_fp = DataManager.get_dataset_addresses_filepath(dataset_id=dataset_id, structure=label)
                    # #download_from_s3(addresses_fp)
                    # addresses = load_pickle(addresses_fp)
                    addresses = DataManager.load_dataset_addresses(dataset_id=dataset_id, structure=label)

                    if label not in merged_features:
                        merged_features[label] = features
                    else:
                        merged_features[label] = np.concatenate([merged_features[label], features])

                    if label not in merged_addresses:
                        merged_addresses[label] = addresses
                    else:
                        merged_addresses[label] += addresses

                except Exception as e:
                    continue

        return merged_features, merged_addresses

    ############################################

    @staticmethod
    def get_brightfield_or_fluorescence(stack, sec=None, fn=None):
        if stack in all_nissl_stacks:
            return 'N'
        else:
            if fn is None:
                fn = metadata_cache['sections_to_filenames'][stack][sec]

            if fn.split('-')[1].startswith('N'):
                return 'N'
            else:
                return 'F'

##################################################

def download_all_metadata():

    for stack in all_stacks:
        try:
            download_from_s3(DataManager.get_sorted_filenames_filename(stack=stack))
        except:
            pass
        try:
            download_from_s3(DataManager.get_anchor_filename_filename(stack=stack))
        except:
            pass
        try:
            download_from_s3(DataManager.get_cropbox_filename(stack=stack))
        except:
            pass

# Temporarily commented out
# download_all_metadata()

# This module stores any meta information that is dynamic.
metadata_cache = {}

def generate_metadata_cache():

    global metadata_cache
    metadata_cache['image_shape'] = {}
    metadata_cache['anchor_fn'] = {}
    metadata_cache['sections_to_filenames'] = {}
    metadata_cache['filenames_to_sections'] = {}
    metadata_cache['section_limits'] = {}
    metadata_cache['cropbox'] = {}
    metadata_cache['valid_sections'] = {}
    metadata_cache['valid_filenames'] = {}
    metadata_cache['valid_sections_all'] = {}
    metadata_cache['valid_filenames_all'] = {}
    for stack in all_stacks:

        try:
            metadata_cache['anchor_fn'][stack] = DataManager.load_anchor_filename(stack)
        except:
            pass
        try:
            metadata_cache['sections_to_filenames'][stack] = DataManager.load_sorted_filenames(stack)[1]
        except:
            pass
        try:
            metadata_cache['filenames_to_sections'][stack] = DataManager.load_sorted_filenames(stack)[0]
            metadata_cache['filenames_to_sections'][stack].pop('Placeholder')
            metadata_cache['filenames_to_sections'][stack].pop('Nonexisting')
            metadata_cache['filenames_to_sections'][stack].pop('Rescan')
        except:
            pass
        try:
            metadata_cache['section_limits'][stack] = DataManager.load_section_limits_v2(stack, prep_id=2)
        except:
            pass
        try:
            # alignedBrainstemCrop cropping box
            metadata_cache['cropbox'][stack] = DataManager.load_cropbox_v2(stack, prep_id=2)
        except:
            pass

        try:
            first_sec, last_sec = metadata_cache['section_limits'][stack]
            metadata_cache['valid_sections'][stack] = [sec for sec in range(first_sec, last_sec+1) if not is_invalid(stack=stack, sec=sec)]
            metadata_cache['valid_filenames'][stack] = [metadata_cache['sections_to_filenames'][stack][sec] for sec in
                                                       metadata_cache['valid_sections'][stack]]
        except:
            pass

        try:
            metadata_cache['valid_sections_all'][stack] = [sec for sec, fn in metadata_cache['sections_to_filenames'][stack].iteritems() if not is_invalid(fn=fn)]
            metadata_cache['valid_filenames_all'][stack] = [fn for sec, fn in metadata_cache['sections_to_filenames'][stack].iteritems() if not is_invalid(fn=fn)]
        except:
            pass

        try:
            metadata_cache['image_shape'][stack] = DataManager.get_image_dimension(stack)
        except:
            pass
        
    return metadata_cache


generate_metadata_cache()

def resolve_actual_setting(setting, stack, fn=None, sec=None):
    """Take a possibly composite setting index, and return the actual setting index according to fn."""

    if stack in all_nissl_stacks:
        stain = 'nissl'
    elif stack in all_ntb_stacks:
        stain = 'ntb'
    elif stack in all_alt_nissl_ntb_stacks:
        if fn is None:
            assert sec is not None
            fn = metadata_cache['sections_to_filenames'][stack][sec]
        if fn.split('-')[1][0] == 'F':
            stain = 'ntb'
        elif fn.split('-')[1][0] == 'N':
            stain = 'nissl'
        else:
            raise Exception('Must be either ntb or nissl.')

    if setting == 12:
        setting_nissl = 2
        setting_ntb = 10

    if setting == 12:
        if stain == 'nissl':
            setting_ = setting_nissl
        else:
            setting_ = setting_ntb
    else:
        setting_ = setting

    return setting_
