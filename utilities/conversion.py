from utilities2015 import *
from metadata import *
from itertools import groupby
sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from data_manager import *

def images_to_volume(images, voxel_size, first_sec=None, last_sec=None, return_bbox=True):
    """
    images_grouped_by_z: dict of images, key is section index. First brain section should have index 0.
    voxel_size: (xdim,ydim,zdim) in unit of pixel size.
    """

    if isinstance(images, dict):
        ydim, xdim = images.values()[0].shape[:2]
        sections = images.keys()
        if last_sec is None:
            last_sec = np.max(sections)
        if first_sec is None:
            first_sec = np.min(sections)

    elif callable(images):
        try:
            ydim, xdim = images(100).shape[:2]
        except:
            ydim, xdim = images(200).shape[:2]
        assert last_sec is not None
        assert first_sec is not None
    else:
        raise Exception('images must be dict or function.')

    voxel_z_size = voxel_size[2]

    z_end = int(np.ceil((last_sec+1)*voxel_z_size))
    z_begin = int(np.floor(first_sec*voxel_z_size))
    zdim = z_end + 1 - z_begin

    # print 'Volume shape:', xdim, ydim, zdim

    volume = np.zeros((ydim, xdim, zdim), images.values()[0].dtype)

    # bar = show_progress_bar(first_sec, last_sec)

    for i in range(len(images.keys())-1):
        # bar.value = sections[i]
        z1 = sections[i] * voxel_z_size
        z2 = sections[i+1] * voxel_z_size
        if isinstance(images, dict):
            im = images[sections[i]]
        elif callable(images):
            im = images(sections[i])
        volume[:, :, int(z1)-z_begin:int(z2)+1-z_begin] = im[..., None]

    volume_bbox = (0,xdim-1,0,ydim-1,z_begin,z_end)

    if return_bbox:
        return volume, volume_bbox
    else:
        return volume


def points2d_to_points3d(pts2d_grouped_by_section, pts2d_downsample, pts3d_downsample, stack, pts3d_origin=(0,0,0)):
    """
    Args:
        pts2d_downsample ((n,2)-ndarray): 2D point (x,y) coordinates on cropped images, in pts2d_downsample resolution.
        pts3d_origin (3-tuple): xmin, ymin, zmin in pts3d_downsample resolution.

    Returns:
        ((n,3)-ndarray)
    """

    pts3d = {}
    for sec, pts2d in pts2d_grouped_by_section.iteritems():
        z_down = np.mean(DataManager.convert_section_to_z(stack=stack, sec=sec, downsample=pts3d_downsample))
        n = len(pts2d)
        xys_down = np.array(pts2d) * pts2d_downsample / pts3d_downsample
        pts3d[sec] = np.c_[xys_down, z_down*np.ones((n,))] - pts3d_origin

    return pts3d


def contours_to_volume(contours_grouped_by_label=None, label_contours_tuples=None, interpolation_direction='z',
                      return_shell=False, len_interval=20):
    """
    Return volume as 3D array, and origin (xmin,xmax,ymin,ymax,zmin,zmax)
    
    Args:
        contours_grouped_by_label ({int: list of (3,n)-arrays}): 
    
    """

    import sys
    sys.path.append(os.environ['REPO_DIR'] + '/utilities')
    from annotation_utilities import interpolate_contours_to_volume


    if label_contours_tuples is not None:
        contours_grouped_by_label = {}
        for label, contours in groupby(contour_label_tuples, key=lambda l, cnts: l):
            contours_grouped_by_label[label] = contours
    else:
        assert contours_grouped_by_label is not None

    if isinstance(contours_grouped_by_label.values()[0], dict):
        # dict value is contours grouped by z
        if interpolation_direction == 'z':
            contours_xyz_grouped_by_label = {label: [(x,y,z) for z, (x,y) in contours_grouped.iteritems()]
                            for label, contours_grouped in contours_grouped_by_label.iteritems()}
        elif interpolation_direction == 'y':
            contours_xyz_grouped_by_label = {label: [(x,y,z) for y, (x,z) in contours_grouped.iteritems()]
                            for label, contours_grouped in contours_grouped_by_label.iteritems()}
        elif interpolation_direction == 'x':
            contours_xyz_grouped_by_label = {label: [(x,y,z) for x, (y,z) in contours_grouped.iteritems()]
                            for label, contours_grouped in contours_grouped_by_label.iteritems()}

    else:
        contours_xyz_grouped_by_label = contours_grouped_by_label
        # dict value is list of (x,y,z) tuples
#         contours_grouped_by_label = {groupby(contours_xyz, lambda x,y,z: z)
#                                      for label, contours_xyz in contours_grouped_by_label.iteritems()}
#         pass

    xyz_max = [0, 0, 0]
    xyz_min = [np.inf, np.inf, np.inf]
    for label, contours in contours_xyz_grouped_by_label.iteritems():
        xyz_max = np.maximum(xyz_max, np.max(np.vstack(contours), axis=0))
        xyz_min = np.minimum(xyz_min, np.min(np.vstack(contours), axis=0))

    xmin, ymin, zmin = np.floor(xyz_min).astype(np.int)
    xmax, ymax, zmax = np.ceil(xyz_max).astype(np.int)
    xdim, ydim, zdim = xmax+1-xmin, ymax+1-ymin, zmax+1-zmin


    volume = np.zeros((ydim, xdim, zdim), np.uint8)

    if return_shell:

        for label, contours in contours_grouped_by_label.iteritems():

            voxels_grouped = interpolate_contours_to_volume(interpolation_direction=interpolation_direction,
                                                            contours_xyz=contours, return_contours=True,
                                                            len_interval=len_interval)

            if interpolation_direction == 'z':
                for z, xys in voxels_grouped.iteritems():
                    volume[xys[:,1]-ymin, xys[:,0]-xmin, z-zmin] = label
            elif interpolation_direction == 'y':
                for y, xzs in voxels_grouped.iteritems():
                    volume[y-ymin, xzs[:,0]-xmin, xzs[:,1]-zmin] = label
            elif interpolation_direction == 'x':
                for x, yzs in voxels_grouped.iteritems():
                    volume[yzs[:,0]-ymin, x-xmin, yzs[:,1]-zmin] = label

        return volume, (xmin,xmax,ymin,ymax,zmin,zmax)

    else:

        for label, contours in contours_grouped_by_label.iteritems():

            voxels_grouped = interpolate_contours_to_volume(interpolation_direction=interpolation_direction,
                                                                 contours_xyz=contours, return_voxels=True)

            if interpolation_direction == 'z':
                for z, xys in voxels_grouped.iteritems():
                    volume[xys[:,1]-ymin, xys[:,0]-xmin, z-zmin] = label
            elif interpolation_direction == 'y':
                for y, xzs in voxels_grouped.iteritems():
                    volume[y-ymin, xzs[:,0]-xmin, xzs[:,1]-zmin] = label
            elif interpolation_direction == 'x':
                for x, yzs in voxels_grouped.iteritems():
                    volume[yzs[:,0]-ymin, x-xmin, yzs[:,1]-zmin] = label

        return volume, (xmin,xmax,ymin,ymax,zmin,zmax)



def volume_to_images(volume, voxel_size, cut_dimension, pixel_size=None):

    volume_shape = volume.shape

    if pixel_size is None:
        pixel_size = min(voxel_size)

    if cut_dimension == 0:
        volume_shape01 = volume_shape[1], volume_shape[2]
        voxel_size01 = voxel_size[1], voxel_size[2]
    elif cut_dimension == 1:
        volume_shape01 = volume_shape[0], volume_shape[2]
        voxel_size01 = voxel_size[0], voxel_size[2]
    elif cut_dimension == 2:
        volume_shape01 = volume_shape[0], volume_shape[1]
        voxel_size01 = voxel_size[0], voxel_size[1]

    volume_dim01 = volume_shape01[0] * voxel_size01[0], volume_shape01[1] * voxel_size01[1]
    sample_voxels_0 = np.arange(0, volume_dim01[0], pixel_size) / voxel_size01[0]
    sample_voxels_1 = np.arange(0, volume_dim01[1], pixel_size) / voxel_size01[1]

    if cut_dimension == 0:
        images = volume[:, sample_voxels_0[:,None], sample_voxels_1]
    elif cut_dimension == 1:
        images = volume[sample_voxels_0[:,None], :, sample_voxels_1]
    elif cut_dimension == 2:
        images = volume[sample_voxels_0[:,None], sample_voxels_1, :]

    return images
