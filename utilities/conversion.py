from utilities2015 import *
from metadata import *

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

def
