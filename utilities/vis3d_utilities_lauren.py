"""This module contains utilities for 3D visualization in VTK."""

import numpy as np
import sys
import time

try:
    import vtk
    from vtk.util import numpy_support
except:
    sys.stderr.write('No vtk\n')

try:
    import mcubes # https://github.com/pmneila/PyMCubes
except:
    sys.stderr.write('No mcubes\n')

from skimage.measure import marching_cubes, correct_mesh_orientation, mesh_surface_area

import os
sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from metadata import *
from data_manager import *

def volume_to_polydata(volume, num_simplify_iter=0, smooth=False, level=0., min_vertices=200, return_vertex_face_list=False):
    """Convert a volume to a mesh, either as vertices/faces tuple or a vtk.Polydata.

    Args:
        level (float): the level to threshold the input volume
        min_vertices (int): minimum number of vertices. Simplification will stop if the number of vertices drops below this value.
        return_vertex_face_list (bool): If True, return only (vertices, faces); otherwise, return polydata.
    """
    volume, origin = convert_volume_forms(volume=volume, out_form=("volume", "origin"))
    vol = volume > level
    vol_padded = np.pad(vol, ((5,5),(5,5),(5,5)), 'constant') # need this otherwise the sides of volume will not close and expose the hollow inside of structures

    t = time.time()
    vs, fs = mcubes.marching_cubes(vol_padded, 0) # more than 5 times faster than skimage.marching_cube + correct_orientation
    sys.stderr.write('marching cube: %.2f seconds\n' % (time.time() - t))

    vs = vs[:, [1,0,2]] + origin - (5,5,5)

    t = time.time()
    polydata = mesh_to_polydata(vs, fs)
    sys.stderr.write('mesh_to_polydata: %.2f seconds\n' % (time.time() - t)) #

    for simplify_iter in range(num_simplify_iter):

        t = time.time()

        deci = vtk.vtkQuadricDecimation()
        deci.SetInputData(polydata)

        deci.SetTargetReduction(0.8)
        # 0.8 means each iteration causes the point number to drop to 20% the original

        deci.Update()

        polydata = deci.GetOutput()

        if smooth:

            smoother = vtk.vtkWindowedSincPolyDataFilter()
            smoother.SetPassBand(.1)
            smoother.SetNumberOfIterations(20)
            smoother.SetInputData(polydata)
            smoother.Update()

            polydata = smoother.GetOutput()

        n_pts = polydata.GetNumberOfPoints()
        sys.stderr.write('simplify %d @ %d: %.2f seconds\n' % (simplify_iter, n_pts, time.time() - t)) #

        if polydata.GetNumberOfPoints() < min_vertices:
            break


    if return_vertex_face_list:
        return polydata_to_mesh(polydata)
    else:
        return polydata
