import vtk
from vtk.util import numpy_support

import numpy as np
import sys

from skimage.measure import marching_cubes, correct_mesh_orientation, mesh_surface_area

from itertools import izip

import time
import mcubes # https://github.com/pmneila/PyMCubes

def polydata_to_mesh(polydata):
    
    vertices = np.array([polydata.GetPoint(i) for i in range(polydata.GetNumberOfPoints())])
    
    face_data_arr = numpy_support.vtk_to_numpy(polydata.GetPolys().GetData())
    
    faces = np.c_[face_data_arr[1::4],
                  face_data_arr[2::4],
                  face_data_arr[3::4]]
    
    return vertices, faces
    
def mesh_to_polydata(vertices, faces):
    
    polydata = vtk.vtkPolyData()

    t = time.time()
    
    points = vtk.vtkPoints()
    
    # points_vtkArray = numpy_support.numpy_to_vtk(vertices.flat)
    # points.SetData(points_vtkArray)
    
    for pt_ind, (x,y,z) in enumerate(vertices):
        points.InsertPoint(pt_ind, x, y, z)
        
    sys.stderr.write('fill point array: %.2f seconds\n' % (time.time() - t))
        
    t = time.time()
    
    cells = vtk.vtkCellArray()
    # for ia, ib, ic in faces:
    #     cells.InsertNextCell(3)
    #     cells.InsertCellPoint(ia)
    #     cells.InsertCellPoint(ib)
    #     cells.InsertCellPoint(ic)
    
    cell_arr = np.empty((len(faces)*4, ), np.int)
    cell_arr[::4] = 3
    cell_arr[1::4] = faces[:,0]
    cell_arr[2::4] = faces[:,1]
    cell_arr[3::4] = faces[:,2]
    cell_vtkArray = numpy_support.numpy_to_vtkIdTypeArray(cell_arr, deep=1)
    cells.SetCells(len(faces), cell_vtkArray)
    
    sys.stderr.write('fill cell array: %.2f seconds\n' % (time.time() - t))

    polydata.SetPoints(points)
    polydata.SetPolys(cells)
    # polydata.SetVerts(cells)

    return polydata
    
def volume_to_polydata(volume, origin, num_simplify_iter=0, smooth=False):
    
    vol = volume.astype(np.bool)
    vol_padded = np.pad(vol, ((5,5),(5,5),(5,5)), 'constant')

    t = time.time()  
    vs, fs = mcubes.marching_cubes(vol_padded, 0) # more than 5 times faster than skimage.marching_cube + correct_orientation
    sys.stderr.write('marching cube: %.2f seconds\n' % (time.time() - t))

    # t = time.time()
    # vs, faces = marching_cubes(vol_padded, 0) # y,x,z
    # sys.stderr.write('marching cube: %.2f seconds\n' % (time.time() - t))
    
    # t = time.time()
    # fs = correct_mesh_orientation(vol_padded, vs, faces)
    # sys.stderr.write('correct orientation: %.2f seconds\n' % (time.time() - t))
    
    vs = vs[:, [1,0,2]] + origin - (5,5,5)
    
    t = time.time()
    area = mesh_surface_area(vs, fs)
    
    print 'area: %.2f' % area

    sys.stderr.write('compute surface area: %.2f seconds\n' % (time.time() - t)) #

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
    #         smoother.NormalizeCoordinatesOn()
            smoother.SetPassBand(.1)
            smoother.SetNumberOfIterations(20)
            smoother.SetInputData(polydata)
            smoother.Update()
            
            polydata = smoother.GetOutput()
        
        n_pts = polydata.GetNumberOfPoints()
            
        if polydata.GetNumberOfPoints() < 200:
            break
            
        sys.stderr.write('simplify %d @ %d: %.2f seconds\n' % (simplify_iter, n_pts, time.time() - t)) #

        
    return polydata    

    
def polydata_to_volume(polydata):
    """
    Parameters
    ----------
    polydata : vtkPolyData
        input polydata
    
    Returns
    -------
    (numpy arr, 3-tuple, vtkImageData)
        (volume, origin, imagedata)
    
    """
        
    bounds = polydata.GetBounds()
    spacing = [1., 1., 1.]
    
    origin = [bounds[0] + spacing[0]/2, 
              bounds[2] + spacing[1]/2, 
              bounds[4] + spacing[2]/2]

    whiteImage = vtk.vtkImageData()
    whiteImage.SetSpacing(spacing)
    whiteImage.SetOrigin(origin)

    dim = np.array([np.ceil(bounds[1]-bounds[0])/spacing[0], 
                    np.ceil(bounds[3]-bounds[2])/spacing[1], 
                    np.ceil(bounds[5]-bounds[4])/spacing[2]], 
                    np.int)

    whiteImage.SetDimensions(dim)
    whiteImage.SetExtent(0, dim[0]-1, 0, dim[1]-1, 0, dim[2]-1)

    # whiteImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    n_pts = whiteImage.GetNumberOfPoints()
   
    # t = time.time() 
#    inval = 255
#    outval = 0
#    for i in range(n_pts):
#        whiteImage.GetPointData().GetScalars().SetTuple1(i, inval)
    whiteImage.GetPointData().SetScalars(numpy_support.numpy_to_vtk(255*np.ones((n_pts, ), np.uint8), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)) # deep copy must be true
    # sys.stderr.write('time 1: %.2f\n' % (time.time() - t) )


    # t = time.time()
    
    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetInputData(polydata)
    pol2stenc.SetOutputOrigin(origin)
    pol2stenc.SetOutputSpacing(spacing)
    pol2stenc.SetOutputWholeExtent(whiteImage.GetExtent())
    pol2stenc.Update()
    
    # sys.stderr.write('time 2: %.2f\n' % (time.time() - t) )

    # t = time.time()

    # cut the corresponding white image and set the background:
    imgstenc = vtk.vtkImageStencil()
    imgstenc.SetInputData(whiteImage)
    imgstenc.SetStencilData(pol2stenc.GetOutput())
    imgstenc.ReverseStencilOff()
    imgstenc.SetBackgroundValue(0)
    imgstenc.Update()
    
    # sys.stderr.write('time 3: %.2f\n' % (time.time() - t) )
    
    # t = time.time()
    
    im = imgstenc.GetOutput()
    x, y, z = im.GetDimensions()
    sc = im.GetPointData().GetScalars()
    a = numpy_support.vtk_to_numpy(sc)
    b = a.reshape(z,y,x)
    b = np.transpose(b, [1,2,0])

    # sys.stderr.write('time 4: %.2f\n' % (time.time() - t) )
    
    return b, origin, im


def volume_to_imagedata(arr):
    
    imagedata = vtk.vtkImageData()
    imagedata.SetDimensions([arr.shape[1], arr.shape[0], arr.shape[2]])
    imagedata.SetSpacing([1., 1., 1.])
    
    v3 = np.transpose(arr, [2,0,1])
    
    if arr.dtype == np.uint8:
        t = vtk.VTK_UNSIGNED_CHAR
    elif arr.dtype == np.float32:
        t = vtk.VTK_FLOAT
        
    imagedata.GetPointData().SetScalars(numpy_support.numpy_to_vtk(v3.flat, deep=True, array_type=t)) # deep copy must be true
    return imagedata

def add_axes(iren):
    axes = vtk.vtkAxesActor()

    widget = vtk.vtkOrientationMarkerWidget()
    widget.SetOutlineColor( 0.9300, 0.5700, 0.1300 );
    widget.SetOrientationMarker( axes );
    widget.SetInteractor( iren );
    widget.SetViewport( 0.0, 0.0, 0.2, 0.2 );
    widget.SetEnabled( 1 );
    widget.InteractiveOn();
    return widget

def load_mesh_stl(fn, return_polydata_only=False):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(fn)
    reader.Update()

    polydata = reader.GetOutput()
    assert polydata is not None
    
    if return_polydata_only:
        return polydata
       
    vertices = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())
    a = numpy_support.vtk_to_numpy(polydata.GetPolys().GetData())
    faces = np.c_[a[1::4], a[2::4], a[3::4]]
    
    return vertices, faces

def save_mesh_stl(polydata, fn):
    stlWriter = vtk.vtkSTLWriter()
    stlWriter.SetFileName(fn)
    stlWriter.SetInputData(polydata)
    stlWriter.Write()   