import numpy as np
import sys
import time

try:
    import vtk
    from vtk.util import numpy_support
    import mcubes # https://github.com/pmneila/PyMCubes
except:
    sys.stderr.write('No vtk\n')

from skimage.measure import marching_cubes, correct_mesh_orientation, mesh_surface_area

import os
sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from metadata import *
from data_manager import *

#######################################################################


def download_volume(stack, what, dest_dir, name_u=None):

    create_if_not_exists(dest_dir)

    if what == 'atlasProjected':
        os.system('scp gcn:/oasis/projects/nsf/csd395/yuncong/CSHL_volumes/%(stack)s/%(stack)s_atlasProjectedVolume.bp %(volume_d)s/' % \
                  {'stack': stack, 'volume_d':dest_dir})

    elif what == 'localAdjusted':
        os.system('scp gcn:/oasis/projects/nsf/csd395/yuncong/CSHL_volumes/%(stack)s/%(stack)s_localAdjustedVolume.bp %(volume_d)s/' % \
                  {'stack': stack, 'volume_d':dest_dir})

    elif what == 'score':
        assert name_u is not None, 'Class name is not provided'
        os.system('scp gcn:/oasis/projects/nsf/csd395/yuncong/CSHL_volumes/%(stack)s/%(stack)s_scoreVolume_%(name_u)s.bp %(volume_d)s/' % \
                      {'stack': stack, 'volume_d':dest_dir, 'name_u': name_u})

        print 'scp gcn:/oasis/projects/nsf/csd395/yuncong/CSHL_volumes/%(stack)s/%(stack)s_scoreVolume_%(name_u)s.bp %(volume_d)s/' % \
                      {'stack': stack, 'volume_d':dest_dir, 'name_u': name_u}


################ Conversion between volume representations #################

def rescale_polydata(polydata, factor):
    v, f = polydata_to_mesh(polydata)
    return mesh_to_polydata(v * factor, f)

def transform_polydata(polydata, transform):
    v, f = polydata_to_mesh(polydata)
    from registration_utilities import transform_points
    new_v = transform_points(transform=transform, pts=v)
    return mesh_to_polydata(new_v, f)

def move_polydata(polydata, d):
    # !!! IMPORTANT!! Note that this operation discards all scalar data (for example heatmap) in the input polydata.
    vs, fs = polydata_to_mesh(polydata)
    return mesh_to_polydata(vs + d, fs)

def poisson_reconstruct_meshlab(polydata=None, input_fn=None, output_fn=None):

    tmp_output_fn = '/tmp/output.stl'
    tmp_input_fn = '/tmp/input.ply'

    if output_fn is None:
        output_fn = tmp_output_fn

    if input_fn is None:
        assert polydata is not None
        input_fn = tmp_input_fn
        save_mesh(polydata, input_fn)

    execute_command('meshlabserver -i %(input_fn)s -o %(output_fn)s -s %(script_fn)s -om vc vn' % \
               dict(input_fn=input_fn, output_fn=output_fn,
               script_fn=os.path.join(REPO_DIR, '3d/outerContour_poisson_reconstruct.mlx')))

    # if input_fn == tmp_input_fn:
    #     execute_command('rm %s' % tmp_input_fn)

    if output_fn == tmp_output_fn:
        output_polydata = load_mesh_stl(output_fn, return_polydata_only=True)
        # execute_command('rm %s' % tmp_output_fn)
        return output_polydata


def polydata_to_mesh(polydata):
    """
    Extract vertice and face data from a polydata object.

    Returns:
        (vertices, faces)
    """

    vertices = np.array([polydata.GetPoint(i) for i in range(polydata.GetNumberOfPoints())])

    try:
        face_data_arr = numpy_support.vtk_to_numpy(polydata.GetPolys().GetData())

        faces = np.c_[face_data_arr[1::4],
                      face_data_arr[2::4],
                      face_data_arr[3::4]]
    except:
        sys.stderr.write('polydata_to_mesh: No faces are loaded.\n')
        faces = []

    return vertices, faces


def vertices_to_surface(vertices, num_simplify_iter=3, smooth=True, neighborhood_size=None, sample_spacing=None):
    """
    Based on vertices, reconstruct the surface as polydata. Uses vtkSurfaceReconstructionFilter.
    """

    polydata = mesh_to_polydata(vertices, [])

    surf = vtk.vtkSurfaceReconstructionFilter()
    if neighborhood_size is not None:
        surf.SetNeighborhoodSize(neighborhood_size)
        # neighborhood_size = 30
    if sample_spacing is not None:
        surf.SetSampleSpacing(sample_spacing)
        # sample_spacing=5
    surf.SetInputData(polydata)
    surf.Update()

    # Visualize signed distance function computed by VTK (VTK bug: error outside actual contour)
    # q = surf.GetOutput()
    # arr = numpy_support.vtk_to_numpy(q.GetPointData().GetScalars())
    # sc = arr.reshape(q.GetDimensions()[::-1])
    # plt.imshow(sc[40,:,:]);
    # plt.colorbar();

    cf = vtk.vtkContourFilter()
    cf.SetInputConnection(surf.GetOutputPort());
    cf.SetValue(0, 0.);
    # print cf.GetNumberOfContours()
    cf.Update()
    # polydata = cf.GetOutput()

    reverse = vtk.vtkReverseSense()
    reverse.SetInputConnection(cf.GetOutputPort())
    reverse.ReverseCellsOn()
    reverse.ReverseNormalsOn()
    reverse.Update()

    polydata = reverse.GetOutput()

    polydata = simplify_polydata(polydata, num_simplify_iter=num_simplify_iter, smooth=smooth)

    return polydata


def simplify_polydata(polydata, num_simplify_iter=0, smooth=False):
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

def mesh_to_polydata(vertices, faces, num_simplify_iter=0, smooth=False):
    """
    Args:
        vertices ((num_vertices, 3) arrays)
        faces ((num_faces, 3) arrays)
    """

    polydata = vtk.vtkPolyData()

    t = time.time()

    points = vtk.vtkPoints()

    # points_vtkArray = numpy_support.numpy_to_vtk(vertices.flat)
    # points.SetData(points_vtkArray)

    for pt_ind, (x,y,z) in enumerate(vertices):
        points.InsertPoint(pt_ind, x, y, z)

    # sys.stderr.write('fill point array: %.2f seconds\n' % (time.time() - t))

    t = time.time()

    if len(faces) > 0:

        cells = vtk.vtkCellArray()

        cell_arr = np.empty((len(faces)*4, ), np.int)
        cell_arr[::4] = 3
        cell_arr[1::4] = faces[:,0]
        cell_arr[2::4] = faces[:,1]
        cell_arr[3::4] = faces[:,2]
        cell_vtkArray = numpy_support.numpy_to_vtkIdTypeArray(cell_arr, deep=1)
        cells.SetCells(len(faces), cell_vtkArray)

    # sys.stderr.write('fill cell array: %.2f seconds\n' % (time.time() - t))

    polydata.SetPoints(points)

    if len(faces) > 0:
        polydata.SetPolys(cells)
        # polydata.SetVerts(cells)

    if len(faces) > 0:
        polydata = simplify_polydata(polydata, num_simplify_iter, smooth)
    else:
        sys.stderr.write('mesh_to_polydata: No faces are provided, so skip simplification.\n')

    return polydata

def volume_to_polydata(volume, num_simplify_iter=0, smooth=False, level=0., min_vertices=200, return_vertex_face_list=False):
    """
    Convert a volume to a mesh, either as vertices/faces tuple or a vtk.Polydata.

    Args:
        level (float): the level to threshold the input volume
        min_vertices (int): minimum number of vertices. Simplification will stop if the number of vertices drops below this value.
        return_vertex_face_list (bool): If True, return only (vertices, faces); otherwise, return polydata.
    """

    volume, origin = convert_volume_forms(volume=volume, out_form=("volume", "origin"))

    vol = volume > level

    # vol_padded = np.zeros(vol.shape+(10,10,10), np.bool)
    # vol_padded[5:-5, 5:-5, 5:-5] = vol
    vol_padded = np.pad(vol, ((5,5),(5,5),(5,5)), 'constant') # need this otherwise the sides of volume will not close and expose the hollow inside of structures

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
    # vs = vs[:, [1,0,2]] + origin

    # t = time.time()
    # area = mesh_surface_area(vs, fs)
    # # print 'area: %.2f' % area
    # sys.stderr.write('compute surface area: %.2f seconds\n' % (time.time() - t)) #

    t = time.time()

#     if return_mesh:
#         return vs, fs

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
        sys.stderr.write('simplify %d @ %d: %.2f seconds\n' % (simplify_iter, n_pts, time.time() - t)) #

        if polydata.GetNumberOfPoints() < min_vertices:
            break


    if return_vertex_face_list:
        return polydata_to_mesh(polydata)
    else:
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


def vectormap_to_imagedata(arr, colors):

    vm = arr.reshape((-1, arr.shape[-1]))

    if np.array(colors).ndim == 1:
        colors = [colors] * len(vm)

    imagedata = vtk.vtkImageData()
    imagedata.SetDimensions([arr.shape[1], arr.shape[0], arr.shape[2]])
    imagedata.SetSpacing([1., 1., 1.])
    imagedata.AllocateScalars(vtk.VTK_FLOAT, arr.shape[3])

    imagedata.GetPointData().SetVectors(numpy_support.numpy_to_vtk(vm, deep=True, array_type=vtk.VTK_FLOAT))

    imagedata.GetPointData().SetScalars(numpy_support.numpy_to_vtk(colors, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR))

#     numpy_support.vtk_to_numpy(imagedata.GetPointData().GetScalars())
#     numpy_support.vtk_to_numpy(imagedata.GetPointData().GetVectors())

    return imagedata

def alpha_blending_v2(volumes, opacities):
    """
    Volumes must be aligned.
    Colors will be R,G,B,B,B...

    Args:
        volumes (list of 3d-arrays): single channel images.
        opacities (list of 3d-arrays of [0,1])
    """

    z = np.zeros_like(volumes[0].astype(np.float32))
    for i in range(len(volumes)-1):
        if i == 0:
            srcG = volumes[0].astype(np.float32)
            srcRGB = np.array([srcG, z, z])
            srcRGB = np.rollaxis(srcRGB, 0, 4)
            srcA = opacities[0]

            dstG = volumes[1].astype(np.float32)
            dstRGB = np.array([z, dstG, z])
            dstRGB = np.rollaxis(dstRGB, 0, 4)
            dstA = opacities[1]
        else:
            srcA = outA
            srcRGB = outRGB

            dstA = opacities[i+1]
            dstRGB = np.array([z, z, volumes[i+1].astype(np.float32)])
            dstRGB = np.rollaxis(dstRGB, 0, 4)

        outA = srcA + dstA * (1-srcA)
        outRGB = (srcRGB * srcA[...,None] + dstRGB*dstA[...,None]*(1-srcA[...,None]))/outA[...,None]
        outRGB[outA==0] = 0

    return outRGB, outA

def volume_to_imagedata_v2(rgb, origin=(0,0,0), alpha=None):
    """
    The result is used for the setting IndependentComponentsOff with 4 components.

    Args:
        rgb ((w,h,d,3)-array): RGB volume
        alpha (3d-array): alpha channel
    """

    v1 = rgb[...,0]
    v2 = rgb[...,1]
    v3 = rgb[...,2]

    imagedata = vtk.vtkImageData()
    imagedata.SetDimensions([rgb.shape[1], rgb.shape[0], rgb.shape[2]])
    imagedata.SetSpacing([1., 1., 1.])
    imagedata.SetOrigin(origin[0], origin[1], origin[2])

    v1 = np.transpose(v1, [2,0,1])
    v1 = v1.flatten()
    v2 = np.transpose(v2, [2,0,1])
    v2 = v2.flatten()
    v3 = np.transpose(v3, [2,0,1])
    v3 = v3.flatten()
    if alpha is None:
        if rgb.dtype == np.uint8:
            alpha = 255*np.ones_like(v3)
        elif rgb.dtype == np.float32:
            alpha = np.ones_like(v3)
        else:
            raise Exception('Data type must be uint8 or float32.')
    else:
        alpha = np.transpose(alpha, [2,0,1])
        alpha = alpha.flatten()
    v4 = np.column_stack([v1, v2, v3, alpha])

    if rgb.dtype == np.uint8:
        t = vtk.VTK_UNSIGNED_CHAR
    elif rgb.dtype == np.float32:
        t = vtk.VTK_FLOAT
    else:
        raise Exception('Data type must be uint8 or float32.')

    imagedata.GetPointData().SetScalars(numpy_support.numpy_to_vtk(v4, deep=True, array_type=t)) # deep copy must be true
    return imagedata

def volume_to_imagedata(arr, origin=(0,0,0), auxdata=None):
    """
    Args:
        arr (3d-array of uint8 or float32):
        origin (3-tuple): the origin coordinate of the given volume
        auxdata (3d-array): add one additional component to the data.

    Returns:
        (vtkImageData): Each point (in vtk parlance) gets ONE scalar value which is the value of an input volume voxel. Respects the (x,y,z) dimension ordering.
    """

    imagedata = vtk.vtkImageData()
    imagedata.SetDimensions([arr.shape[1], arr.shape[0], arr.shape[2]])
    imagedata.SetSpacing([1., 1., 1.])
    imagedata.SetOrigin(origin[0], origin[1], origin[2])

    v3 = np.transpose(arr, [2,0,1])
    v3 = v3.flatten()
    if auxdata is not None:
        auxdata = np.transpose(auxdata, [2,0,1])
        v3 = np.column_stack([v3, auxdata.flatten()])

    if arr.dtype == np.uint8:
        t = vtk.VTK_UNSIGNED_CHAR
    elif arr.dtype == np.float32:
        t = vtk.VTK_FLOAT
    else:
        raise Exception('Data type must be uint8 or float32.')

    imagedata.GetPointData().SetScalars(numpy_support.numpy_to_vtk(v3, deep=True, array_type=t)) # deep copy must be true
    return imagedata

############################### VTK Utils #####################################

def take_screenshot_as_numpy(win, magnification=10):
    windowToImageFilter = vtk.vtkWindowToImageFilter()

    windowToImageFilter.SetInput(win);
    windowToImageFilter.SetMagnification(magnification);
    # output image will be `magnification` times the render window size
    windowToImageFilter.SetInputBufferTypeToRGBA();
    windowToImageFilter.ReadFrontBufferOff();
    windowToImageFilter.Update();

    # https://stackoverflow.com/questions/14553523/vtk-render-window-image-to-numpy-array
    vtk_image = windowToImageFilter.GetOutput()
    height, width, _ = vtk_image.GetDimensions()
    vtk_array = vtk_image.GetPointData().GetScalars()
    components = vtk_array.GetNumberOfComponents()
    arr = numpy_support.vtk_to_numpy(vtk_array).reshape(height, width, components)
    return arr

def take_screenshot(win, file_path, magnification=10):

    windowToImageFilter = vtk.vtkWindowToImageFilter()

    windowToImageFilter.SetInput(win);
    windowToImageFilter.SetMagnification(magnification);
    # output image will be `magnification` times the render window size
    windowToImageFilter.SetInputBufferTypeToRGBA();
    windowToImageFilter.ReadFrontBufferOff();
    windowToImageFilter.Update();

    writer = vtk.vtkPNGWriter()
    create_parent_dir_if_not_exists(file_path)
    writer.SetFileName(file_path);
    writer.SetInputConnection(windowToImageFilter.GetOutputPort());
    writer.Write();

def actor_sphere(position=(0,0,0), radius=.5, color=(1., 1., 1.), opacity=1.):
    sphereSource = vtk.vtkSphereSource()
    sphereSource.SetCenter(position[0], position[1], position[2])
    sphereSource.SetRadius(radius)

    #create a mapper
    sphereMapper = vtk.vtkPolyDataMapper()
    sphereMapper.SetInputConnection(sphereSource.GetOutputPort())

    #create an actor
    sphereActor = vtk.vtkActor()
    sphereActor.SetMapper(sphereMapper)
    sphereActor.GetProperty().SetColor(color)
    sphereActor.GetProperty().SetOpacity(opacity)

    return sphereActor


def add_axes(iren, text_color=(1,1,1)):

    axes = vtk.vtkAxesActor()

    # put axes at origin
    transform = vtk.vtkTransform()
    transform.Translate(0.0, 0.0, 0.0);
    axes.SetUserTransform(transform)

    axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(text_color[0],text_color[1],text_color[2]);
    axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetColor(text_color[0],text_color[1],text_color[2]);
    axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetColor(text_color[0],text_color[1],text_color[2]);

    widget = vtk.vtkOrientationMarkerWidget()
    widget.SetOutlineColor( 0.9300, 0.5700, 0.1300 );
    widget.SetOrientationMarker( axes );
    widget.SetInteractor( iren );
    # widget.SetViewport( 0.0, 0.0, 0.2, 0.2 );
    widget.SetEnabled( 1 );
    widget.InteractiveOn();
    return widget

def load_mesh_stl(fn, return_polydata_only=False):
    """
    Args:
        return_polydata_only (bool): If true, return polydata; if false (default), return (vertices, faces)
    """

    if not os.path.exists(fn):
        sys.stderr.write('load_mesh_stl: File does not exist %s\n' % fn)
        return None

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

def save_mesh_ply(polydata, fn, color=(255,255,255)):
    # Export to rgb PLY file (STL file can not have color info)
    # https://github.com/Kitware/VTK/blob/master/IO/PLY/Testing/Python/TestPLYReadWrite.py#L31
    # http://www.vtk.org/doc/nightly/html/classvtkPLYWriter.html#aa7f0bdbb2decdc7a7360a890a6c10e8b

    writer = vtk.vtkPLYWriter()
    writer.SetFileName(fn)
    writer.SetInputData(polydata)
    writer.SetColorModeToUniformColor()
    writer.SetColor(color[0],color[1],color[2])
    writer.SetFileTypeToASCII();
    writer.Write()

def save_mesh(polydata, fn, color=(255,255,255)):

    create_if_not_exists(os.path.dirname(fn))

    if fn.endswith('.ply'):
        save_mesh_ply(polydata, fn, color=color)
    elif fn.endswith('.stl'):
        save_mesh_stl(polydata, fn)
    else:
        raise Exception('Mesh format must be ply or stl')

# http://stackoverflow.com/questions/32636503/how-to-get-the-key-code-in-a-vtk-keypressevent-using-python
try: # for environment that does not have vtk.
    class MyInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):

        def __init__(self, iren=None, renWin=None, snapshot_fn=None, camera=None):
            self.iren = iren
            self.renWin = renWin
            self.snapshot_fn = snapshot_fn
            self.camera = camera

            self.AddObserver("KeyPressEvent",self.keyPressEvent)

        def keyPressEvent(self,obj,event):
            key = self.iren.GetKeySym()
            if key == 'g':
                print 'viewup: %f, %f, %f\n' % self.camera.GetViewUp()
                print 'focal point: %f, %f, %f\n' % self.camera.GetFocalPoint()
                print 'position: %f, %f, %f\n' % self.camera.GetPosition()
            elif key == 'e':
                print 'Quit.'
                self.renWin.Finalize()
                self.iren.TerminateApp()
            elif key == 's':
                take_screenshot(self.renWin, self.snapshot_fn, 1)
            # return
except:
    pass

class vtkRecordVideoTimerCallback():
    def __init__(self, win, iren, camera, movie_fp, framerate=10):
        self.timer_count = 0
        self.movie_fp = movie_fp
        self.framerate = framerate
        self.iren = iren
        self.win = win
        self.camera = camera

        self.start_tick = 5 # wait 5 second then start

        self.azimuth_stepsize = 5.
        self.elevation_stepsize = 5.
        self.azimuth_rotation_start_tick = self.start_tick
        self.azimith_rotation_end_tick = self.azimuth_rotation_start_tick + 360./self.azimuth_stepsize
        self.elevation_rotation_start_tick = self.azimith_rotation_end_tick
        self.elevation_rotation_end_tick = self.elevation_rotation_start_tick + 360./self.elevation_stepsize

        self.finish_tick = self.elevation_rotation_end_tick

        create_parent_dir_if_not_exists('/tmp/brain_video/')
        execute_command('rm /tmp/brain_video/*')

    def execute(self,obj,event):

        # print self.timer_count
        # for actor in self.actors:
        #     actor.SetPosition(self.timer_count, self.timer_count,0)

        if self.timer_count >= self.start_tick:

            if self.timer_count >= self.azimuth_rotation_start_tick and self.timer_count < self.azimith_rotation_end_tick:
                self.camera.Azimuth(self.azimuth_stepsize)
            elif self.timer_count >= self.elevation_rotation_start_tick and self.timer_count < self.elevation_rotation_end_tick:
                self.camera.Elevation(self.elevation_stepsize)
                self.camera.OrthogonalizeViewUp() # This is important! http://vtk.1045678.n5.nabble.com/rotating-vtkCamera-td1232623.html
            # arr = take_screenshot_as_numpy(self.win, magnification=1)

        if self.movie_fp is not None:
            take_screenshot(self.win, '/tmp/brain_video/%03d.png' % self.timer_count, magnification=1)

            if self.timer_count == self.finish_tick:

                cmd = '/home/yuncong/ffmpeg-3.4.1-64bit-static/ffmpeg -framerate %(framerate)d -pattern_type glob -i "/tmp/brain_video/*.png" -c:v libx264 -vf "scale=-1:1080,format=yuv420p" %(output_fp)s' % \
                {'framerate': self.framerate, 'output_fp': self.movie_fp}
                execute_command(cmd)

                self.win.Finalize()
                self.iren.TerminateApp()
                del self.iren, self.win
                return

        self.win.Render()
        self.timer_count += 1


def launch_vtk(actors, init_angle='45', window_name=None, window_size=None,
            interactive=True, snapshot_fn=None, snapshot_magnification=3,
            axes=True, background_color=(1,1,1), axes_label_color=(1,1,1),
            animate=False, movie_fp=None, framerate=10,
              view_up=None, position=None, focal=None, distance=1, depth_peeling=True):
    """
    Press q to close render window.
    s to take snapshot.
    g to print current viewup/position/focal.
    """

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(background_color)

    renWin = vtk.vtkRenderWindow()
    renWin.SetSize(1200,1080)
    # renWin.SetFullScreen(1)
    renWin.AddRenderer(renderer)

    ##########################

    # cullers = ren.GetCullers()
    # cullers.InitTraversal()
    # culler = cullers.GetNextItem()
    # # culler.SetSortingStyleToBackToFront()
    # culler.SetSortingStyleToFrontToBack()

    ##########################################
    if depth_peeling:
        # Enable depth peeling
        # http://www.vtk.org/Wiki/VTK/Examples/Cxx/Visualization/CorrectlyRenderTranslucentGeometry

        # 1. Use a render window with alpha bits (as initial value is 0 (false)):
        renWin.SetAlphaBitPlanes(True)

        # 2. Force to not pick a framebuffer with a multisample buffer
        # (as initial value is 8):
        renWin.SetMultiSamples(0);

        # 3. Choose to use depth peeling (if supported) (initial value is 0 (false)):
        renderer.SetUseDepthPeeling(True);

        # 4. Set depth peeling parameters
        # - Set the maximum number of rendering passes (initial value is 4):
        maxNoOfPeels = 8
        renderer.SetMaximumNumberOfPeels(maxNoOfPeels);
        # - Set the occlusion ratio (initial value is 0.0, exact image):
        occlusionRatio = 0.0
        renderer.SetOcclusionRatio(occlusionRatio);

    ##########################################

    camera = vtk.vtkCamera()

    if view_up is not None and position is not None and focal is not None:
        camera.SetViewUp(view_up[0], view_up[1], view_up[2])
        camera.SetPosition(position[0], position[1], position[2])
        camera.SetFocalPoint(focal[0], focal[1], focal[2])

    elif init_angle == '15':

        # 30 degree
        camera.SetViewUp(0, -1, 0)
        camera.SetPosition(-20, -20, -20)
        camera.SetFocalPoint(1, 1, 1)

    elif init_angle == '30':

        # 30 degree
        camera.SetViewUp(0, -1, 0)
        camera.SetPosition(-10, -5, -5)
        camera.SetFocalPoint(1, 1, 1)

    elif init_angle == '45':

        # 45 degree
        camera.SetViewUp(0, -1, 0)
        camera.SetPosition(-20, -30, -10)
        camera.SetFocalPoint(1, 1, 1)

    elif init_angle == 'sagittal': # left to right

        camera.SetViewUp(0, -1, 0)
        camera.SetPosition(0, 0, -distance)
        camera.SetFocalPoint(0, 0, 1)

    elif init_angle == 'coronal' or init_angle == 'coronal_posteriorToAnterior' :
        # posterior to anterior

        # coronal
        camera.SetViewUp(1.1, 0, 0)
        camera.SetPosition(-distance, 0, 0)
        camera.SetFocalPoint(-1, 0, 0)

#     elif init_angle == 'coronal_anteriorToPosterior':

#         # coronal
#         camera.SetViewUp(0, -1, 0)
#         camera.SetPosition(-2, 0, 0)
#         camera.SetFocalPoint(-1, 0, 0)

    elif init_angle == 'horizontal_bottomUp':

        # horizontal
        camera.SetViewUp(0, 0, -1)
        camera.SetPosition(0, distance, 0)
        camera.SetFocalPoint(0, -1, 0)

    elif init_angle == 'horizontal_topDown':

        # horizontal
        camera.SetViewUp(0, 0, 1)
        camera.SetPosition(0, -distance, 0)
        camera.SetFocalPoint(0, 1, 0)
    else:
        raise Exception("init_angle %s is not recognized." % init_angle)

    renderer.SetActiveCamera(camera)
    renderer.ResetCamera()

    # This must be before  renWin.render(), otherwise the animation is stuck.
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    int_style = MyInteractorStyle(iren=iren, renWin=renWin, snapshot_fn='/tmp/tmp.png', camera=camera)
    iren.SetInteractorStyle(int_style) # Have problem closing window if use this

    for actor in actors:
        if actor is not None:
            renderer.AddActor(actor)

    renWin.Render()

    if window_name is not None:
        renWin.SetWindowName(window_name)

    if window_size is not None:
        renWin.SetSize(window_size)

    ##################

    if axes:
        axes = add_axes(iren, text_color=axes_label_color)

    if animate:
        # http://www.vtk.org/Wiki/VTK/Examples/Python/Animation
        iren.Initialize()

        # Sign up to receive TimerEvent from interactor
        cb = vtkRecordVideoTimerCallback(movie_fp=movie_fp, win=renWin, iren=iren, camera=camera, framerate=framerate)
        cb.actors = actors
        iren.AddObserver('TimerEvent', cb.execute)
        timerId = iren.CreateRepeatingTimer(1000); # This cannot be too fast because otherwise image export cannot catch up.

    ##################

    if interactive:
        # if not animate:
        #     iren.Initialize()
        iren.Start()
    else:
        take_screenshot(renWin, snapshot_fn, magnification=snapshot_magnification)

    del int_style.iren
    del int_style.renWin

    if animate:
        if hasattr(cb, 'iren'):
            del cb.iren
        if hasattr(cb, 'win'):
            del cb.win
    # In order for window to successfully close, MUST MAKE SURE NO REFERENCE
    # TO IREN AND WIN still remain.

# def close_window(iren):
#     render_window = iren.GetRenderWindow()
#     render_window.Finalize()
#     iren.TerminateApp()

################## Functions for generating actors #######################

def actor_arrows(anchor_point, anchor_vector0, anchor_vector1, anchor_vector2, opacity=1.):
    arrowSource = vtk.vtkArrowSource()

    actors = []
    for anchor_vector, c in zip([anchor_vector2, anchor_vector1, anchor_vector0],
                                [(1.,0.,0.),(0.,1.,0.),(0.,0.,1.)]):

        length = np.linalg.norm(anchor_vector)
        normalizedX = anchor_vector/length

        arbitrary = np.random.uniform(-10, 10, 3)
        normalizedZ = np.cross(normalizedX, arbitrary)
        normalizedZ = normalizedZ/np.linalg.norm(normalizedZ)
        normalizedY = np.cross(normalizedZ, normalizedX)
        normalizedY = normalizedY/np.linalg.norm(normalizedY)

        matrix = vtk.vtkMatrix4x4()

        # Create the direction cosine matrix
        matrix.Identity()
        for i in range(3):
            matrix.SetElement(i, 0, normalizedX[i])
            matrix.SetElement(i, 1, normalizedY[i])
            matrix.SetElement(i, 2, normalizedZ[i])

        # Apply the transforms
        transform = vtk.vtkTransform()
        transform.Translate(anchor_point)
        transform.Concatenate(matrix)
        transform.Scale(length, length, length)

        # Transform the polydata
        transformPD = vtk.vtkTransformPolyDataFilter()
        transformPD.SetTransform(transform)
        transformPD.SetInputConnection(arrowSource.GetOutputPort())

        #Create a mapper and actor for the arrow
        m = vtk.vtkPolyDataMapper()
        m.SetInputConnection(transformPD.GetOutputPort())

        a = vtk.vtkActor()
        a.SetMapper(m)
        a.GetProperty().SetColor(c)
        a.GetProperty().SetOpacity(opacity)

        actors.append(a)

    return actors



def actor_ellipse(anchor_point, anchor_vector0, anchor_vector1, anchor_vector2,
                 color=(1.,0.,0.), wireframe=False, opacity=1.):

    length0 = np.linalg.norm(anchor_vector0)
    normalizedX = anchor_vector0/length0

    length1 = np.linalg.norm(anchor_vector1)
    normalizedY = anchor_vector1/length1

    length2 = np.linalg.norm(anchor_vector2)
    normalizedZ = anchor_vector2/length2

    matrix = vtk.vtkMatrix4x4()

    # Create the direction cosine matrix
    matrix.Identity()
    for i in range(3):
        matrix.SetElement(i, 0, normalizedX[i])
        matrix.SetElement(i, 1, normalizedY[i])
        matrix.SetElement(i, 2, normalizedZ[i])

    # Apply the transforms
    transform = vtk.vtkTransform()
    transform.Translate(anchor_point)
    transform.Concatenate(matrix)
    transform.Scale(length0, length1, length2)

    # Transform the polydata
    transformPD = vtk.vtkTransformPolyDataFilter()
    transformPD.SetTransform(transform)

    sphereSource = vtk.vtkSphereSource()
    transformPD.SetInputConnection(sphereSource.GetOutputPort())

    #Create a mapper and actor for the arrow
    m = vtk.vtkPolyDataMapper()
    m.SetInputConnection(transformPD.GetOutputPort())

    a = vtk.vtkActor()
    a.SetMapper(m)

    if wireframe:
        a.GetProperty().SetRepresentationToWireframe()

    a.GetProperty().SetColor(color)
    a.GetProperty().SetOpacity(opacity)

    return a

def actor_volume_v2(rgb, alpha=None, origin=(0,0,0)):
    """
    Args:
        volumes ((w,h,d,3)-array): RGB volume
        alpha (3d-array same shape as volume): alpha volume
    """

    imagedata = volume_to_imagedata_v2(rgb=rgb, origin=origin, alpha=alpha)

    volumeMapper = vtk.vtkSmartVolumeMapper()
    volumeMapper.SetBlendModeToComposite()
    volumeMapper.SetInputData(imagedata)
    volumeProperty = vtk.vtkVolumeProperty()

    volumeProperty.IndependentComponentsOff()

    compositeOpacity = vtk.vtkPiecewiseFunction()
    compositeOpacity.AddPoint(0.0, 0.0)
    compositeOpacity.AddPoint(1.0, 1.0)
    volumeProperty.SetScalarOpacity(0, compositeOpacity)

    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    return volume

def actor_volume(volume, what, auxdata=None, origin=(0,0,0), c=(1,1,1), tb_colors=None, tb_opacity=.05,
                white_more_transparent=True):
    """
    Args:
        volume (3d-array)
        what (str): tb, score or probability. A caveat when what="probability" is that zero-valued voxels are not transparent, so later actors will block previous actors.
        c (3-tuple): color
        tb_colors (dict {int: 3-tuple}): step points of color transfer function that maps intensity value to color tuple.
        auxdata (3d-array same shape as volume)
    """

    imagedata = volume_to_imagedata(volume, origin=origin, auxdata=auxdata)

    volumeMapper = vtk.vtkSmartVolumeMapper()
    volumeMapper.SetBlendModeToComposite()

    # volumeMapper.SetBlendModeToAdditive()
    # volumeMapper.SetBlendModeToMinimumIntensity()
    # volumeMapper.SetBlendModeToMaximumIntensity()
    # volumeMapper.SetBlendModeToAverageIntensity()

    # Setting this results in blank
    # funcRayCast = vtk.vtkVolumeRayCastCompositeFunction()
    # funcRayCast.SetCompositeMethodToInterpolateFirst()
    # volumeMapper = vtk.vtkVolumeRayCastMapper()
    # volumeMapper.SetVolumeRayCastFunction(funcRayCast)

    volumeMapper.SetInputData(imagedata)
    volumeProperty = vtk.vtkVolumeProperty()

    if what == 'tb':

        if white_more_transparent:

            compositeOpacity = vtk.vtkPiecewiseFunction()
            compositeOpacity.AddPoint(0.0, 0.)

            if tb_colors is not None:
                for v, c in sorted(tb_colors.items()):
                    vl = v - .5
                    vr = v + .5
                    cp1 = vl-.25
                    cp2 = vr-.25
                    compositeOpacity.AddPoint(cp1, .5*cp1/200., .5, 1.)
                    compositeOpacity.AddPoint(v, 1., .5, 1.)
                    compositeOpacity.AddPoint(cp2, .5*cp2/200., .5, 1.)
                compositeOpacity.AddPoint(vr, .5*vr/200.)
            compositeOpacity.AddPoint(240., tb_opacity)
            compositeOpacity.AddPoint(255.0, tb_opacity)
        else:
            compositeOpacity = vtk.vtkPiecewiseFunction()
            compositeOpacity.AddPoint(255.0, 0.)

            if tb_colors is not None:
                for v, c in sorted(tb_colors.items()):
                    vl = v - .5
                    vr = v + .5
                    cp1 = vl-.25
                    cp2 = vr-.25
                    compositeOpacity.AddPoint(cp1, .5*cp1/200., .5, 1.)
                    compositeOpacity.AddPoint(v, 1., .5, 1.)
                    compositeOpacity.AddPoint(cp2, .5*cp2/200., .5, 1.)
                compositeOpacity.AddPoint(vr, .5*vr/200.)

            compositeOpacity.AddPoint(15., tb_opacity)
            compositeOpacity.AddPoint(0., tb_opacity)
            # compositeOpacity.AddPoint(240., .5)
            # compositeOpacity.AddPoint(255.0, .5)

        color = vtk.vtkColorTransferFunction()
        color.AddRGBPoint(0.0, 0,0,0)

        if tb_colors is not None:
            for v, c in sorted(tb_colors.items()):
                vl = v - .5
                vr = v + .5
                cp1 = vl-.25
                cp2 = vr-.25
                color.AddRGBPoint(cp1, .5*cp1/200., .5*cp1/200., .5*cp1/200., .5, 1.)
                color.AddRGBPoint(v, c[0], c[1], c[2], .5, 1.)
                color.AddRGBPoint(cp2, .5*cp2/200., .5*cp2/200., .5*cp2/200., .5, 1.)
            color.AddRGBPoint(vr, .5*vr/200., .5*vr/200., .5*vr/200.)
        color.AddRGBPoint(200.0, .5,.5,.5)
        color.AddRGBPoint(255.0, 1,1,1)

    # volumeGradientOpacity = vtk.vtkPiecewiseFunction()
    # volumeGradientOpacity.AddPoint(0,   0.0)
    # volumeGradientOpacity.AddPoint(1,  0.5)
    # volumeGradientOpacity.AddPoint(2, 1.0)

    elif what == 'score':

        volumeProperty.IndependentComponentsOff()

        compositeOpacity = vtk.vtkPiecewiseFunction()
        compositeOpacity.AddPoint(0.0, 0.0)
        # compositeOpacity.AddPoint(0.95, 0.0)
        compositeOpacity.AddPoint(1.0, 1.0)

        color = vtk.vtkColorTransferFunction()
        color.AddRGBPoint(0.0, 0,0,0)
        if tb_colors is not None:
            for v, c in sorted(tb_colors.items()):
                vl = v - .5
                vr = v + .5
                cp1 = vl-.25
                cp2 = vr-.25
                color.AddRGBPoint(cp1, .5*cp1/200., .5*cp1/200., .5*cp1/200., .5, 1.)
                color.AddRGBPoint(v, c[0], c[1], c[2], .5, 1.)
                color.AddRGBPoint(cp2, .5*cp2/200., .5*cp2/200., .5*cp2/200., .5, 1.)
            color.AddRGBPoint(vr, .5*vr/200., .5*vr/200., .5*vr/200.)
        color.AddRGBPoint(200.0, .5,.5,.5)
        color.AddRGBPoint(255.0, 1,1,1)

#         color.AddRGBPoint(0.0, c[0], c[1], c[2])
#         color.AddRGBPoint(1.0, c[0], c[1], c[2])

        # volumeGradientOpacity = vtk.vtkPiecewiseFunction()
        # volumeGradientOpacity.AddPoint(0,  0.0)
        # volumeGradientOpacity.AddPoint(10,  1.0)
        # volumeGradientOpacity.AddPoint(2, 1.0)

    elif what == 'probability':
        compositeOpacity = vtk.vtkPiecewiseFunction()
        compositeOpacity.AddPoint(0.0, 0.)
        compositeOpacity.AddPoint(.9, 0.05)
        compositeOpacity.AddPoint(1., 0.05)

        r,g,b = c

        color = vtk.vtkColorTransferFunction()
        color.AddRGBPoint(0.0, r,g,b)
        # color.AddRGBPoint(.95, .5,.5,.5)
        color.AddRGBPoint(1., r,g,b)

        # lookupTable = vtkLookupTable()
        # lookupTable.SetNumberOfTableValues(2);
        # lookupTable.SetRange(0.0,1.0);
        # lookupTable.SetTableValue( 0, 0.0, 0.0, 0.0, 0.0 ); #label 0 is transparent
        # lookupTable.SetTableValue( 1, 0.0, 1.0, 0.0, 1.0 ); #label 1 is opaque and green
        # lookupTable.Build()
        #
        # mapTransparency = vtkImageMapToColors()
        # mapTransparency.SetLookupTable(lookupTable)
        # mapTransparency.PassAlphaToOutputOn()
        # mapTransparency.SetInputData(maskImage)
        #
        # maskActor = vtkImageActor()
        # maskActor.GetMapper().SetInputConnection(mapTransparency.GetOutputPort())
        #
        #
        # volumeProperty.SetScalarOpacity(compositeOpacity)
        # volumeProperty.SetColor(color)
        #
        # volume = vtk.vtkVolume()
        # volume.SetMapper(volumeMapper)
        # volume.SetProperty(volumeProperty)
        #
        # return volume

    else:
        raise Exception('Color/opacity profile not recognized.')

    # volumeProperty.ShadeOff()
    volumeProperty.SetColor(color)
    volumeProperty.SetScalarOpacity(compositeOpacity)
    # Use the scalar at index 1
    # volumeProperty.SetScalarOpacity(0, compositeOpacity)
    # volumeProperty.SetGradientOpacity(volumeGradientOpacity)
    # volumeProperty.SetInterpolationTypeToLinear()

    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    return volume


def load_thumbnail_volume(stack, scoreVol_limit=None, convert_to_scoreSpace=False):

    tb_volume = bp.unpack_ndarray_file(volume_dir + "/%(stack)s/%(stack)s_down32Volume.bp" % {'stack': stack})

    if convert_to_scoreSpace:

        # from scipy.ndimage.interpolation import zoom
        # tb_volume_scaledToScoreVolume = img_as_ubyte(zoom(tb_volume, 2)[ymin:ymax+1, xmin:xmax+1, zmin:zmax+1])

        tb_xdim, tb_ydim, tb_zdim = tb_volume.shape

        if scoreVol_limit is None:

            # xmin, xmax, ymin, ymax, zmin, zmax = np.loadtxt(volume_dir + "/%(stack)s/%(stack)s_scoreVolume_limits.txt" % {'stack': stack}, np.int)
            xmin, xmax, ymin, ymax, zmin, zmax = \
            DataManager.load_volume_bbox(stack=stack, type='annotation', downscale=32)

        else:
            xmin, xmax, ymin, ymax, zmin, zmax = scoreVol_limit

        # SUPER SLOW!!
        # from scipy.ndimage import zoom
        # m = zoom(tb_volume, 2)

        m = np.zeros((tb_xdim*2, tb_ydim*2, tb_zdim*2), np.uint8)
        m[::2,::2,::2] = img_as_ubyte(tb_volume)
        m[::2,::2,1::2] = img_as_ubyte(tb_volume)
        m[::2,1::2,::2] = img_as_ubyte(tb_volume)
        m[::2,1::2,1::2] = img_as_ubyte(tb_volume)
        m[1::2,::2,::2] = img_as_ubyte(tb_volume)
        m[1::2,::2,1::2] = img_as_ubyte(tb_volume)
        m[1::2,1::2,::2] = img_as_ubyte(tb_volume)
        m[1::2,1::2,1::2] = img_as_ubyte(tb_volume)
        # tb_volume_scaledToScoreVolume = m[ymin:ymax+1, xmin:xmax+1, zmin:zmax+1].copy()
        tb_volume_scaledToScoreVolume = m[ymin:ymax+1, xmin:xmax+1, zmin:zmax+1].copy()

        return tb_volume_scaledToScoreVolume

    else:
        return tb_volume

def actor_mesh(polydata, color=(1.,1.,1.), wireframe=False, wireframe_linewidth=None, opacity=1., origin=(0,0,0)):
    """
    Args:
        color (float array): rgb between 0 and 1.
        origin: the initial shift for the mesh.
    """

    if polydata.GetNumberOfPoints() == 0:
        return None

    if origin[0] == 0 and origin[1] == 0 and origin[2] == 0:
        polydata_shifted = polydata
    else:
        polydata_shifted = move_polydata(polydata, origin)
        # Note that move_polydata() discards scalar data stored in polydata.

    m = vtk.vtkPolyDataMapper()
    m.SetInputData(polydata_shifted)
    a = vtk.vtkActor()
    a.SetMapper(m)

    # IF USE LOOKUP TABLE

    # from vtk.util.colors import *
    # lut = vtk.vtkLookupTable()
    # lut.SetNumberOfColors(256)
    # lut.Build()
    # for i in range(0, 16):
    #     lut.SetTableValue(i*16, red[0], red[1], red[2], 1)
    #     lut.SetTableValue(i*16+1, green[0], green[1], green[2], 1)
    #     lut.SetTableValue(i*16+2, blue[0], blue[1], blue[2], 1)
    #     lut.SetTableValue(i*16+3, black[0], black[1], black[2], 1)
    # m.SetLookupTable(lut)

    # m.ScalarVisibilityOn()
    # m.ScalarVisibilityOff()
    # m.SetScalarModeToDefault()
    # m.SetColorModeToDefault()
    # m.InterpolateScalarsBeforeMappingOff()
    # m.UseLookupTableScalarRangeOff()
    # m.ImmediateModeRenderingOff()
    # m.SetScalarMaterialModeToDefault()
    # m.GlobalImmediateModeRenderingOff()

    if wireframe:
        a.GetProperty().SetRepresentationToWireframe()
        if wireframe_linewidth is not None:
            a.GetProperty().SetLineWidth(wireframe_linewidth)

    a.GetProperty().SetColor(color)
    a.GetProperty().SetOpacity(opacity)

    return a

def polydata_heat_sphere(func, loc, phi_resol=100, theta_resol=100, radius=1, vmin=None, vmax=None):
    """
    Default color lookup table 0 = red, 1 = blue
    """

    sphereSource = vtk.vtkSphereSource()
    sphereSource.SetCenter(loc[0], loc[1], loc[2]);
    sphereSource.SetPhiResolution(phi_resol);
    sphereSource.SetThetaResolution(theta_resol);
    sphereSource.SetRadius(radius);
    sphereSource.Update()

    sphere_polydata = sphereSource.GetOutput()

    pts = (numpy_support.vtk_to_numpy(sphere_polydata.GetPoints().GetData()) - loc) / radius

    values = np.array([func(pt) for pt in pts])

    if vmin is None:
        vmin = values.min()
    if vmax is None:
        vmax = values.max()
    # print 'vmin', vmin, 'vmax', vmax
    values = (np.maximum(np.minimum(values, vmax), vmin) - vmin) / (vmax - vmin)

    val_arr = numpy_support.numpy_to_vtk(np.array(values), deep=1, array_type=vtk.VTK_FLOAT)
    # val_arr = numpy_support.numpy_to_vtk((np.array(values)*255).astype(np.uint8), deep=1, array_type=vtk.VTK_UNSIGNED_CHAR)
    sphere_polydata.GetPointData().SetScalars(val_arr)

    return sphere_polydata

# def mirror_volume(volume, origin):
#     """
#     Use to get the mirror image of the volume.
#     volume argument is the volume in right hemisphere.
#     origin argument is the origin of the mirrored result volume.
#
#     !!! This assumes the mirror plane is vertical; Consider adding a mirror plane as argument
#     """
#     ydim, xdim, zdim = volume.shape
#     real_origin = origin - (0,0, zdim-1)
#     volume = volume[:,:,::-1].copy()
#     return volume, real_origin

# from scipy.spatial import KDTree
# from collections import defaultdict

# def icp(fixed_pts, moving_pts, num_iter=10, rotation_only=True):
#     # https://www.wikiwand.com/en/Orthogonal_Procrustes_problem
#     # https://www.wikiwand.com/en/Kabsch_algorithm
#
#     fixed_pts_c0 = fixed_pts.mean(axis=0)
#     moving_pts_c0 = moving_pts.mean(axis=0)
#
#     fixed_pts_centered = fixed_pts - fixed_pts_c0
#     moving_pts_centered = moving_pts - moving_pts_c0
#
#     tree = KDTree(fixed_pts_centered)
#
#     moving_pts0 = moving_pts_centered.copy()
#
#     for i in range(num_iter):
#
#         t = time.time()
#
#         ds, nns = tree.query(moving_pts_centered)
# #         fixed_pts_nn = fixed_pts[nns]
#
#         a = defaultdict(list)
#         for mi, fi in enumerate(nns):
#             a[fi].append(mi)
#
#         inlier_moving_indices = []
#         inlier_fixed_indices = []
#         inlier_moving_pts = []
#         inlier_fixed_pts = []
#         for fi, mis in a.iteritems():
#             inlier_fixed_indices.append(fi)
#             mi = a[fi][np.argsort(ds[mis])[0]]
#             inlier_moving_indices.append(mi)
#             inlier_fixed_pts.append(fixed_pts_centered[fi])
#             inlier_moving_pts.append(moving_pts_centered[mi])
#
#         inlier_fixed_pts = np.array(inlier_fixed_pts)
#         inlier_moving_pts = np.array(inlier_moving_pts)
#         n_inlier = len(inlier_fixed_pts)
#
#         c_fixed = inlier_fixed_pts.mean(axis=0)
#         inlier_fixed_pts_centered = inlier_fixed_pts - c_fixed
#
#
#         c_moving = inlier_moving_pts.mean(axis=0)
#         inlier_moving_pts_centered = inlier_moving_pts - c_moving
#
#
#         random_indices = np.random.choice(range(n_inlier), 50)
#
#         inlier_fixed_pts_centered = inlier_fixed_pts_centered[random_indices]
#         inlier_moving_pts_centered = inlier_moving_pts_centered[random_indices]
#
#
#
#         M = np.dot(inlier_moving_pts_centered.T, inlier_fixed_pts_centered)
#
#         U, s, VT = np.linalg.svd(M)
#
#         if rotation_only:
#             s2 = np.ones_like(s)
#             s2[-1] = np.sign(np.linalg.det(np.dot(U, VT).T))
#             R = np.dot(np.dot(U, np.diag(s2)), VT).T
#             # print R
#         else:
#             R = np.dot(U, VT).T
#
#         moving_pts_centered = np.dot(moving_pts_centered - c_moving, R.T) + c_fixed
#
#         d = np.mean(np.sqrt(np.sum((inlier_moving_pts - inlier_fixed_pts)**2, axis=1)))
#         # if i > 1:
#         #    sys.stderr.write('mean change = %f\n' % abs(d_prev - d))
#         if i > 1 and abs(d_prev - d) < 1e-5:
#             break
#         d_prev = d
#
#         # sys.stderr.write('icp @ %d err %f @ %d inlier: %.2f seconds\n' % (i, d, len(inlier_moving_indices), time.time() - t))
#
#     c_fixed = inlier_fixed_pts.mean(axis=0)
#     inlier_fixed_pts_centered = inlier_fixed_pts - c_fixed
#
#     c_moving = moving_pts0[inlier_moving_indices].mean(axis=0)
#     inlier_moving_pts_centered =  moving_pts0[inlier_moving_indices] - c_moving
#
#     M = np.dot(inlier_moving_pts_centered.T, inlier_fixed_pts_centered)
#     U, _, VT = np.linalg.svd(M)
#     R = np.dot(U, VT).T
#
#     moving_pts_centered = np.dot(moving_pts0 - c_moving, R.T) + c_fixed
#
#     return moving_pts_centered + moving_pts_c0
#     # return moving_pts
#     # return moving_pts_centered

def average_shape(polydata_list=None, volume_origin_list=None, volume_list=None, origin_list=None, surface_level=None, num_simplify_iter=0, smooth=False, force_symmetric=False,
                 sigma=2., return_vertices_faces=False):
    """
    Compute the mean shape based on many co-registered volumes.

    Args:
        polydata_list (list of Polydata): List of meshes whose centroids are at zero.
        surface_level (float): If None, only return the probabilistic volume and origin. Otherwise, also return the surface mesh thresholded at the given percentage.
        num_simplify_iter (int): Number of simplification iterations for thresholded mesh generation.
        smooth (bool): Whether to smooth for thresholded mesh generation.
        force_symmetric (bool): If True, force the resulting volume and mesh to be symmetric wrt z.
        sigma (float): sigma of gaussian kernel used to smooth the probability values.

    Returns:
        average_volume_prob (3D ndarray):
        common_mins ((3,)-ndarray): coordinate of the volume's origin
        average_polydata (Polydata): mesh of the 3D boundary thresholded at concensus_percentage
    """

    if volume_origin_list is not None:
        volume_list, origin_list = map(list, zip(*volume_origin_list))

    if volume_list is None:
        volume_list = []
        origin_list = []

        for p in polydata_list:
            # t = time.time()
            v, orig, _ = polydata_to_volume(p)
            # sys.stderr.write('polydata_to_volume: %.2f seconds.\n' % (time.time() - t))
            volume_list.append(v)
            origin_list.append(np.array(orig, np.int))

    bbox_list = [(xm, xm+v.shape[1]-1, ym, ym+v.shape[0]-1, zm, zm+v.shape[2]-1) for v,(xm,ym,zm) in zip(volume_list, origin_list)]
    common_volume_list, common_volume_bbox = convert_vol_bbox_dict_to_overall_vol(vol_bbox_tuples=zip(volume_list, bbox_list))
    common_volume_list = map(lambda v: (v > 0).astype(np.int), common_volume_list)

    average_volume = np.sum(common_volume_list, axis=0)
    average_volume_prob = average_volume / float(average_volume.max())

    if force_symmetric:
        average_volume_prob = symmetricalize_volume(average_volume_prob)

    if sigma is not None:
        from skimage.filters import gaussian
        average_volume_prob = gaussian(average_volume_prob, sigma) # Smooth the probability

    common_origin = np.array(common_volume_bbox)[[0,2,4]]

    if surface_level is not None:
        average_volume_thresholded = average_volume_prob >= surface_level
        average_polydata = volume_to_polydata(volume=(average_volume_thresholded, common_origin), num_simplify_iter=num_simplify_iter, smooth=smooth, return_vertex_face_list=return_vertices_faces)
        return average_volume_prob, common_origin, average_polydata
    else:
        return average_volume_prob, common_origin


def symmetricalize_volume(prob_vol):
    """
    Replace the volume with the average of its left half and right half.
    """

    zc = prob_vol.shape[2]/2
    prob_vol_symmetric = prob_vol.copy()
    left_half = prob_vol[..., :zc]
    right_half = prob_vol[..., -zc:]
    left_half_averaged = (left_half + right_half[..., ::-1])/2.
    prob_vol_symmetric[..., :zc] = left_half_averaged
    prob_vol_symmetric[..., -zc:] = left_half_averaged[..., ::-1]
    return prob_vol_symmetric
