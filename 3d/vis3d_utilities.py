import vtk
from vtk.util import numpy_support

import numpy as np
import sys

from skimage.measure import marching_cubes, correct_mesh_orientation, mesh_surface_area

from itertools import izip

import os
sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from metadata import *

import time
import mcubes # https://github.com/pmneila/PyMCubes

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

    t = time.time()
    area = mesh_surface_area(vs, fs)

    # print 'area: %.2f' % area

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


def volume_to_imagedata(arr, origin=(0,0,0)):

    imagedata = vtk.vtkImageData()
    imagedata.SetDimensions([arr.shape[1], arr.shape[0], arr.shape[2]])
    imagedata.SetSpacing([1., 1., 1.])
    imagedata.SetOrigin(origin[0], origin[1], origin[2])

    v3 = np.transpose(arr, [2,0,1])

    if arr.dtype == np.uint8:
        t = vtk.VTK_UNSIGNED_CHAR
    elif arr.dtype == np.float32:
        t = vtk.VTK_FLOAT
    else:
        sys.stderr.write('Data type must be uint8 or float32.\n')

    imagedata.GetPointData().SetScalars(numpy_support.numpy_to_vtk(v3.flat, deep=True, array_type=t)) # deep copy must be true
    return imagedata

############################### VTK Utils #####################################

def take_screenshot(win, file_path):

    windowToImageFilter = vtk.vtkWindowToImageFilter()

    windowToImageFilter.SetInput(win);
    windowToImageFilter.SetMagnification(3);
    windowToImageFilter.SetInputBufferTypeToRGBA();
    windowToImageFilter.ReadFrontBufferOff();
    windowToImageFilter.Update();

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(file_path);
    writer.SetInputConnection(windowToImageFilter.GetOutputPort());
    writer.Write();

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



def launch_vtk(actors, init_angle='30', window_name=None, window_size=None,
            interactive=True, snapshot_fn=None, axes=True, background_color=(0,0,0),
            animate=False, movie_fn=None):

    ren1 = vtk.vtkRenderer()
    ren1.SetBackground(background_color)

    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren1)

    for actor in actors:
        ren1.AddActor(actor)

    camera = vtk.vtkCamera()

    if init_angle == '15':

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

    elif init_angle == 'sagittal':

        # saggital
        camera.SetViewUp(0, -1, 0)
        camera.SetPosition(0, 0, -2)
        camera.SetFocalPoint(0, 0, 1)

    elif init_angle == 'coronal':

        # coronal
        camera.SetViewUp(0, -1, 0)
        camera.SetPosition(-2, 0, 0)
        camera.SetFocalPoint(-1, 0, 0)

    elif init_angle == 'horizontal_bottomUp':

        # horizontal
        camera.SetViewUp(0, 0, -1)
        camera.SetPosition(0, 1, 0)
        camera.SetFocalPoint(0, -1, 0)

    elif init_angle == 'horizontal_topDown':

        # horizontal
        camera.SetViewUp(0, 0, 1)
        camera.SetPosition(0, -1, 0)
        camera.SetFocalPoint(0, 1, 0)

    ren1.SetActiveCamera(camera)
    ren1.ResetCamera()

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    if axes:
        axes = add_axes(iren)

    renWin.Render()

    if window_name is not None:
        renWin.SetWindowName(window_name)

    if window_size is not None:
        renWin.SetSize(window_size)

    ##################
    # http://www.vtk.org/Wiki/VTK/Examples/Python/Animation

    if animate:

        iren.Initialize()
        # Sign up to receive TimerEvent
        cb = vtkTimerCallback()
        cb.actors = actors
        cb.camera = camera
        iren.AddObserver('TimerEvent', cb.execute)
        timerId = iren.CreateRepeatingTimer(100);
    ##################

    if interactive:
        iren.Start()

        # if movie_fn is not None:
        #
        #     windowToImageFilter = vtk.vtkWindowToImageFilter()
        #     windowToImageFilter.SetInput(renWin)
        #     windowToImageFilter.SetInputBufferTypeToRGBA()
        #     windowToImageFilter.ReadFrontBufferOff()
        #     windowToImageFilter.Update()
        #
        #     movieWriter = vtk.vtkAVIWriter()
        #     movieWriter.SetInputConnection(windowToImageFilter.GetOutputPort())
        #     movieWriter.SetFileName(movie_fn)
        #     movieWriter.Start()
        #
        #     imageFilter.Modified()
        #     moviewriter.Write()
        #
        #     moviewriter.End()
    else:
        take_screenshot(renWin, snapshot_fn)


class vtkTimerCallback():
    def __init__(self):
        self.timer_count = 0

    def execute(self,obj,event):
        iren = obj
        iren.GetRenderWindow().Render()
        self.camera.Azimuth(5.)
        self.timer_count += 1

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


def actor_volume(volume, what, origin=(0,0,0), c=(1,1,1)):

    imagedata = volume_to_imagedata(volume, origin=origin)

    volumeMapper = vtk.vtkSmartVolumeMapper()
    #     volumeMapper.SetBlendModeToComposite()
    volumeMapper.SetInputData(imagedata)

    volumeProperty = vtk.vtkVolumeProperty()
    #     volumeProperty.ShadeOff()
    # volumeProperty.SetInterpolationType(vtk.VTK_LINEAR_INTERPOLATION)

    if what == 'tb':

        compositeOpacity = vtk.vtkPiecewiseFunction()
        compositeOpacity.AddPoint(0.0, 0.0)
        compositeOpacity.AddPoint(0.9, 1.)
        compositeOpacity.AddPoint(1., 1.)
        compositeOpacity.AddPoint(1.1, 0.)
        compositeOpacity.AddPoint(240., 0.05)
        compositeOpacity.AddPoint(255.0, 0.05)

        color = vtk.vtkColorTransferFunction()
        color.AddRGBPoint(0.0, 0,0,0)
        color.AddRGBPoint(.9, 1,0,0)
        color.AddRGBPoint(1., 1,0,0)
        color.AddRGBPoint(1.1, 0,0,0)
        color.AddRGBPoint(200.0, .5,.5,.5)
        color.AddRGBPoint(255.0, 1,1,1)

        # compositeOpacity = vtk.vtkPiecewiseFunction()
        # compositeOpacity.AddPoint(0.0, 0.05)
        # compositeOpacity.AddPoint(20.0, 0.05)
        # compositeOpacity.AddPoint(240., 0.)
        # compositeOpacity.AddPoint(1.1, 0.)
        # compositeOpacity.AddPoint(240., 0.05)
        # compositeOpacity.AddPoint(255.0, 0.05)
        #
        # color = vtk.vtkColorTransferFunction()
        #
        # color.AddRGBPoint(0.0, 0,0,0)
        # color.AddRGBPoint(.9, 1,0,0)
        # color.AddRGBPoint(1., 1,0,0)
        # color.AddRGBPoint(1.1, 0,0,0)
        # color.AddRGBPoint(200.0, .5,.5,.5)
        # color.AddRGBPoint(255.0, 1,1,1)

    elif what == 'score':

        compositeOpacity = vtk.vtkPiecewiseFunction()
        compositeOpacity.AddPoint(0.0, 0.0)
        compositeOpacity.AddPoint(0.95, 0.0)
        compositeOpacity.AddPoint(1.0, 1.0)
        volumeProperty.SetScalarOpacity(compositeOpacity)

        color = vtk.vtkColorTransferFunction()
        c = (1., 1., 1.)
        color.AddRGBPoint(0.0, c[0], c[1], c[2])
        color.AddRGBPoint(255.0, c[0], c[1], c[2])

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
        sys.stderr.write('Color/opacity profile not recognized.\n')

    volumeProperty.SetScalarOpacity(compositeOpacity)
    volumeProperty.SetColor(color)

    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    return volume


def load_thumbnail_volume(stack, scoreVol_limit=None, convert_to_scoreSpace=False):

    tb_volume = bp.unpack_ndarray_file(volume_dir + "/%(stack)s/%(stack)s_thumbnailVolume.bp" % {'stack': stack})

    if convert_to_scoreSpace:

        # from scipy.ndimage.interpolation import zoom
        # tb_volume_scaledToScoreVolume = img_as_ubyte(zoom(tb_volume, 2)[ymin:ymax+1, xmin:xmax+1, zmin:zmax+1])

        tb_xdim, tb_ydim, tb_zdim = tb_volume.shape

        if scoreVol_limit is None:

            xmin, xmax, ymin, ymax, zmin, zmax = np.loadtxt(volume_dir + "/%(stack)s/%(stack)s_scoreVolume_limits.txt" % {'stack': stack}, np.int)

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


def load_score_volume(stack, name_u):

    vol_fn = volume_dir + '/%(stack)s/%(stack)s_scoreVolume_%(name)s.bp' % {'stack':stack, 'name':name_u}

    if not os.path.exists(vol_fn):
        download_volume(stack, 'score', dest_dir=volume_dir + '/%(stack)s' % {'stack': stack}, name_u=name_u)

    score_volume = bp.unpack_ndarray_file(vol_fn)

    return score_volume

def actor_mesh(polydata, color=(1.,1.,1.), wireframe=False, opacity=1.):

    m = vtk.vtkPolyDataMapper()
    m.SetInputData(polydata)

    a = vtk.vtkActor()
    a.SetMapper(m)
    if wireframe:
        a.GetProperty().SetRepresentationToWireframe()

    a.GetProperty().SetColor(color)
    a.GetProperty().SetOpacity(opacity)

    return a


def polydata_heat_sphere(func, loc, phi_resol=100, theta_resol=100, radius=1, vmin=None, vmax=None):

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
    values = (np.maximum(np.minimum(values, vmax), vmin) - vmin) / (vmax - vmin)

    val_arr = numpy_support.numpy_to_vtk(np.array(values), deep=1, array_type=vtk.VTK_FLOAT)
    sphere_polydata.GetPointData().SetScalars(val_arr)

    # default color: 0 = red, 1 = blue

    return sphere_polydata


###################################################################################

def mirror_volume(volume, origin):
    """
    Use to get the mirror image of the volume.
    volume is the volume in right hand orientation.
    """
    ydim, xdim, zdim = volume.shape
    real_origin = origin - (0,0, zdim-1)
    volume = volume[:,:,::-1].copy()
    return volume, real_origin

from scipy.spatial import KDTree
from collections import defaultdict

def icp(fixed_pts, moving_pts, num_iter=10, rotation_only=True):
    # https://www.wikiwand.com/en/Orthogonal_Procrustes_problem
    # https://www.wikiwand.com/en/Kabsch_algorithm


    fixed_pts_c0 = fixed_pts.mean(axis=0)
    moving_pts_c0 = moving_pts.mean(axis=0)

    fixed_pts_centered = fixed_pts - fixed_pts_c0
    moving_pts_centered = moving_pts - moving_pts_c0

    tree = KDTree(fixed_pts_centered)

    moving_pts0 = moving_pts_centered.copy()

    for i in range(num_iter):

        t = time.time()

        ds, nns = tree.query(moving_pts_centered)
#         fixed_pts_nn = fixed_pts[nns]

        a = defaultdict(list)
        for mi, fi in enumerate(nns):
            a[fi].append(mi)

        inlier_moving_indices = []
        inlier_fixed_indices = []
        inlier_moving_pts = []
        inlier_fixed_pts = []
        for fi, mis in a.iteritems():
            inlier_fixed_indices.append(fi)
            mi = a[fi][np.argsort(ds[mis])[0]]
            inlier_moving_indices.append(mi)
            inlier_fixed_pts.append(fixed_pts_centered[fi])
            inlier_moving_pts.append(moving_pts_centered[mi])

        inlier_fixed_pts = np.array(inlier_fixed_pts)
        inlier_moving_pts = np.array(inlier_moving_pts)
        n_inlier = len(inlier_fixed_pts)

        c_fixed = inlier_fixed_pts.mean(axis=0)
        inlier_fixed_pts_centered = inlier_fixed_pts - c_fixed


        c_moving = inlier_moving_pts.mean(axis=0)
        inlier_moving_pts_centered = inlier_moving_pts - c_moving


        random_indices = np.random.choice(range(n_inlier), 50)

        inlier_fixed_pts_centered = inlier_fixed_pts_centered[random_indices]
        inlier_moving_pts_centered = inlier_moving_pts_centered[random_indices]



        M = np.dot(inlier_moving_pts_centered.T, inlier_fixed_pts_centered)

        U, s, VT = np.linalg.svd(M)

        if rotation_only:
            s2 = np.ones_like(s)
            s2[-1] = np.sign(np.linalg.det(np.dot(U, VT).T))
            R = np.dot(np.dot(U, np.diag(s2)), VT).T
            # print R
        else:
            R = np.dot(U, VT).T

        moving_pts_centered = np.dot(moving_pts_centered - c_moving, R.T) + c_fixed

        d = np.mean(np.sqrt(np.sum((inlier_moving_pts - inlier_fixed_pts)**2, axis=1)))
        # if i > 1:
        #    sys.stderr.write('mean change = %f\n' % abs(d_prev - d))
        if i > 1 and abs(d_prev - d) < 1e-5:
            break
        d_prev = d

        # sys.stderr.write('icp @ %d err %f @ %d inlier: %.2f seconds\n' % (i, d, len(inlier_moving_indices), time.time() - t))

    c_fixed = inlier_fixed_pts.mean(axis=0)
    inlier_fixed_pts_centered = inlier_fixed_pts - c_fixed

    c_moving = moving_pts0[inlier_moving_indices].mean(axis=0)
    inlier_moving_pts_centered =  moving_pts0[inlier_moving_indices] - c_moving

    M = np.dot(inlier_moving_pts_centered.T, inlier_fixed_pts_centered)
    U, _, VT = np.linalg.svd(M)
    R = np.dot(U, VT).T

    moving_pts_centered = np.dot(moving_pts0 - c_moving, R.T) + c_fixed

    return moving_pts_centered + moving_pts_c0
    # return moving_pts
    # return moving_pts_centered


def average_shape(polydata_list, concensus_percentage=.5, num_simplify_iter=0, smooth=False):

    volume_list = []
    origin_list = []

    for p in polydata_list:
        t = time.time()
        v, orig, _ = polydata_to_volume(p)
        sys.stderr.write('polydata_to_volume: %.2f\n' % (time.time() - t))

        volume_list.append(v)
        origin_list.append(np.array(orig, np.int))

    t = time.time()

    common_mins = np.min(origin_list, axis=0).astype(np.int)
    relative_origins = origin_list - common_mins

    common_xdim, common_ydim, common_zdim = np.max([(v.shape[1]+o[0], v.shape[0]+o[1], v.shape[2]+o[2])
                                                    for v,o in zip(volume_list, relative_origins)], axis=0)

    common_volume_list = []

    for i, v in enumerate(volume_list):
        common_volume = np.zeros( (common_ydim, common_xdim, common_zdim), np.uint8)
        x0, y0, z0 = relative_origins[i]
        ydim, xdim, zdim = v.shape
        common_volume[y0:y0+ydim, x0:x0+xdim, z0:z0+zdim] = v

        common_volume_list.append((common_volume > 0).astype(np.int))


    average_volume = np.sum(common_volume_list, axis=0)
    average_volume_prob = average_volume / float(average_volume.max())

    average_volume_thresholded = average_volume >= max(2, len(common_volume_list)*concensus_percentage)

    sys.stderr.write('find common: %.2f\n' % (time.time() - t))

    t = time.time()
    average_polydata = volume_to_polydata(average_volume_thresholded, common_mins, num_simplify_iter=num_simplify_iter,
                                          smooth=smooth)
    sys.stderr.write('volume_to_polydata: %.2f\n' % (time.time() - t))

    return average_volume_prob, common_mins, average_polydata



def fit_plane(X):
    """
    Fit a plane to a set of 3d points

    Parameters
    ----------
    X : n x 3 array
        points

    Returns
    ------
    normal : (3,) vector
        the normal vector of the plane
    c : (3,) vector
        a point on the plane
    """

    # http://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
    # http://math.stackexchange.com/a/3871
    c = X.mean(axis=0)
    Xc = X - c
    U, _, VT = np.linalg.svd(Xc.T)
    return U[:,-1], c

def R_align_two_vectors(a, b):
    """
    Find the
    """
    # http://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/897677#897677

    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    v_skew = np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])
    R = np.eye(3) + v_skew + np.dot(v_skew, v_skew)*(1-c)/s**2
    return R

def average_location(centroid_allLandmarks):
    """
    Return (0,0,0) centered coordinates where (0,0,0) is a point on the midplane
    """

    mean_centroid_allLandmarks = {name: np.mean(centroids, axis=0)
                                  for name, centroids in centroid_allLandmarks.iteritems()}

    names = set([convert_name_to_unsided(name_s) for name_s in centroid_allLandmarks.keys()])

    # Fit a midplane from the midpoints of symmetric landmark centroids
    midpoints = {}
    for name in names:
        lname = convert_to_left_name(name)
        rname = convert_to_right_name(name)

#         names = labelMap_unsidedToSided[name]

#         # maybe ignoring singular instances is better
#         if len(names) == 2:
        if lname in mean_centroid_allLandmarks and rname in mean_centroid_allLandmarks:
            midpoints[name] = .5 * mean_centroid_allLandmarks[lname] + .5 * mean_centroid_allLandmarks[rname]
        else:
            midpoints[name] = mean_centroid_allLandmarks[name]

    midplane_normal, midplane_point = fit_plane(np.c_[midpoints.values()])

    print midplane_normal,'@', midplane_point

    R_to_canonical = R_align_two_vectors(midplane_normal , np.r_[0, 0, 1])

    points_midplane_oriented = {name: np.dot(R_to_canonical, p - midplane_point)
                                for name, p in mean_centroid_allLandmarks.iteritems()}

    canonical_locations = {}

    for name in names:

        lname = convert_to_left_name(name)
        rname = convert_to_right_name(name)

        if lname in points_midplane_oriented and rname in points_midplane_oriented:

            x, y, mz = .5 * points_midplane_oriented[lname] + .5 * points_midplane_oriented[rname]

            canonical_locations[lname] = np.r_[x, y, points_midplane_oriented[lname][2]-mz]
            canonical_locations[rname] = np.r_[x, y, points_midplane_oriented[rname][2]-mz]
        else:
            x, y, _ = points_midplane_oriented[name]
            canonical_locations[name] = np.r_[x, y, 0]

    return canonical_locations, midplane_point, midplane_normal
