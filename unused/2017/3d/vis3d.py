
"""
This example demonstrates cutting of a volume with a cap added
"""

import sys
import numpy as np

from vispy import scene, io

from vispy.geometry.generation import create_sphere
from vispy.color.colormap import get_colormaps

from vispy.io import load_data_file, read_png
from vispy.visuals.shaders import Function
import weakref

from vispy.visuals.transforms import STTransform, MatrixTransform

try:
    from sip import setapi
    setapi("QVariant", 2)
    setapi("QString", 2)
except ImportError:
    pass

from PyQt4 import QtGui, QtCore


class ObjectWidget(QtGui.QWidget):
    """
    Widget for editing OBJECT parameters
    """
    signal_objet_changed = QtCore.pyqtSignal(name='objectChanged')

    def __init__(self, parent=None):
        super(ObjectWidget, self).__init__(parent)

        l_nbr_steps = QtGui.QLabel("Slice at ")
        self.nbr_steps = QtGui.QSpinBox()
        self.nbr_steps.setMinimum(0)
        self.nbr_steps.setMaximum(2000)
        self.nbr_steps.setValue(150)
        self.nbr_steps.valueChanged.connect(self.update_param)
        # self.nbr_steps.keyboardTracking(False)

        gbox = QtGui.QGridLayout()
        gbox.addWidget(l_nbr_steps, 0, 0)
        gbox.addWidget(self.nbr_steps, 0, 1)

        vbox = QtGui.QVBoxLayout()
        vbox.addLayout(gbox)
        vbox.addStretch(1.0)

        self.setLayout(vbox)

    def update_param(self, option):
        self.signal_objet_changed.emit()

class MainWindow(QtGui.QMainWindow):

    def __init__(self):
        QtGui.QMainWindow.__init__(self)

        self.resize(950, 950)
        self.setWindowTitle('vispy example ...')

        splitter = QtGui.QSplitter(QtCore.Qt.Horizontal)

        self.canvas = Canvas()
        self.canvas.create_native()
        self.canvas.native.setParent(self)

        self.props = ObjectWidget()
        splitter.addWidget(self.props)
        splitter.addWidget(self.canvas.native)

        self.setCentralWidget(splitter)
        self.props.signal_objet_changed.connect(self.updateSlice)

    def updateSlice(self):
        self.canvas.updateSlice(self.props.nbr_steps.value())
       
class BlackToAlpha(object):
    def __init__(self):
        self.shader = Function("""
            void apply_alpha() {
                // perception
                const vec3 w = vec3(1, 1, 1);
                float value = dot(gl_FragColor.rgb, w);
                if ( value == 0) {
                    gl_FragColor.a = 0;
                }
            }
        """)

    def _attach(self, visual):
        self._visual = weakref.ref(visual)
        hook = visual._get_hook('frag', 'post')
        hook.add(self.shader())


class Canvas(scene.SceneCanvas):

    def __init__(self):
        scene.SceneCanvas.__init__(self, keys='interactive', size=(960, 960), show=True, bgcolor='black', title='MRI', vsync=False)

        self.unfreeze()
        self.view = self.central_widget.add_view()
        
        self.vol_data = np.load('/home/yuncong/CSHL_volumes/volume_MD589_thumbnail.npz')['arr_0']
        # self.vol_data = np.flipud(np.rollaxis(self.vol_data, 1))

        self.sectionTo = 150

        self.volume = scene.visuals.Volume(self.vol_data[:,0:self.sectionTo,:], parent=self.view.scene)
        self.volume.transform = scene.STTransform(translate=(0,0,0))
        CMAP=self.volume.cmap


        self.section2D = self.vol_data[:,self.sectionTo,:]


        self.plane = scene.visuals.Image(self.section2D, parent=self.view.scene, cmap=CMAP, relative_step_size=1.5)
        # self.plane.transform = scene.STTransform(translate=(0,self.sectionTo,0))
        # self.plane.transform = scene.STTransform(translate=(0,0,0))
        self.plane.transform = MatrixTransform()
        self.plane.transform.rotate(90, (1,0,0))
        self.plane.transform.translate((0,self.sectionTo,0))
        
        self.plane.attach(BlackToAlpha())

        self.view.camera = scene.cameras.ArcballCamera(parent=self.view.scene)


    def updateSlice(self,cutAt):

        self.sectionTo = cutAt
        self.volume.set_data(self.vol_data[:,0:self.sectionTo,:])

        self.section2D = self.vol_data[:,self.sectionTo,:]
        self.plane.set_data(self.section2D)

        self.plane.transform = MatrixTransform()
        self.plane.transform.rotate(90, (1,0,0))
        self.plane.transform.translate((0,self.sectionTo,0))

        # self.plane.transform = scene.STTransform(translate=(0,self.sectionTo,0))

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    appQt = QtGui.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    appQt.exec_()