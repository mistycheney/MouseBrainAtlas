from PyQt4 import QtGui
from matplotlib.backends.backend_qt4agg \
	import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# class MplCanvas(FigureCanvas):
#     def __init__(self):
        # FigureCanvas.setParent(self, self.centralwidget)
        # FigureCanvas.setFocusPolicy(self, Qt.StrongFocus)
        # FigureCanvas.setFocus(self)

		# self.fig = Figure()
		# self.ax = self.fig.add_subplot(111)
		# FigureCanvas.__init__(self, self.fig)
		# FigureCanvas.setSizePolicy(self, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
		# FigureCanvas.updateGeometry(self)

class MplWidget(QtGui.QWidget):
    def __init__(self, parent = None):
        QtGui.QWidget.__init__(self, parent)

        self.fig = Figure((10.0, 10.0), dpi=100)
        self.canvas = FigureCanvas(self.fig)

        self.vbl = QtGui.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)
