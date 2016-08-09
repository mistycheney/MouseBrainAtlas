import sys
from PyQt4 import QtCore, QtGui
from stacko import Ui_Form

class Test(QtGui.QWidget, Ui_Form):
	def __init__(self):
		super(Test, self).__init__()
		print self
		self.setupUi(self)
		self.pushButton.clicked.connect(self.onClick)

	def onClick(self):
		print "clicked!"

if __name__ == "__main__":

	app = QtGui.QApplication(sys.argv)
	window = Test()
	window.show()
	sys.exit(app.exec_())
