# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/accept_new_landmark_dialog.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_AcceptNewLandmarkDialog(object):
    def setupUi(self, AcceptNewLandmarkDialog):
        AcceptNewLandmarkDialog.setObjectName(_fromUtf8("AcceptNewLandmarkDialog"))
        AcceptNewLandmarkDialog.resize(460, 105)
        self.gridLayout = QtGui.QGridLayout(AcceptNewLandmarkDialog)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.buttonBox = QtGui.QDialogButtonBox(AcceptNewLandmarkDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.No|QtGui.QDialogButtonBox.Yes)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.gridLayout.addWidget(self.buttonBox, 2, 1, 1, 1)
        self.label = QtGui.QLabel(AcceptNewLandmarkDialog)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout.addWidget(self.label, 1, 1, 1, 1)

        self.retranslateUi(AcceptNewLandmarkDialog)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), AcceptNewLandmarkDialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), AcceptNewLandmarkDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(AcceptNewLandmarkDialog)

    def retranslateUi(self, AcceptNewLandmarkDialog):
        AcceptNewLandmarkDialog.setWindowTitle(_translate("AcceptNewLandmarkDialog", "Accept new structure?", None))
        self.label.setText(_translate("AcceptNewLandmarkDialog", "This set of regions have consistent texture. \n"
"Accept as a new structure?", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    AcceptNewLandmarkDialog = QtGui.QDialog()
    ui = Ui_AcceptNewLandmarkDialog()
    ui.setupUi(AcceptNewLandmarkDialog)
    AcceptNewLandmarkDialog.show()
    sys.exit(app.exec_())

