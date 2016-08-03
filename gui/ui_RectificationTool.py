# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'RectificationTool.ui'
#
# Created: Tue Aug  2 19:54:37 2016
#      by: PyQt4 UI code generator 4.10.4
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

class Ui_RectificationGUI(object):
    def setupUi(self, RectificationGUI):
        RectificationGUI.setObjectName(_fromUtf8("RectificationGUI"))
        RectificationGUI.resize(1521, 1113)
        self.centralwidget = QtGui.QWidget(RectificationGUI)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.coronal_gview = QtGui.QGraphicsView(self.centralwidget)
        self.coronal_gview.setGeometry(QtCore.QRect(870, 10, 631, 511))
        self.coronal_gview.setObjectName(_fromUtf8("coronal_gview"))
        self.horizontal_gview = QtGui.QGraphicsView(self.centralwidget)
        self.horizontal_gview.setGeometry(QtCore.QRect(10, 530, 841, 521))
        self.horizontal_gview.setObjectName(_fromUtf8("horizontal_gview"))
        self.sagittal_gview = QtGui.QGraphicsView(self.centralwidget)
        self.sagittal_gview.setGeometry(QtCore.QRect(10, 10, 841, 511))
        self.sagittal_gview.setObjectName(_fromUtf8("sagittal_gview"))
        self.button_sameX = QtGui.QPushButton(self.centralwidget)
        self.button_sameX.setGeometry(QtCore.QRect(930, 690, 98, 27))
        self.button_sameX.setObjectName(_fromUtf8("button_sameX"))
        self.button_sameY = QtGui.QPushButton(self.centralwidget)
        self.button_sameY.setGeometry(QtCore.QRect(930, 720, 98, 27))
        self.button_sameY.setObjectName(_fromUtf8("button_sameY"))
        self.button_sameZ = QtGui.QPushButton(self.centralwidget)
        self.button_sameZ.setGeometry(QtCore.QRect(930, 750, 98, 27))
        self.button_sameZ.setObjectName(_fromUtf8("button_sameZ"))
        RectificationGUI.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(RectificationGUI)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1521, 25))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        RectificationGUI.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(RectificationGUI)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        RectificationGUI.setStatusBar(self.statusbar)

        self.retranslateUi(RectificationGUI)
        QtCore.QMetaObject.connectSlotsByName(RectificationGUI)

    def retranslateUi(self, RectificationGUI):
        RectificationGUI.setWindowTitle(_translate("RectificationGUI", "MainWindow", None))
        self.button_sameX.setText(_translate("RectificationGUI", "Same x", None))
        self.button_sameY.setText(_translate("RectificationGUI", "Same y", None))
        self.button_sameZ.setText(_translate("RectificationGUI", "Same z", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    RectificationGUI = QtGui.QMainWindow()
    ui = Ui_RectificationGUI()
    ui.setupUi(RectificationGUI)
    RectificationGUI.show()
    sys.exit(app.exec_())

