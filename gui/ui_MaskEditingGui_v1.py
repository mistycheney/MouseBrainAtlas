# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MaskEditingGui_v1.ui'
#
# Created: Wed Aug 17 18:35:05 2016
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

class Ui_MaskEditingGui(object):
    def setupUi(self, MaskEditingGui):
        MaskEditingGui.setObjectName(_fromUtf8("MaskEditingGui"))
        MaskEditingGui.resize(1521, 1113)
        self.centralwidget = QtGui.QWidget(MaskEditingGui)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.gridLayout = QtGui.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.section_gview = QtGui.QGraphicsView(self.centralwidget)
        self.section_gview.setObjectName(_fromUtf8("section_gview"))
        self.horizontalLayout.addWidget(self.section_gview)
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        spacerItem = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.button_load = QtGui.QPushButton(self.centralwidget)
        self.button_load.setObjectName(_fromUtf8("button_load"))
        self.verticalLayout.addWidget(self.button_load)
        self.button_save = QtGui.QPushButton(self.centralwidget)
        self.button_save.setObjectName(_fromUtf8("button_save"))
        self.verticalLayout.addWidget(self.button_save)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        MaskEditingGui.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MaskEditingGui)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1521, 25))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        MaskEditingGui.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MaskEditingGui)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MaskEditingGui.setStatusBar(self.statusbar)
        self.toolBar = QtGui.QToolBar(MaskEditingGui)
        self.toolBar.setObjectName(_fromUtf8("toolBar"))
        MaskEditingGui.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)

        self.retranslateUi(MaskEditingGui)
        QtCore.QMetaObject.connectSlotsByName(MaskEditingGui)

    def retranslateUi(self, MaskEditingGui):
        MaskEditingGui.setWindowTitle(_translate("MaskEditingGui", "MainWindow", None))
        self.button_load.setText(_translate("MaskEditingGui", "Load", None))
        self.button_save.setText(_translate("MaskEditingGui", "Save", None))
        self.toolBar.setWindowTitle(_translate("MaskEditingGui", "toolBar", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    MaskEditingGui = QtGui.QMainWindow()
    ui = Ui_MaskEditingGui()
    ui.setupUi(MaskEditingGui)
    MaskEditingGui.show()
    sys.exit(app.exec_())

