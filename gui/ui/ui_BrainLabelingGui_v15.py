# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/BrainLabelingGui_v15.ui'
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

class Ui_BrainLabelingGui(object):
    def setupUi(self, BrainLabelingGui):
        BrainLabelingGui.setObjectName(_fromUtf8("BrainLabelingGui"))
        BrainLabelingGui.resize(1521, 1113)
        self.centralwidget = QtGui.QWidget(BrainLabelingGui)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.gridLayout_2 = QtGui.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.splitter_3 = QtGui.QSplitter(self.centralwidget)
        self.splitter_3.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_3.setObjectName(_fromUtf8("splitter_3"))
        self.sagittal_gview = QtGui.QGraphicsView(self.splitter_3)
        self.sagittal_gview.setObjectName(_fromUtf8("sagittal_gview"))
        self.splitter_2 = QtGui.QSplitter(self.splitter_3)
        self.splitter_2.setOrientation(QtCore.Qt.Vertical)
        self.splitter_2.setObjectName(_fromUtf8("splitter_2"))
        self.splitter = QtGui.QSplitter(self.splitter_2)
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setObjectName(_fromUtf8("splitter"))
        self.coronal_gview = QtGui.QGraphicsView(self.splitter)
        self.coronal_gview.setObjectName(_fromUtf8("coronal_gview"))
        self.horizontal_gview = QtGui.QGraphicsView(self.splitter)
        self.horizontal_gview.setObjectName(_fromUtf8("horizontal_gview"))
        self.sagittal_tb_gview = QtGui.QGraphicsView(self.splitter)
        self.sagittal_tb_gview.setObjectName(_fromUtf8("sagittal_tb_gview"))
        self.layoutWidget = QtGui.QWidget(self.splitter_2)
        self.layoutWidget.setObjectName(_fromUtf8("layoutWidget"))
        self.gridLayout = QtGui.QGridLayout(self.layoutWidget)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 4, 0, 1, 1)
        self.button_save = QtGui.QPushButton(self.layoutWidget)
        self.button_save.setObjectName(_fromUtf8("button_save"))
        self.gridLayout.addWidget(self.button_save, 4, 7, 1, 1)
        self.button_load = QtGui.QPushButton(self.layoutWidget)
        self.button_load.setObjectName(_fromUtf8("button_load"))
        self.gridLayout.addWidget(self.button_load, 4, 8, 1, 1)
        spacerItem1 = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem1, 0, 6, 1, 1)
        self.button_displayStructures = QtGui.QPushButton(self.layoutWidget)
        self.button_displayStructures.setObjectName(_fromUtf8("button_displayStructures"))
        self.gridLayout.addWidget(self.button_displayStructures, 4, 4, 1, 1)
        self.button_displayOptions = QtGui.QPushButton(self.layoutWidget)
        self.button_displayOptions.setObjectName(_fromUtf8("button_displayOptions"))
        self.gridLayout.addWidget(self.button_displayOptions, 4, 6, 1, 1)
        self.button_inferSide = QtGui.QPushButton(self.layoutWidget)
        self.button_inferSide.setObjectName(_fromUtf8("button_inferSide"))
        self.gridLayout.addWidget(self.button_inferSide, 4, 5, 1, 1)
        self.button_stop = QtGui.QPushButton(self.layoutWidget)
        self.button_stop.setObjectName(_fromUtf8("button_stop"))
        self.gridLayout.addWidget(self.button_stop, 4, 3, 1, 1)
        self.label_7 = QtGui.QLabel(self.layoutWidget)
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.gridLayout.addWidget(self.label_7, 4, 1, 1, 1)
        self.lineEdit_username = QtGui.QLineEdit(self.layoutWidget)
        self.lineEdit_username.setObjectName(_fromUtf8("lineEdit_username"))
        self.gridLayout.addWidget(self.lineEdit_username, 4, 2, 1, 1)
        self.gridLayout_2.addWidget(self.splitter_3, 0, 0, 1, 1)
        BrainLabelingGui.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(BrainLabelingGui)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1521, 19))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        BrainLabelingGui.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(BrainLabelingGui)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        BrainLabelingGui.setStatusBar(self.statusbar)
        self.toolBar = QtGui.QToolBar(BrainLabelingGui)
        self.toolBar.setObjectName(_fromUtf8("toolBar"))
        BrainLabelingGui.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)

        self.retranslateUi(BrainLabelingGui)
        QtCore.QMetaObject.connectSlotsByName(BrainLabelingGui)

    def retranslateUi(self, BrainLabelingGui):
        BrainLabelingGui.setWindowTitle(_translate("BrainLabelingGui", "MainWindow", None))
        self.button_save.setText(_translate("BrainLabelingGui", "Save", None))
        self.button_load.setText(_translate("BrainLabelingGui", "Load", None))
        self.button_displayStructures.setText(_translate("BrainLabelingGui", "Display Structures", None))
        self.button_displayOptions.setText(_translate("BrainLabelingGui", "Display Options", None))
        self.button_inferSide.setText(_translate("BrainLabelingGui", "Infer Side", None))
        self.button_stop.setText(_translate("BrainLabelingGui", "Stop Loading", None))
        self.label_7.setText(_translate("BrainLabelingGui", "Username:", None))
        self.toolBar.setWindowTitle(_translate("BrainLabelingGui", "toolBar", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    BrainLabelingGui = QtGui.QMainWindow()
    ui = Ui_BrainLabelingGui()
    ui.setupUi(BrainLabelingGui)
    BrainLabelingGui.show()
    sys.exit(app.exec_())

