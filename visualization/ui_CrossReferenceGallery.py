# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'CrossReferenceGallery.ui'
#
# Created: Sat Jan 31 23:42:52 2015
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

class Ui_CrossReferenceGallery(object):
    def setupUi(self, CrossReferenceGallery):
        CrossReferenceGallery.setObjectName(_fromUtf8("CrossReferenceGallery"))
        CrossReferenceGallery.resize(1024, 600)
        self.cWidget = QtGui.QWidget(CrossReferenceGallery)
        self.cWidget.setObjectName(_fromUtf8("cWidget"))
        self.verticalLayout = QtGui.QVBoxLayout(self.cWidget)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.vLayout = QtGui.QVBoxLayout()
        self.vLayout.setSizeConstraint(QtGui.QLayout.SetDefaultConstraint)
        self.vLayout.setObjectName(_fromUtf8("vLayout"))
        self.topLayout = QtGui.QHBoxLayout()
        self.topLayout.setObjectName(_fromUtf8("topLayout"))
        self.leftListLayout = QtGui.QVBoxLayout()
        self.leftListLayout.setObjectName(_fromUtf8("leftListLayout"))
        self.stack_list = QtGui.QListView(self.cWidget)
        self.stack_list.setObjectName(_fromUtf8("stack_list"))
        self.leftListLayout.addWidget(self.stack_list)
        self.section_list = QtGui.QListView(self.cWidget)
        self.section_list.setObjectName(_fromUtf8("section_list"))
        self.leftListLayout.addWidget(self.section_list)
        self.topLayout.addLayout(self.leftListLayout)
        self.vLayout.addLayout(self.topLayout)
        self.bottomLayout = QtGui.QHBoxLayout()
        self.bottomLayout.setObjectName(_fromUtf8("bottomLayout"))
        self.buttonQuit = QtGui.QPushButton(self.cWidget)
        self.buttonQuit.setObjectName(_fromUtf8("buttonQuit"))
        self.bottomLayout.addWidget(self.buttonQuit)
        self.vLayout.addLayout(self.bottomLayout)
        self.verticalLayout.addLayout(self.vLayout)
        CrossReferenceGallery.setCentralWidget(self.cWidget)
        self.menubar = QtGui.QMenuBar(CrossReferenceGallery)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1024, 25))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        CrossReferenceGallery.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(CrossReferenceGallery)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        CrossReferenceGallery.setStatusBar(self.statusbar)

        self.retranslateUi(CrossReferenceGallery)
        QtCore.QMetaObject.connectSlotsByName(CrossReferenceGallery)

    def retranslateUi(self, CrossReferenceGallery):
        CrossReferenceGallery.setWindowTitle(_translate("CrossReferenceGallery", "Cross Reference Gallery", None))
        self.buttonQuit.setText(_translate("CrossReferenceGallery", "Quit", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    CrossReferenceGallery = QtGui.QMainWindow()
    ui = Ui_CrossReferenceGallery()
    ui.setupUi(CrossReferenceGallery)
    CrossReferenceGallery.show()
    sys.exit(app.exec_())

