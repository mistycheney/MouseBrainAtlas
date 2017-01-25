# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MaskEditingGui2.ui'
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

class Ui_MaskEditingGui2(object):
    def setupUi(self, MaskEditingGui2):
        MaskEditingGui2.setObjectName(_fromUtf8("MaskEditingGui2"))
        MaskEditingGui2.resize(1046, 825)
        self.gridLayout_2 = QtGui.QGridLayout(MaskEditingGui2)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.splitter_2 = QtGui.QSplitter(MaskEditingGui2)
        self.splitter_2.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_2.setObjectName(_fromUtf8("splitter_2"))
        self.gview_finalMask = QtGui.QGraphicsView(self.splitter_2)
        self.gview_finalMask.setObjectName(_fromUtf8("gview_finalMask"))
        self.splitter = QtGui.QSplitter(self.splitter_2)
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setObjectName(_fromUtf8("splitter"))
        self.listview_bad = QtGui.QListView(self.splitter)
        self.listview_bad.setObjectName(_fromUtf8("listview_bad"))
        self.widget = QtGui.QWidget(self.splitter)
        self.widget.setObjectName(_fromUtf8("widget"))
        self.gridLayout = QtGui.QGridLayout(self.widget)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.button_drawInitContour = QtGui.QPushButton(self.widget)
        self.button_drawInitContour.setObjectName(_fromUtf8("button_drawInitContour"))
        self.gridLayout.addWidget(self.button_drawInitContour, 1, 1, 1, 1)
        self.button_redoThisSection = QtGui.QPushButton(self.widget)
        self.button_redoThisSection.setObjectName(_fromUtf8("button_redoThisSection"))
        self.gridLayout.addWidget(self.button_redoThisSection, 1, 2, 1, 1)
        self.button_closeMaskGui = QtGui.QPushButton(self.widget)
        self.button_closeMaskGui.setObjectName(_fromUtf8("button_closeMaskGui"))
        self.gridLayout.addWidget(self.button_closeMaskGui, 2, 2, 1, 1)
        self.button_acceptAllSubmasks = QtGui.QPushButton(self.widget)
        self.button_acceptAllSubmasks.setObjectName(_fromUtf8("button_acceptAllSubmasks"))
        self.gridLayout.addWidget(self.button_acceptAllSubmasks, 1, 0, 1, 1)
        self.button_saveReview = QtGui.QPushButton(self.widget)
        self.button_saveReview.setObjectName(_fromUtf8("button_saveReview"))
        self.gridLayout.addWidget(self.button_saveReview, 2, 1, 1, 1)
        self.button_loadReview = QtGui.QPushButton(self.widget)
        self.button_loadReview.setObjectName(_fromUtf8("button_loadReview"))
        self.gridLayout.addWidget(self.button_loadReview, 2, 0, 1, 1)
        self.button_removeFromList = QtGui.QPushButton(self.widget)
        self.button_removeFromList.setObjectName(_fromUtf8("button_removeFromList"))
        self.gridLayout.addWidget(self.button_removeFromList, 0, 1, 1, 1)
        self.button_redoList = QtGui.QPushButton(self.widget)
        self.button_redoList.setObjectName(_fromUtf8("button_redoList"))
        self.gridLayout.addWidget(self.button_redoList, 0, 2, 1, 1)
        self.button_addToList = QtGui.QPushButton(self.widget)
        self.button_addToList.setObjectName(_fromUtf8("button_addToList"))
        self.gridLayout.addWidget(self.button_addToList, 0, 0, 1, 1)
        self.button_confirmMasks = QtGui.QPushButton(self.widget)
        self.button_confirmMasks.setObjectName(_fromUtf8("button_confirmMasks"))
        self.gridLayout.addWidget(self.button_confirmMasks, 3, 0, 1, 1)
        self.button_uploadMasks = QtGui.QPushButton(self.widget)
        self.button_uploadMasks.setObjectName(_fromUtf8("button_uploadMasks"))
        self.gridLayout.addWidget(self.button_uploadMasks, 3, 1, 1, 1)
        self.gridLayout_2.addWidget(self.splitter_2, 0, 0, 1, 1)

        self.retranslateUi(MaskEditingGui2)
        QtCore.QMetaObject.connectSlotsByName(MaskEditingGui2)

    def retranslateUi(self, MaskEditingGui2):
        MaskEditingGui2.setWindowTitle(_translate("MaskEditingGui2", "Review Masks", None))
        self.button_drawInitContour.setText(_translate("MaskEditingGui2", "Draw Initial Contour", None))
        self.button_redoThisSection.setText(_translate("MaskEditingGui2", "Redo this Section", None))
        self.button_closeMaskGui.setText(_translate("MaskEditingGui2", "Close", None))
        self.button_acceptAllSubmasks.setText(_translate("MaskEditingGui2", "Accept All", None))
        self.button_saveReview.setText(_translate("MaskEditingGui2", "Save Review", None))
        self.button_loadReview.setText(_translate("MaskEditingGui2", "Load Review", None))
        self.button_removeFromList.setText(_translate("MaskEditingGui2", "Remove from List", None))
        self.button_redoList.setText(_translate("MaskEditingGui2", "Redo for List", None))
        self.button_addToList.setText(_translate("MaskEditingGui2", "Add to List", None))
        self.button_confirmMasks.setText(_translate("MaskEditingGui2", "Confirm", None))
        self.button_uploadMasks.setText(_translate("MaskEditingGui2", "Upload", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    MaskEditingGui2 = QtGui.QDialog()
    ui = Ui_MaskEditingGui2()
    ui.setupUi(MaskEditingGui2)
    MaskEditingGui2.show()
    sys.exit(app.exec_())

