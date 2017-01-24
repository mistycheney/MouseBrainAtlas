# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MaskEditingGui.ui'
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

class Ui_MaskEditingGui(object):
    def setupUi(self, MaskEditingGui):
        MaskEditingGui.setObjectName(_fromUtf8("MaskEditingGui"))
        MaskEditingGui.resize(1050, 827)
        self.gridLayout_2 = QtGui.QGridLayout(MaskEditingGui)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.splitter_2 = QtGui.QSplitter(MaskEditingGui)
        self.splitter_2.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_2.setObjectName(_fromUtf8("splitter_2"))
        self.widget = QtGui.QWidget(self.splitter_2)
        self.widget.setObjectName(_fromUtf8("widget"))
        self.gridLayout = QtGui.QGridLayout(self.widget)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.verticalLayout_3 = QtGui.QVBoxLayout()
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.gview_1 = QtGui.QGraphicsView(self.widget)
        self.gview_1.setObjectName(_fromUtf8("gview_1"))
        self.verticalLayout_3.addWidget(self.gview_1)
        self.button_good_1 = QtGui.QPushButton(self.widget)
        self.button_good_1.setText(_fromUtf8(""))
        self.button_good_1.setObjectName(_fromUtf8("button_good_1"))
        self.verticalLayout_3.addWidget(self.button_good_1)
        self.gridLayout.addLayout(self.verticalLayout_3, 0, 0, 1, 1)
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.gview_2 = QtGui.QGraphicsView(self.widget)
        self.gview_2.setObjectName(_fromUtf8("gview_2"))
        self.verticalLayout.addWidget(self.gview_2)
        self.button_good_2 = QtGui.QPushButton(self.widget)
        self.button_good_2.setText(_fromUtf8(""))
        self.button_good_2.setObjectName(_fromUtf8("button_good_2"))
        self.verticalLayout.addWidget(self.button_good_2)
        self.gridLayout.addLayout(self.verticalLayout, 0, 1, 1, 1)
        self.verticalLayout_4 = QtGui.QVBoxLayout()
        self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
        self.gview_3 = QtGui.QGraphicsView(self.widget)
        self.gview_3.setObjectName(_fromUtf8("gview_3"))
        self.verticalLayout_4.addWidget(self.gview_3)
        self.button_good_3 = QtGui.QPushButton(self.widget)
        self.button_good_3.setText(_fromUtf8(""))
        self.button_good_3.setObjectName(_fromUtf8("button_good_3"))
        self.verticalLayout_4.addWidget(self.button_good_3)
        self.gridLayout.addLayout(self.verticalLayout_4, 1, 0, 1, 1)
        self.verticalLayout_5 = QtGui.QVBoxLayout()
        self.verticalLayout_5.setObjectName(_fromUtf8("verticalLayout_5"))
        self.gview_4 = QtGui.QGraphicsView(self.widget)
        self.gview_4.setObjectName(_fromUtf8("gview_4"))
        self.verticalLayout_5.addWidget(self.gview_4)
        self.button_good_4 = QtGui.QPushButton(self.widget)
        self.button_good_4.setText(_fromUtf8(""))
        self.button_good_4.setObjectName(_fromUtf8("button_good_4"))
        self.verticalLayout_5.addWidget(self.button_good_4)
        self.gridLayout.addLayout(self.verticalLayout_5, 1, 1, 1, 1)
        self.splitter = QtGui.QSplitter(self.splitter_2)
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setObjectName(_fromUtf8("splitter"))
        self.gview_finalMask = QtGui.QGraphicsView(self.splitter)
        self.gview_finalMask.setObjectName(_fromUtf8("gview_finalMask"))
        self.widget1 = QtGui.QWidget(self.splitter)
        self.widget1.setObjectName(_fromUtf8("widget1"))
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.widget1)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.listview_bad = QtGui.QListView(self.widget1)
        self.listview_bad.setObjectName(_fromUtf8("listview_bad"))
        self.verticalLayout_2.addWidget(self.listview_bad)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.button_addToList = QtGui.QPushButton(self.widget1)
        self.button_addToList.setObjectName(_fromUtf8("button_addToList"))
        self.horizontalLayout.addWidget(self.button_addToList)
        self.button_removeFromList = QtGui.QPushButton(self.widget1)
        self.button_removeFromList.setObjectName(_fromUtf8("button_removeFromList"))
        self.horizontalLayout.addWidget(self.button_removeFromList)
        self.button_redoList = QtGui.QPushButton(self.widget1)
        self.button_redoList.setObjectName(_fromUtf8("button_redoList"))
        self.horizontalLayout.addWidget(self.button_redoList)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.gridLayout_2.addWidget(self.splitter_2, 0, 0, 1, 1)
        self.button_acceptAllSubmasks = QtGui.QPushButton(MaskEditingGui)
        self.button_acceptAllSubmasks.setObjectName(_fromUtf8("button_acceptAllSubmasks"))
        self.gridLayout_2.addWidget(self.button_acceptAllSubmasks, 1, 0, 1, 1)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.button_confirmMasks = QtGui.QPushButton(MaskEditingGui)
        self.button_confirmMasks.setObjectName(_fromUtf8("button_confirmMasks"))
        self.horizontalLayout_2.addWidget(self.button_confirmMasks)
        self.button_saveReview = QtGui.QPushButton(MaskEditingGui)
        self.button_saveReview.setObjectName(_fromUtf8("button_saveReview"))
        self.horizontalLayout_2.addWidget(self.button_saveReview)
        self.button_uploadMasks = QtGui.QPushButton(MaskEditingGui)
        self.button_uploadMasks.setObjectName(_fromUtf8("button_uploadMasks"))
        self.horizontalLayout_2.addWidget(self.button_uploadMasks)
        self.button_closeMaskGui = QtGui.QPushButton(MaskEditingGui)
        self.button_closeMaskGui.setObjectName(_fromUtf8("button_closeMaskGui"))
        self.horizontalLayout_2.addWidget(self.button_closeMaskGui)
        self.gridLayout_2.addLayout(self.horizontalLayout_2, 2, 0, 1, 1)
        self.splitter.raise_()
        self.splitter_2.raise_()
        self.button_acceptAllSubmasks.raise_()
        self.gview_finalMask.raise_()

        self.retranslateUi(MaskEditingGui)
        QtCore.QMetaObject.connectSlotsByName(MaskEditingGui)

    def retranslateUi(self, MaskEditingGui):
        MaskEditingGui.setWindowTitle(_translate("MaskEditingGui", "Review Masks ", None))
        self.button_addToList.setText(_translate("MaskEditingGui", "Add to List", None))
        self.button_removeFromList.setText(_translate("MaskEditingGui", "Remove from List", None))
        self.button_redoList.setText(_translate("MaskEditingGui", "Redo for List", None))
        self.button_acceptAllSubmasks.setText(_translate("MaskEditingGui", "Accept All", None))
        self.button_confirmMasks.setText(_translate("MaskEditingGui", "Confirm", None))
        self.button_saveReview.setText(_translate("MaskEditingGui", "Save Review", None))
        self.button_uploadMasks.setText(_translate("MaskEditingGui", "Upload to Gordon", None))
        self.button_closeMaskGui.setText(_translate("MaskEditingGui", "Close", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    MaskEditingGui = QtGui.QDialog()
    ui = Ui_MaskEditingGui()
    ui.setupUi(MaskEditingGui)
    MaskEditingGui.show()
    sys.exit(app.exec_())

