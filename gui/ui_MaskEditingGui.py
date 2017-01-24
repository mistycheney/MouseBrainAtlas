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
        self.gridLayout = QtGui.QGridLayout(MaskEditingGui)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.splitter = QtGui.QSplitter(MaskEditingGui)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName(_fromUtf8("splitter"))
        self.layoutWidget = QtGui.QWidget(self.splitter)
        self.layoutWidget.setObjectName(_fromUtf8("layoutWidget"))
        self.verticalLayout = QtGui.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.gview_1 = QtGui.QGraphicsView(self.layoutWidget)
        self.gview_1.setObjectName(_fromUtf8("gview_1"))
        self.verticalLayout.addWidget(self.gview_1)
        self.button_good_1 = QtGui.QPushButton(self.layoutWidget)
        self.button_good_1.setText(_fromUtf8(""))
        self.button_good_1.setObjectName(_fromUtf8("button_good_1"))
        self.verticalLayout.addWidget(self.button_good_1)
        self.layoutWidget1 = QtGui.QWidget(self.splitter)
        self.layoutWidget1.setObjectName(_fromUtf8("layoutWidget1"))
        self.verticalLayout_3 = QtGui.QVBoxLayout(self.layoutWidget1)
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.gview_2 = QtGui.QGraphicsView(self.layoutWidget1)
        self.gview_2.setObjectName(_fromUtf8("gview_2"))
        self.verticalLayout_3.addWidget(self.gview_2)
        self.button_good_2 = QtGui.QPushButton(self.layoutWidget1)
        self.button_good_2.setText(_fromUtf8(""))
        self.button_good_2.setObjectName(_fromUtf8("button_good_2"))
        self.verticalLayout_3.addWidget(self.button_good_2)
        self.layoutWidget2 = QtGui.QWidget(self.splitter)
        self.layoutWidget2.setObjectName(_fromUtf8("layoutWidget2"))
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.layoutWidget2)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.gview_3 = QtGui.QGraphicsView(self.layoutWidget2)
        self.gview_3.setObjectName(_fromUtf8("gview_3"))
        self.verticalLayout_2.addWidget(self.gview_3)
        self.button_good_3 = QtGui.QPushButton(self.layoutWidget2)
        self.button_good_3.setText(_fromUtf8(""))
        self.button_good_3.setObjectName(_fromUtf8("button_good_3"))
        self.verticalLayout_2.addWidget(self.button_good_3)
        self.gridLayout.addWidget(self.splitter, 0, 0, 1, 1)
        self.splitter_2 = QtGui.QSplitter(MaskEditingGui)
        self.splitter_2.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_2.setObjectName(_fromUtf8("splitter_2"))
        self.layoutWidget3 = QtGui.QWidget(self.splitter_2)
        self.layoutWidget3.setObjectName(_fromUtf8("layoutWidget3"))
        self.verticalLayout_4 = QtGui.QVBoxLayout(self.layoutWidget3)
        self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
        self.gview_4 = QtGui.QGraphicsView(self.layoutWidget3)
        self.gview_4.setObjectName(_fromUtf8("gview_4"))
        self.verticalLayout_4.addWidget(self.gview_4)
        self.button_good_4 = QtGui.QPushButton(self.layoutWidget3)
        self.button_good_4.setText(_fromUtf8(""))
        self.button_good_4.setObjectName(_fromUtf8("button_good_4"))
        self.verticalLayout_4.addWidget(self.button_good_4)
        self.layoutWidget4 = QtGui.QWidget(self.splitter_2)
        self.layoutWidget4.setObjectName(_fromUtf8("layoutWidget4"))
        self.verticalLayout_5 = QtGui.QVBoxLayout(self.layoutWidget4)
        self.verticalLayout_5.setObjectName(_fromUtf8("verticalLayout_5"))
        self.gview_5 = QtGui.QGraphicsView(self.layoutWidget4)
        self.gview_5.setObjectName(_fromUtf8("gview_5"))
        self.verticalLayout_5.addWidget(self.gview_5)
        self.button_good_5 = QtGui.QPushButton(self.layoutWidget4)
        self.button_good_5.setText(_fromUtf8(""))
        self.button_good_5.setObjectName(_fromUtf8("button_good_5"))
        self.verticalLayout_5.addWidget(self.button_good_5)
        self.layoutWidget5 = QtGui.QWidget(self.splitter_2)
        self.layoutWidget5.setObjectName(_fromUtf8("layoutWidget5"))
        self.verticalLayout_6 = QtGui.QVBoxLayout(self.layoutWidget5)
        self.verticalLayout_6.setObjectName(_fromUtf8("verticalLayout_6"))
        self.listview_bad = QtGui.QListView(self.layoutWidget5)
        self.listview_bad.setObjectName(_fromUtf8("listview_bad"))
        self.verticalLayout_6.addWidget(self.listview_bad)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.button_addToList = QtGui.QPushButton(self.layoutWidget5)
        self.button_addToList.setObjectName(_fromUtf8("button_addToList"))
        self.horizontalLayout.addWidget(self.button_addToList)
        self.button_removeFromList = QtGui.QPushButton(self.layoutWidget5)
        self.button_removeFromList.setObjectName(_fromUtf8("button_removeFromList"))
        self.horizontalLayout.addWidget(self.button_removeFromList)
        self.button_redoList = QtGui.QPushButton(self.layoutWidget5)
        self.button_redoList.setObjectName(_fromUtf8("button_redoList"))
        self.horizontalLayout.addWidget(self.button_redoList)
        self.verticalLayout_6.addLayout(self.horizontalLayout)
        self.gridLayout.addWidget(self.splitter_2, 1, 0, 1, 1)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.button_confirmMasks = QtGui.QPushButton(MaskEditingGui)
        self.button_confirmMasks.setObjectName(_fromUtf8("button_confirmMasks"))
        self.horizontalLayout_2.addWidget(self.button_confirmMasks)
        self.button_closeMaskGui = QtGui.QPushButton(MaskEditingGui)
        self.button_closeMaskGui.setObjectName(_fromUtf8("button_closeMaskGui"))
        self.horizontalLayout_2.addWidget(self.button_closeMaskGui)
        self.gridLayout.addLayout(self.horizontalLayout_2, 2, 0, 1, 1)

        self.retranslateUi(MaskEditingGui)
        QtCore.QMetaObject.connectSlotsByName(MaskEditingGui)

    def retranslateUi(self, MaskEditingGui):
        MaskEditingGui.setWindowTitle(_translate("MaskEditingGui", "Review Masks ", None))
        self.button_addToList.setText(_translate("MaskEditingGui", "Add to List", None))
        self.button_removeFromList.setText(_translate("MaskEditingGui", "Remove from List", None))
        self.button_redoList.setText(_translate("MaskEditingGui", "Redo for List", None))
        self.button_confirmMasks.setText(_translate("MaskEditingGui", "Confirm", None))
        self.button_closeMaskGui.setText(_translate("MaskEditingGui", "Close", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    MaskEditingGui = QtGui.QDialog()
    ui = Ui_MaskEditingGui()
    ui.setupUi(MaskEditingGui)
    MaskEditingGui.show()
    sys.exit(app.exec_())

