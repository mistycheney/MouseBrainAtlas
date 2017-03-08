# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GalleryDialog.ui'
#
# Created: Sat Sep  3 19:53:13 2016
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

class Ui_gallery_dialog(object):
    def setupUi(self, gallery_dialog):
        gallery_dialog.setObjectName(_fromUtf8("gallery_dialog"))
        gallery_dialog.resize(1007, 728)
        self.gridLayout = QtGui.QGridLayout(gallery_dialog)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.gallery_widget = QtGui.QWidget(gallery_dialog)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.gallery_widget.sizePolicy().hasHeightForWidth())
        self.gallery_widget.setSizePolicy(sizePolicy)
        self.gallery_widget.setObjectName(_fromUtf8("gallery_widget"))
        self.verticalLayout.addWidget(self.gallery_widget)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.button_moveBefore = QtGui.QPushButton(gallery_dialog)
        self.button_moveBefore.setObjectName(_fromUtf8("button_moveBefore"))
        self.horizontalLayout.addWidget(self.button_moveBefore)
        self.button_moveAfter = QtGui.QPushButton(gallery_dialog)
        self.button_moveAfter.setObjectName(_fromUtf8("button_moveAfter"))
        self.horizontalLayout.addWidget(self.button_moveAfter)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        self.sorted_gview = QtGui.QGraphicsView(gallery_dialog)
        self.sorted_gview.setObjectName(_fromUtf8("sorted_gview"))
        self.horizontalLayout_2.addWidget(self.sorted_gview)
        self.gridLayout.addLayout(self.horizontalLayout_2, 0, 0, 1, 1)

        self.retranslateUi(gallery_dialog)
        QtCore.QMetaObject.connectSlotsByName(gallery_dialog)

    def retranslateUi(self, gallery_dialog):
        gallery_dialog.setWindowTitle(_translate("gallery_dialog", "Dialog", None))
        self.button_moveBefore.setText(_translate("gallery_dialog", "Move before", None))
        self.button_moveAfter.setText(_translate("gallery_dialog", "Move after", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    gallery_dialog = QtGui.QDialog()
    ui = Ui_gallery_dialog()
    ui.setupUi(gallery_dialog)
    gallery_dialog.show()
    sys.exit(app.exec_())

