# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/PreprocessTool_v2.ui'
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

class Ui_PreprocessGui(object):
    def setupUi(self, PreprocessGui):
        PreprocessGui.setObjectName(_fromUtf8("PreprocessGui"))
        PreprocessGui.resize(1906, 1088)
        self.centralwidget = QtGui.QWidget(PreprocessGui)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.verticalLayout_3 = QtGui.QVBoxLayout(self.centralwidget)
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.horizontalLayout_9 = QtGui.QHBoxLayout()
        self.horizontalLayout_9.setObjectName(_fromUtf8("horizontalLayout_9"))
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout_8 = QtGui.QHBoxLayout()
        self.horizontalLayout_8.setObjectName(_fromUtf8("horizontalLayout_8"))
        self.sorted_sections_gview = QtGui.QGraphicsView(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sorted_sections_gview.sizePolicy().hasHeightForWidth())
        self.sorted_sections_gview.setSizePolicy(sizePolicy)
        self.sorted_sections_gview.setMinimumSize(QtCore.QSize(600, 500))
        self.sorted_sections_gview.setObjectName(_fromUtf8("sorted_sections_gview"))
        self.horizontalLayout_8.addWidget(self.sorted_sections_gview)
        self.label_sorted_sections_status = QtGui.QLabel(self.centralwidget)
        self.label_sorted_sections_status.setObjectName(_fromUtf8("label_sorted_sections_status"))
        self.horizontalLayout_8.addWidget(self.label_sorted_sections_status)
        self.verticalLayout.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_6 = QtGui.QHBoxLayout()
        self.horizontalLayout_6.setObjectName(_fromUtf8("horizontalLayout_6"))
        self.label_sorted_sections_filename = QtGui.QLabel(self.centralwidget)
        self.label_sorted_sections_filename.setObjectName(_fromUtf8("label_sorted_sections_filename"))
        self.horizontalLayout_6.addWidget(self.label_sorted_sections_filename)
        self.label_sorted_sections_index = QtGui.QLabel(self.centralwidget)
        self.label_sorted_sections_index.setObjectName(_fromUtf8("label_sorted_sections_index"))
        self.horizontalLayout_6.addWidget(self.label_sorted_sections_index)
        self.verticalLayout.addLayout(self.horizontalLayout_6)
        self.gridLayout_2 = QtGui.QGridLayout()
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.button_edit_transform = QtGui.QPushButton(self.centralwidget)
        self.button_edit_transform.setObjectName(_fromUtf8("button_edit_transform"))
        self.gridLayout_2.addWidget(self.button_edit_transform, 3, 4, 1, 1)
        self.button_load_crop = QtGui.QPushButton(self.centralwidget)
        self.button_load_crop.setObjectName(_fromUtf8("button_load_crop"))
        self.gridLayout_2.addWidget(self.button_load_crop, 3, 1, 1, 1)
        self.button_save_crop = QtGui.QPushButton(self.centralwidget)
        self.button_save_crop.setObjectName(_fromUtf8("button_save_crop"))
        self.gridLayout_2.addWidget(self.button_save_crop, 3, 3, 1, 1)
        self.comboBox_show = QtGui.QComboBox(self.centralwidget)
        self.comboBox_show.setObjectName(_fromUtf8("comboBox_show"))
        self.comboBox_show.addItem(_fromUtf8(""))
        self.comboBox_show.addItem(_fromUtf8(""))
        self.comboBox_show.addItem(_fromUtf8(""))
        self.comboBox_show.addItem(_fromUtf8(""))
        self.comboBox_show.addItem(_fromUtf8(""))
        self.gridLayout_2.addWidget(self.comboBox_show, 2, 1, 1, 1)
        self.button_update_order = QtGui.QPushButton(self.centralwidget)
        self.button_update_order.setObjectName(_fromUtf8("button_update_order"))
        self.gridLayout_2.addWidget(self.button_update_order, 2, 4, 1, 1)
        self.button_toggle_show_hide_invalid = QtGui.QPushButton(self.centralwidget)
        self.button_toggle_show_hide_invalid.setObjectName(_fromUtf8("button_toggle_show_hide_invalid"))
        self.gridLayout_2.addWidget(self.button_toggle_show_hide_invalid, 2, 3, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout_2)
        self.horizontalLayout_9.addLayout(self.verticalLayout)
        self.verticalLayout_3.addLayout(self.horizontalLayout_9)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.button_save_sorted_filenames = QtGui.QPushButton(self.centralwidget)
        self.button_save_sorted_filenames.setObjectName(_fromUtf8("button_save_sorted_filenames"))
        self.horizontalLayout_2.addWidget(self.button_save_sorted_filenames)
        self.button_load_sorted_filenames = QtGui.QPushButton(self.centralwidget)
        self.button_load_sorted_filenames.setObjectName(_fromUtf8("button_load_sorted_filenames"))
        self.horizontalLayout_2.addWidget(self.button_load_sorted_filenames)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        PreprocessGui.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(PreprocessGui)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1906, 25))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        PreprocessGui.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(PreprocessGui)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        PreprocessGui.setStatusBar(self.statusbar)

        self.retranslateUi(PreprocessGui)
        QtCore.QMetaObject.connectSlotsByName(PreprocessGui)

    def retranslateUi(self, PreprocessGui):
        PreprocessGui.setWindowTitle(_translate("PreprocessGui", "MainWindow", None))
        self.label_sorted_sections_status.setText(_translate("PreprocessGui", "TextLabel", None))
        self.label_sorted_sections_filename.setText(_translate("PreprocessGui", "Filename", None))
        self.label_sorted_sections_index.setText(_translate("PreprocessGui", "Section Index", None))
        self.button_edit_transform.setText(_translate("PreprocessGui", "Edit Transform", None))
        self.button_load_crop.setText(_translate("PreprocessGui", "Load Crop", None))
        self.button_save_crop.setText(_translate("PreprocessGui", "Save Crop", None))
        self.comboBox_show.setItemText(0, _translate("PreprocessGui", "Original", None))
        self.comboBox_show.setItemText(1, _translate("PreprocessGui", "Original Aligned", None))
        self.comboBox_show.setItemText(2, _translate("PreprocessGui", "Mask Contoured", None))
        self.comboBox_show.setItemText(3, _translate("PreprocessGui", "Brainstem Cropped", None))
        self.comboBox_show.setItemText(4, _translate("PreprocessGui", "Brainstem Cropped Masked", None))
        self.button_update_order.setText(_translate("PreprocessGui", "Update order", None))
        self.button_toggle_show_hide_invalid.setText(_translate("PreprocessGui", "Hide Invalid", None))
        self.button_save_sorted_filenames.setText(_translate("PreprocessGui", "Save Sorted Filenames", None))
        self.button_load_sorted_filenames.setText(_translate("PreprocessGui", "Load Sorted Filenames", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    PreprocessGui = QtGui.QMainWindow()
    ui = Ui_PreprocessGui()
    ui.setupUi(PreprocessGui)
    PreprocessGui.show()
    sys.exit(app.exec_())

