# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'PreprocessTool.ui'
#
# Created: Mon Sep 19 20:38:16 2016
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

class Ui_PreprocessGui(object):
    def setupUi(self, PreprocessGui):
        PreprocessGui.setObjectName(_fromUtf8("PreprocessGui"))
        PreprocessGui.resize(1904, 1088)
        self.centralwidget = QtGui.QWidget(PreprocessGui)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.verticalLayout_3 = QtGui.QVBoxLayout(self.centralwidget)
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.horizontalLayout_9 = QtGui.QHBoxLayout()
        self.horizontalLayout_9.setObjectName(_fromUtf8("horizontalLayout_9"))
        self.verticalLayout_2 = QtGui.QVBoxLayout()
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.slide_gview = QtGui.QGraphicsView(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.slide_gview.sizePolicy().hasHeightForWidth())
        self.slide_gview.setSizePolicy(sizePolicy)
        self.slide_gview.setMinimumSize(QtCore.QSize(1200, 500))
        self.slide_gview.setMaximumSize(QtCore.QSize(1200, 500))
        self.slide_gview.setObjectName(_fromUtf8("slide_gview"))
        self.verticalLayout_2.addWidget(self.slide_gview)
        self.horizontalLayout_7 = QtGui.QHBoxLayout()
        self.horizontalLayout_7.setObjectName(_fromUtf8("horizontalLayout_7"))
        self.button_download = QtGui.QPushButton(self.centralwidget)
        self.button_download.setObjectName(_fromUtf8("button_download"))
        self.horizontalLayout_7.addWidget(self.button_download)
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout_7.addWidget(self.label)
        self.comboBox_slide_position_adjustment = QtGui.QComboBox(self.centralwidget)
        self.comboBox_slide_position_adjustment.setObjectName(_fromUtf8("comboBox_slide_position_adjustment"))
        self.comboBox_slide_position_adjustment.addItem(_fromUtf8(""))
        self.comboBox_slide_position_adjustment.addItem(_fromUtf8(""))
        self.comboBox_slide_position_adjustment.addItem(_fromUtf8(""))
        self.comboBox_slide_position_adjustment.addItem(_fromUtf8(""))
        self.comboBox_slide_position_adjustment.addItem(_fromUtf8(""))
        self.comboBox_slide_position_adjustment.addItem(_fromUtf8(""))
        self.comboBox_slide_position_adjustment.addItem(_fromUtf8(""))
        self.comboBox_slide_position_adjustment.addItem(_fromUtf8(""))
        self.comboBox_slide_position_adjustment.addItem(_fromUtf8(""))
        self.comboBox_slide_position_adjustment.addItem(_fromUtf8(""))
        self.comboBox_slide_position_adjustment.addItem(_fromUtf8(""))
        self.comboBox_slide_position_adjustment.addItem(_fromUtf8(""))
        self.horizontalLayout_7.addWidget(self.comboBox_slide_position_adjustment)
        self.verticalLayout_2.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_9.addLayout(self.verticalLayout_2)
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
        self.gridLayout_2.addWidget(self.button_edit_transform, 0, 1, 1, 1)
        self.button_crop = QtGui.QPushButton(self.centralwidget)
        self.button_crop.setObjectName(_fromUtf8("button_crop"))
        self.gridLayout_2.addWidget(self.button_crop, 2, 3, 1, 1)
        self.button_load_crop = QtGui.QPushButton(self.centralwidget)
        self.button_load_crop.setObjectName(_fromUtf8("button_load_crop"))
        self.gridLayout_2.addWidget(self.button_load_crop, 2, 1, 1, 1)
        self.comboBox_show = QtGui.QComboBox(self.centralwidget)
        self.comboBox_show.setObjectName(_fromUtf8("comboBox_show"))
        self.comboBox_show.addItem(_fromUtf8(""))
        self.comboBox_show.addItem(_fromUtf8(""))
        self.comboBox_show.addItem(_fromUtf8(""))
        self.comboBox_show.addItem(_fromUtf8(""))
        self.comboBox_show.addItem(_fromUtf8(""))
        self.gridLayout_2.addWidget(self.comboBox_show, 2, 0, 1, 1)
        self.button_save_crop = QtGui.QPushButton(self.centralwidget)
        self.button_save_crop.setObjectName(_fromUtf8("button_save_crop"))
        self.gridLayout_2.addWidget(self.button_save_crop, 2, 2, 1, 1)
        self.button_confirm_alignment = QtGui.QPushButton(self.centralwidget)
        self.button_confirm_alignment.setObjectName(_fromUtf8("button_confirm_alignment"))
        self.gridLayout_2.addWidget(self.button_confirm_alignment, 0, 2, 1, 1)
        self.button_align = QtGui.QPushButton(self.centralwidget)
        self.button_align.setObjectName(_fromUtf8("button_align"))
        self.gridLayout_2.addWidget(self.button_align, 0, 0, 1, 1)
        self.button_mask = QtGui.QPushButton(self.centralwidget)
        self.button_mask.setObjectName(_fromUtf8("button_mask"))
        self.gridLayout_2.addWidget(self.button_mask, 3, 0, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout_2)
        self.horizontalLayout_9.addLayout(self.verticalLayout)
        self.verticalLayout_3.addLayout(self.horizontalLayout_9)
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.section2_gview = QtGui.QGraphicsView(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.section2_gview.sizePolicy().hasHeightForWidth())
        self.section2_gview.setSizePolicy(sizePolicy)
        self.section2_gview.setObjectName(_fromUtf8("section2_gview"))
        self.gridLayout.addWidget(self.section2_gview, 0, 1, 1, 1)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.label_section2_filename = QtGui.QLabel(self.centralwidget)
        self.label_section2_filename.setObjectName(_fromUtf8("label_section2_filename"))
        self.horizontalLayout_3.addWidget(self.label_section2_filename)
        self.label_section2_index = QtGui.QLabel(self.centralwidget)
        self.label_section2_index.setObjectName(_fromUtf8("label_section2_index"))
        self.horizontalLayout_3.addWidget(self.label_section2_index)
        self.gridLayout.addLayout(self.horizontalLayout_3, 1, 1, 1, 1)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.label_section3_filename = QtGui.QLabel(self.centralwidget)
        self.label_section3_filename.setObjectName(_fromUtf8("label_section3_filename"))
        self.horizontalLayout.addWidget(self.label_section3_filename)
        self.label_section3_index = QtGui.QLabel(self.centralwidget)
        self.label_section3_index.setObjectName(_fromUtf8("label_section3_index"))
        self.horizontalLayout.addWidget(self.label_section3_index)
        self.gridLayout.addLayout(self.horizontalLayout, 1, 0, 1, 1)
        self.section1_gview = QtGui.QGraphicsView(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.section1_gview.sizePolicy().hasHeightForWidth())
        self.section1_gview.setSizePolicy(sizePolicy)
        self.section1_gview.setObjectName(_fromUtf8("section1_gview"))
        self.gridLayout.addWidget(self.section1_gview, 0, 2, 1, 1)
        self.horizontalLayout_4 = QtGui.QHBoxLayout()
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.label_section1_filename = QtGui.QLabel(self.centralwidget)
        self.label_section1_filename.setObjectName(_fromUtf8("label_section1_filename"))
        self.horizontalLayout_4.addWidget(self.label_section1_filename)
        self.label_section1_index = QtGui.QLabel(self.centralwidget)
        self.label_section1_index.setObjectName(_fromUtf8("label_section1_index"))
        self.horizontalLayout_4.addWidget(self.label_section1_index)
        self.gridLayout.addLayout(self.horizontalLayout_4, 1, 2, 1, 1)
        self.section3_gview = QtGui.QGraphicsView(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.section3_gview.sizePolicy().hasHeightForWidth())
        self.section3_gview.setSizePolicy(sizePolicy)
        self.section3_gview.setObjectName(_fromUtf8("section3_gview"))
        self.gridLayout.addWidget(self.section3_gview, 0, 0, 1, 1)
        self.verticalLayout_3.addLayout(self.gridLayout)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.button_sort = QtGui.QPushButton(self.centralwidget)
        self.button_sort.setObjectName(_fromUtf8("button_sort"))
        self.horizontalLayout_2.addWidget(self.button_sort)
        self.button_save_sorted_filenames = QtGui.QPushButton(self.centralwidget)
        self.button_save_sorted_filenames.setObjectName(_fromUtf8("button_save_sorted_filenames"))
        self.horizontalLayout_2.addWidget(self.button_save_sorted_filenames)
        self.button_load_sorted_filenames = QtGui.QPushButton(self.centralwidget)
        self.button_load_sorted_filenames.setObjectName(_fromUtf8("button_load_sorted_filenames"))
        self.horizontalLayout_2.addWidget(self.button_load_sorted_filenames)
        self.button_confirm_order = QtGui.QPushButton(self.centralwidget)
        self.button_confirm_order.setObjectName(_fromUtf8("button_confirm_order"))
        self.horizontalLayout_2.addWidget(self.button_confirm_order)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_5 = QtGui.QHBoxLayout()
        self.horizontalLayout_5.setObjectName(_fromUtf8("horizontalLayout_5"))
        self.button_save_slide_position_map = QtGui.QPushButton(self.centralwidget)
        self.button_save_slide_position_map.setObjectName(_fromUtf8("button_save_slide_position_map"))
        self.horizontalLayout_5.addWidget(self.button_save_slide_position_map)
        self.button_load_slide_position_map = QtGui.QPushButton(self.centralwidget)
        self.button_load_slide_position_map.setObjectName(_fromUtf8("button_load_slide_position_map"))
        self.horizontalLayout_5.addWidget(self.button_load_slide_position_map)
        self.verticalLayout_3.addLayout(self.horizontalLayout_5)
        PreprocessGui.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(PreprocessGui)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1904, 25))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        PreprocessGui.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(PreprocessGui)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        PreprocessGui.setStatusBar(self.statusbar)

        self.retranslateUi(PreprocessGui)
        QtCore.QMetaObject.connectSlotsByName(PreprocessGui)

    def retranslateUi(self, PreprocessGui):
        PreprocessGui.setWindowTitle(_translate("PreprocessGui", "MainWindow", None))
        self.button_download.setText(_translate("PreprocessGui", "Download Macros and Thumbnails", None))
        self.label.setText(_translate("PreprocessGui", "Adjustment", None))
        self.comboBox_slide_position_adjustment.setItemText(0, _translate("PreprocessGui", "Move all nonexisting to left", None))
        self.comboBox_slide_position_adjustment.setItemText(1, _translate("PreprocessGui", "Move all nonexisting to right", None))
        self.comboBox_slide_position_adjustment.setItemText(2, _translate("PreprocessGui", "Move placeholder to right", None))
        self.comboBox_slide_position_adjustment.setItemText(3, _translate("PreprocessGui", "Move rescan to right", None))
        self.comboBox_slide_position_adjustment.setItemText(4, _translate("PreprocessGui", "Move nonexisting to right", None))
        self.comboBox_slide_position_adjustment.setItemText(5, _translate("PreprocessGui", "Move placeholder to left", None))
        self.comboBox_slide_position_adjustment.setItemText(6, _translate("PreprocessGui", "Move rescan to left", None))
        self.comboBox_slide_position_adjustment.setItemText(7, _translate("PreprocessGui", "Move nonexisting to left", None))
        self.comboBox_slide_position_adjustment.setItemText(8, _translate("PreprocessGui", "Reverse positions", None))
        self.comboBox_slide_position_adjustment.setItemText(9, _translate("PreprocessGui", "Reverse positions on all slides", None))
        self.comboBox_slide_position_adjustment.setItemText(10, _translate("PreprocessGui", "Reverse positions on all IHC slides", None))
        self.comboBox_slide_position_adjustment.setItemText(11, _translate("PreprocessGui", "Reverse positions on all N slides", None))
        self.label_sorted_sections_status.setText(_translate("PreprocessGui", "TextLabel", None))
        self.label_sorted_sections_filename.setText(_translate("PreprocessGui", "TextLabel", None))
        self.label_sorted_sections_index.setText(_translate("PreprocessGui", "TextLabel", None))
        self.button_edit_transform.setText(_translate("PreprocessGui", "Edit Transform", None))
        self.button_crop.setText(_translate("PreprocessGui", "Crop", None))
        self.button_load_crop.setText(_translate("PreprocessGui", "Load Crop", None))
        self.comboBox_show.setItemText(0, _translate("PreprocessGui", "Original", None))
        self.comboBox_show.setItemText(1, _translate("PreprocessGui", "Original Aligned", None))
        self.comboBox_show.setItemText(2, _translate("PreprocessGui", "Brainstem Cropped", None))
        self.comboBox_show.setItemText(3, _translate("PreprocessGui", "Mask", None))
        self.comboBox_show.setItemText(4, _translate("PreprocessGui", "Brainstem Cropped Masked", None))
        self.button_save_crop.setText(_translate("PreprocessGui", "Save Crop", None))
        self.button_confirm_alignment.setText(_translate("PreprocessGui", "Compose", None))
        self.button_align.setText(_translate("PreprocessGui", "Align", None))
        self.button_mask.setText(_translate("PreprocessGui", "Generate/Warp/Crop Mask", None))
        self.label_section2_filename.setText(_translate("PreprocessGui", "File", None))
        self.label_section2_index.setText(_translate("PreprocessGui", "TextLabel", None))
        self.label_section3_filename.setText(_translate("PreprocessGui", "File", None))
        self.label_section3_index.setText(_translate("PreprocessGui", "TextLabel", None))
        self.label_section1_filename.setText(_translate("PreprocessGui", "File", None))
        self.label_section1_index.setText(_translate("PreprocessGui", "TextLabel", None))
        self.button_sort.setText(_translate("PreprocessGui", "Sort Sections", None))
        self.button_save_sorted_filenames.setText(_translate("PreprocessGui", "Save Sorted Filenames", None))
        self.button_load_sorted_filenames.setText(_translate("PreprocessGui", "Load Sorted Filenames", None))
        self.button_confirm_order.setText(_translate("PreprocessGui", "Confirm Order", None))
        self.button_save_slide_position_map.setText(_translate("PreprocessGui", "Save Slide Position -> Filename", None))
        self.button_load_slide_position_map.setText(_translate("PreprocessGui", "Load Slide Position -> Filename", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    PreprocessGui = QtGui.QMainWindow()
    ui = Ui_PreprocessGui()
    ui.setupUi(PreprocessGui)
    PreprocessGui.show()
    sys.exit(app.exec_())

