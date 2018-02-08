#! /usr/bin/env python

# import sip
# sip.setapi('QVariant', 2) # http://stackoverflow.com/questions/21217399/pyqt4-qtcore-qvariant-object-instead-of-a-string

import sys
import os
import numpy as np

from PyQt4.QtCore import *
from PyQt4.QtGui import *

# from matplotlib.backends import qt4_compat
# use_pyside = qt4_compat.QT_API == qt4_compat.QT_API_PYSIDE
# if use_pyside:
#     #print 'Using PySide'
#     from PySide.QtCore import *
#     from PySide.QtGui import *
# else:
#     #print 'Using PyQt4'
#     from PyQt4.QtCore import *
#     from PyQt4.QtGui import *


class CustomQCompleter(QCompleter):
    # adapted from http://stackoverflow.com/a/26440173
    def __init__(self, *args):#parent=None):
        super(CustomQCompleter, self).__init__(*args)
        self.local_completion_prefix = ""
        self.source_model = None
        self.filterProxyModel = QSortFilterProxyModel(self)
        self.usingOriginalModel = False

    def setModel(self, model):
        self.source_model = model
        self.filterProxyModel = QSortFilterProxyModel(self)
        self.filterProxyModel.setSourceModel(self.source_model)
        super(CustomQCompleter, self).setModel(self.filterProxyModel)
        self.usingOriginalModel = True

    def updateModel(self):
        if not self.usingOriginalModel:
            self.filterProxyModel.setSourceModel(self.source_model)

        pattern = QRegExp(self.local_completion_prefix,
                                Qt.CaseInsensitive,
                                QRegExp.FixedString)

        self.filterProxyModel.setFilterRegExp(pattern)

    def splitPath(self, path):
        self.local_completion_prefix = path
        self.updateModel()
        if self.filterProxyModel.rowCount() == 0:
            self.usingOriginalModel = False
            self.filterProxyModel.setSourceModel(QStringListModel([path]))
            return [path]

        return []

class AutoCompleteComboBox(QComboBox):
    # adapted from http://stackoverflow.com/a/26440173
    def __init__(self, labels, *args, **kwargs):
        super(AutoCompleteComboBox, self).__init__(*args, **kwargs)

        self.setEditable(True)
        self.setInsertPolicy(self.NoInsert)

        self.comp = CustomQCompleter(self)
        self.comp.setCompletionMode(QCompleter.PopupCompletion)
        self.setCompleter(self.comp)#
        self.setModel(labels)

        self.clearEditText()

    def setModel(self, strList):
        self.clear()
        self.insertItems(0, strList)
        self.comp.setModel(self.model())

    def focusInEvent(self, event):
        # self.clearEditText()
        super(AutoCompleteComboBox, self).focusInEvent(event)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Return:

            # make sure that the completer does not set the
            # currentText of the combobox to "" when pressing enter
            text = self.currentText()
            self.setCompleter(None)
            self.setEditText(text)
            self.setCompleter(self.comp)

        return super(AutoCompleteComboBox, self).keyPressEvent(event)

class AutoCompleteInputDialog(QDialog):

    def __init__(self, labels, *args, **kwargs):
        super(AutoCompleteInputDialog, self).__init__(*args, **kwargs)
        self.comboBox = AutoCompleteComboBox(parent=self, labels=labels)
        va = QVBoxLayout(self)
        va.addWidget(self.comboBox)
        box = QWidget(self)
        ha = QHBoxLayout(self)
        va.addWidget(box)
        box.setLayout(ha)
        self.OK = QPushButton("OK", self)
        self.OK.setDefault(True)
        # cancel = QPushButton("Cancel", self)
        ha.addWidget(self.OK)
        # ha.addWidget(cancel)

    def set_test_callback(self, callback):
        self.OK.clicked.connect(callback)
        # OK.clicked.connect(self.accept)
        # cancel.clicked.connect(self.reject)

class ListSelection(QDialog):
    """
    https://stackoverflow.com/questions/41310023/pyside-popup-showing-list-and-multiple-select
    """
    def __init__(self, title, message, items, items_checked, parent=None):
        """
        Args:
            checked (list of str)
        """
        super(ListSelection, self).__init__(parent=parent)
        form = QFormLayout(self)
        form.addRow(QLabel(message))
        self.listView = QListView(self)
        form.addRow(self.listView)
        model = QStandardItemModel(self.listView)
        self.setWindowTitle(title)
        for item in items:
            # create an item with a caption
            standardItem = QStandardItem(item)
            standardItem.setCheckable(True)
            if item in items_checked:
                standardItem.setCheckState(True)
            model.appendRow(standardItem)
        self.listView.setModel(model)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        form.addRow(buttonBox)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def itemsSelected(self):
        selected = []
        model = self.listView.model()
        i = 0
        while model.item(i):
            if model.item(i).checkState():
                selected.append(model.item(i).text())
            i += 1
        return selected


# class ListSelection(QDialog):
#     def __init__(self, item_ls, parent=None):
#         super(ListSelection, self).__init__(parent)
#
#         self.setWindowTitle('Detect which landmarks ?')
#
#         self.selected = set([])
#
#         self.listWidget = QListWidget()
#         for item in item_ls:
#             w_item = QListWidgetItem(item)
#             self.listWidget.addItem(w_item)
#
#             w_item.setFlags(w_item.flags() | Qt.ItemIsUserCheckable)
#             w_item.setCheckState(False)
#
#         self.listWidget.itemChanged.connect(self.OnSingleClick)
#
#         layout = QGridLayout()
#         layout.addWidget(self.listWidget,0,0,1,3)
#
#         self.but_ok = QPushButton("OK")
#         layout.addWidget(self.but_ok ,1,1)
#         self.but_ok.clicked.connect(self.OnOk)
#
#         self.but_cancel = QPushButton("Cancel")
#         layout.addWidget(self.but_cancel ,1,2)
#         self.but_cancel.clicked.connect(self.OnCancel)
#
#         self.setLayout(layout)
#         self.setGeometry(300, 200, 460, 350)
#
#     def OnSingleClick(self, item):
#         if not item.checkState():
#         #   item.setCheckState(False)
#             self.selected = self.selected - {str(item.text())}
#         #   print self.selected
#         else:
#         #   item.setCheckState(True)
#             self.selected.add(str(item.text()))
#
#         print self.selected
#
#     def OnOk(self):
#         self.close()
#
#     def OnCancel(self):
#         self.selected = set([])
#         self.close()
