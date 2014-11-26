
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import sys

import subprocess
import os
import json
import param_settings_v2
import string



class MainDialog(QDialog,param_settings_v2.Ui_form):

    def __init__(self,parent=None):
	"""
	Initialization of parameter settings
	"""
        super(MainDialog,self).__init__(parent)
	self.setupUi(self)

	### Save Parameters pushbuttons ###
	self.save_gabor.clicked.connect(self.save_new_gabor)
	self.save_segm.clicked.connect(self.save_new_segm)
	self.save_vq.clicked.connect(self.save_new_vq)
 

	### Item selection from list ###
	self.list_gabor.itemClicked.connect(self.select_gabor)
	self.list_segm.itemClicked.connect(self.select_segm)
	self.list_vq.itemClicked.connect(self.select_vq)

        ### Populate list ###
	self.path_main ="/home/s4myers/brain_registration/param_settings/" #Needs to be changed to directory which holds parameters
	self.path_gabor = os.path.join(self.path_main,"gabor")
	self.path_segm = os.path.join(self.path_main,"segm")
	self.path_vq = os.path.join(self.path_main,"vq")
	
	self.update_gabor_list()
	self.update_segm_list()
	self.update_vq_list()
	
	
    ### Update list widgets ###
    def update_gabor_list(self):
	"""
	Updates the gabor listwidget
	"""
	self.list_gabor.clear()
	files = subprocess.check_output(['ls',self.path_gabor]).split()
        for a_file in files:
            file_name = string.replace(a_file,"_gabor.json","")
	    self.list_gabor.addItem(file_name)

    def update_segm_list(self):
	"""
	Updates the segmentation listwidget
	"""
	self.list_segm.clear()
        files = subprocess.check_output(['ls',self.path_segm]).split()
        for a_file in files:
            file_name = string.replace(a_file,"_segm.json","")
            self.list_segm.addItem(file_name)

    def update_vq_list(self):
	"""
	Updates the vector quantization listwidget
	"""
	self.list_vq.clear()
        files = subprocess.check_output(['ls',self.path_vq]).split()
        for a_file in files:
            file_name = string.replace(a_file,"_vq.json","")	
            self.list_vq.addItem(file_name)
	
	
	
    ### Item Selected from List ###
    def select_gabor(self):
	"""
	Controls the selection of an gabor filter parameters in the list. Selecting a file with 
	the left click will update spin boxes with the appropriate value. A right click will take 
	you to the pop-up menu which allows the ability to rename or delete existing files.
	"""
	item = self.list_gabor.currentItem()
	file_name = str(item.text())+"_gabor.json"
	if self.list_gabor.on_item_clicked()==1:
            path = os.path.join(self.path_gabor,file_name)
	    content = json.load(open(path,'r'))
            self.spinBox_min_wavelen.setValue(content["min_wave"])
            self.spinBox_max_wavelen.setValue(content["max_wave"])
            self.spinBox_freq_step.setValue(content["freq_step"])
            self.spinBox_bandwidth.setValue(content["band"])
            self.spinBox_theta_interval.setValue(content["theta_i"])
	elif self.list_gabor.on_item_clicked()==2:
	    self.pop_up_menu(item,file_name,"gabor")
	else:
	    return
	 
    def select_segm(self):
	""" 
	Controls the selection of an segmentation parameters in the list. Selecting a file with 
	the left click will update spin boxes with the appropriate value. A right click will take 
	you to the pop-up menu which allows the ability to rename or delete existing files.
	"""
	item = self.list_segm.currentItem()
	file_name = str(item.text())+"_segm.json"
	if self.list_segm.on_item_clicked()==1:
            path = os.path.join(self.path_segm,file_name)
            content = json.load(open(path,'r'))
            self.spinBox_n_superpixels.setValue(content["n_sp"])
            self.spinBox_slic_compactness.setValue(content["slic_comp"])
            self.spinBox_slic_sigma.setValue(content["slic_sig"])
	elif self.list_segm.on_item_clicked()==2:
	    self.pop_up_menu(item,file_name,"segm")
	else:
	    return

    def select_vq(self):
	""" 
	Controls the selection of an vector quantization parameters in the list. Selecting a file 
	with the left click will update spin boxes with the appropriate value. A right click will take you 
	to the pop-up menu which allows the ability to rename or delete existing files.
	"""
	item = self.list_vq.currentItem()
	file_name = str(item.text())+"_vq.json"
	if self.list_vq.on_item_clicked()==1:
            path = os.path.join(self.path_vq,file_name)
            content = json.load(open(path,'r'))
            self.spinBox_n_texton.setValue(content["n_texton"])
            self.spinBox_n_sample.setValue(content["n_sample"])
            self.spinBox_n_iter.setValue(content["n_iter"])
	    return
	elif self.list_vq.on_item_clicked()==2:
	    self.pop_up_menu(item,file_name,"vq")
	    return
	else:
	    return

    ### Save Callbacks ###
    def save_new_gabor(self):
	"""
	Callback function for the save gabor filter push button.  Overwrites <name>_gabor.json file 
	if same as one in the list. Otherwise, creates a new file according to the spinbox values.
	"""
	name = str(self.name_gabor.text())
	if  name !="":	
	    file_name =  name+"_gabor.json"
	    content={}
	    content["min_wave"] = self.spinBox_min_wavelen.value()
	    content["max_wave"] = self.spinBox_max_wavelen.value()
	    content["freq_step"] = self.spinBox_freq_step.value()
	    content["band"]=self.spinBox_bandwidth.value()
	    content["theta_i"] = self.spinBox_theta_interval.value()	
	    file_path = os.path.join(self.path_gabor,file_name)
	    json.dump(content,open(os.path.join(file_path),'w'))
	    subprocess.call(['chmod','755',file_path])
	    self.update_gabor_list()
	    self.name_gabor.clear()
	else:
	    return

    def save_new_segm(self):
	"""
	Callback function for the save segmentation push button.  Overwrites <name>_segm.json file
	if same as one in the list. Otherwise, creates a new file according to the spinbox values.
	"""
	name = str(self.name_segm.text())
	if  name !="":
            file_name =  name+"_segm.json"
            content={}
            content["n_sp"] = self.spinBox_n_superpixels.value()
            content["slic_comp"] = self.spinBox_slic_compactness.value()
            content["slic_sig"] = self.spinBox_slic_sigma.value()
            file_path = os.path.join(self.path_segm,file_name)
            json.dump(content,open(os.path.join(file_path),'w'))
	    subprocess.call(['chmod','755',file_path])
            self.update_segm_list()
	    self.name_segm.clear()
	else:
	    return

    def save_new_vq(self):
	"""
	Callback function for the save vector quantization parameters push button.  Overwrites <name>_segm.json file 
	if same as one in the list. Otherwise, creates a new file according to the spinbox values.
	"""
	name = str(self.name_vq.text())
	if  name !="":       
	    file_name =  name+"_vq.json"
            content={}
            content["n_texton"] = self.spinBox_n_texton.value()
            content["n_sample"] = self.spinBox_n_sample.value()
            content["n_iter"] = self.spinBox_n_iter.value()
            file_path = os.path.join(self.path_vq,file_name)
            json.dump(content,open(os.path.join(file_path),'w'))
	    subprocess.call(['chmod','755',file_path])
            self.update_vq_list()
	    self.name_vq.clear()
	else:
	    return

    ### Pop Up Menu ###
    def pop_up_menu(self,item,file_name,suffix):
	"""
	Pop up menu function controls the ability to rename or delete pre-existing files. By right clicking
	on an item in the list you will have the ability to either rename or delete the file.
	"""
	file_path = os.path.join(self.path_main,suffix,file_name)
	menu = QMenu()
	
	renameAction = menu.addAction("Rename")
	deleteAction = menu.addAction("Delete")
	action = menu.exec_(QCursor.pos())
	
	if action == deleteAction:
	    subprocess.call(["rm",file_path])
	    self.update_gabor_list()
	    self.update_segm_list()
	    self.update_vq_list()
	
	elif action == renameAction and suffix == "gabor":
	    try:
		self.list_gabor.itemChanged.disconnect()
	    except:
		pass
	    item.setFlags(item.flags() | Qt.ItemIsEditable)
	    self.list_gabor.editItem(item)
	    self.list_gabor.itemChanged.connect(lambda: self.rename(item,file_name,file_path,suffix))
	
	elif action == renameAction and suffix == "segm":
	    try:
		self.list_segm.itemChanged.disconnect()
	    except:
		pass
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            self.list_segm.editItem(item)
            self.list_segm.itemChanged.connect(lambda: self.rename(item,file_name,file_path,suffix))

	elif action == renameAction and suffix == "vq":
	    try:
		self.list_vq.itemChanged.disconnect()
	    except:
		pass
	    item.setFlags(item.flags() | Qt.ItemIsEditable)
            self.list_vq.editItem(item)
            self.list_vq.itemChanged.connect(lambda: self.rename(item,file_name,file_path,suffix))
	return

    def rename(self,item,file_name,old_path,suffix):
	"""
	This controls the renaming of a file. Once editing is finished, a subprocess
	call is used to rename the file.
	"""
	
	if  suffix == "gabor" and self.list_gabor.currentItem() !=None:
	    name = str(self.list_gabor.currentItem().text())
	    old = old_path
            new = os.path.join(self.path_main,suffix,name+"_"+suffix+".json")
            subprocess.call(['mv',old,new])
	    self.update_gabor_list()
	    self.list_gabor.itemChanged.disconnect()
	    return

	elif  suffix == "segm" and self.list_segm.currentItem() !=None:
	    name = str(self.list_segm.currentItem().text())
	    old = old_path
            new = os.path.join(self.path_main,suffix,name+"_"+suffix+".json")
            subprocess.call(['mv',old,new])
	    self.update_segm_list()
 	    self.list_segm.itemChanged.disconnect()
	    return

	elif  suffix == "vq" and self.list_vq.currentItem() !=None:
	    name = str(self.list_vq.currentItem().text())	
            old = old_path
            new = os.path.join(self.path_main,suffix,name+"_"+suffix+".json")
	    subprocess.call(['mv',old,new])
	    self.update_vq_list()
	    self.list_vq.itemChanged.disconnect()
	    return
	
	else:
	    return
	
	

app = QApplication(sys.argv)
form = MainDialog()
form.show()
app.exec_()
