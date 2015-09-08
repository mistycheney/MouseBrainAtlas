from __future__ import print_function
import sys
import os
import subprocess
import argparse

### Setup argument variables ###
parser = argparse.ArgumentParser(
formatter_class=argparse.RawDescriptionHelpFormatter,
description="extract sections from slides",
epilog ="""This script will extract sections from a slide according to the bounding box data; 
e.g. extract_sections.py CC99 CC99_brainstem_bboxes.txt -o BrainstemImages2015 -x 1.25,5,20
This generates the brainstem-only dataset at various resolutions""")

parser.add_argument("stack", type = str, help= "the stack to be sectioned")
parser.add_argument("bbox", type = str, help = "the text file containing bounding box data")
parser.add_argument("-o","--out_dir",type = str, help = "the directory where to save the extracted sections",default="BrainstemImages2015")
parser.add_argument("-d","--slide_dir", type = str, help = "the directory where the slides are saved",default = "DavidData2015slides")
parser.add_argument("-x","--res", nargs = '+',type =str, help="resolutions to be sectioned")

args = parser.parse_args()
stack = args.stack
slide_dir = args.slide_dir
bbox_file = args.bbox
out_dir = args.out_dir
res = args.res

### Functions ###
def create_if_not_exists(path_list):
	for path in path_list:
		if not os.path.exists(path):
			subprocess.call(["mkdir",path])
			print("Made Directory "+path)

def identify_slide(path):
	#currently scales the x5 values for x20 do to integer overflow errors with identify
	if res == "x20":
		img_id = subprocess.check_output(["identify",path.replace("x20","x5")]).split()
		w = 4*int(img_id[2].split('x')[0]) 
		h = 4*int(img_id[2].split('x')[1])
	else:
		img_id = subprocess.check_output(["identify",path]).split()
		w = int(img_id[2].split('x')[0]) 
		h = int(img_id[2].split('x')[1])
			

	return (w,h)

def append_save_labels(bbox_data):
	sections=[]
	save_label=[]
	old_section=0
	save_no=1
	for i in range(len(bbox_data)):
		next_section = int(bbox_data[i][0])
		if next_section != old_section:
			save_no=1
			old_section=next_section
		else:
			save_no+=1
			old_section=next_section
		save_label+=[[str(save_no)]]
	bbox_data = [y+x for (x,y) in zip(save_label,bbox_data)] 
	return bbox_data


def get_pixels(slideimg_path,dim):
	x,y,w,h=dim
	(total_w,total_h) = identify_slide(slideimg_path)
	x = float(x)/100
	y = float(y)/100
	w = float(w)/100
	h = float(h)/100
	(x_pix,y_pix,w_pix,h_pix) = (int(total_w*x),int(total_h*y),int(total_w*w),int(total_h*h))
	return (x_pix,y_pix,w_pix,h_pix)

def change_permissions(path,val):
	subprocess.call(["chmod",str(val),path])
	return

def change_format(path,oldformat,newformat):
	dir_name = os.path.dirname(path)
	new_name = os.path.basename(path).replace(oldformat,newformat)
	new_path = os.path.join(dir_name,new_name)
	subprocess.call(["convert",path,new_path])
	
	#Removes old format image
	subprocess.call(["rm",path])

	#Change permissions of new file
	change_permissions(new_path,755)
	return


### Setup main directory and resolution list ###
main_dir = "/oasis/projects/nsf/csd181/yuncong/"
bbox_data_path = os.path.join(main_dir,"Brain","BoundingBoxData",stack,bbox_file)
res_list = res[0].split(',')

### Retrieve bounding box data and append save labels ###
bbox_data = [list(line.split()) for line in open(bbox_data_path)]
bbox_data = append_save_labels(bbox_data)

### This for loop runs the extraction at each resolution ###
for res in res_list:
	
	### Setup of Gordon Directories ###
	res = 'x'+res
	slide_path = os.path.join(main_dir,slide_dir,stack,res)
	save_path = os.path.join(main_dir,out_dir,stack,res)
	
	### Create required directories for saving sections ###
	a = os.path.join(main_dir,out_dir)
	b = os.path.join(main_dir,out_dir,stack)       # e.g. */BrainstemImages/CC99
	c = os.path.join(main_dir,out_dir,stack,res)   # e.g. */BrainstemImages/CC99/x1.25
	
	create_if_not_exists([a,b,c])


	# Section the slide images #
	total = len(bbox_data)
	for i,item in enumerate(bbox_data):

		print("cropping image %d of %d" % (i+1,total),end='\r',file=sys.stdout.flush())
		
		slide_no = item[0].zfill(2)
		save_no = item[-1].zfill(2)
		image = '_'.join([stack,slide_no,res,"z0.tif"])
		image_path = os.path.join(slide_path,image)
		crop_image = '_'.join([stack,res,slide_no,save_no+".tif"])
		crop_path = os.path.join(save_path,crop_image)
		(x,y,w,h) = get_pixels(image_path,item[2:6])
		geometry = '+'.join([str(w)+'x'+str(h),str(x),str(y)])
		crop_cmd =  ["convert",image_path,"-crop",geometry,crop_path]
		subprocess.call(crop_cmd)
		change_format(crop_path,"tif","jpeg")

	print("Completed extraction at %s resolution" % res)	
