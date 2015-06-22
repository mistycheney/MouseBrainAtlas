from __future__ import print_function
import sys
import os
import subprocess
import argparse


### Setup argument variables ###
parser = argparse.ArgumentParser(
formatter_class=argparse.RawDescriptionHelpFormatter,
description="Rescale mask for various resolutions",
epilog ="""This script will scale down the x20 resolutions; 
e.g. generate_makes_allres.py CC99 -x 1.25,5
This generates masks from the 20x resolutions at the desired resolution for the CC99 stack and 
saves to appropriate directories. This script requires x20 mask.""")

parser.add_argument("stack", type = str, help= "the stack to generate mask")
parser.add_argument("-d","--dir", type = str, help = "the directory where the stack is located",default = "BrainstemImages2015")
parser.add_argument("-x","--res", nargs = '+',type =str, help="resolutions to be sectioned")

args = parser.parse_args()
stack = args.stack
slide_dir = args.dir
res = args.res

### Functions ###
def identify_img(path):
	#currently scales the x5 values for x20 due to integer overflow errors with identify
	if res == "x20":
		img_id = subprocess.check_output(["identify",path.replace("x20","x5")]).split()
		w = 4*int(img_id[2].split('x')[0]) 
		h = 4*int(img_id[2].split('x')[1])
	else:
		img_id = subprocess.check_output(["identify",path]).split()
		w = int(img_id[2].split('x')[0]) 
		h = int(img_id[2].split('x')[1])
	return (w,h)


def change_permissions(path,val):
	subprocess.call(["chmod",str(val),path])
	return

def generate_mask(img_tuple,res):
	mask_x20 = img_tuple[0]
	img = img_tuple[1]
	mask = img_tuple[2]
	mask_x20_path = os.path.join(stack_path,"x20",mask_x20)
	img_path = os.path.join(stack_path,res,img)
	mask_path = os.path.join(stack_path,res,mask)
	w, h = identify_img(img_path)
	cmd = ["convert",mask_x20_path,"-resize",str(w)+'x'+str(h)+'!',mask_path]

	#Calls the resize command
	subprocess.call(cmd)

	#Change permissions of resized image
	change_permissions(mask_path,755)
	
	return


### Setup main directory and resolution list ###
main_dir = "/oasis/projects/nsf/csd181/yuncong/"
stack_path = os.path.join(main_dir,slide_dir,stack)
res_list = res[0].split(',')



### Iterates through the x20 resolutions and generates the desired resized masks ###
images_path_x20 = os.path.join(stack_path,"x20")
img_list_x20 = subprocess.check_output(["ls",images_path_x20]).split()
mask_list_x20=[]

for img in img_list_x20:
	if img.find("mask") > 0:
		mask_list_x20+=[img]
for res in res_list:        
	mask_list = [img.replace("x20",'x'+res) for img in mask_list_x20]
	img_list =[img.replace("_mask.png",".jpeg") for img in mask_list]
	joint_list = [(x,y,z) for (x,y,z) in zip(mask_list_x20,img_list,mask_list)]

	total = len(joint_list)
	for i,item in enumerate(joint_list):

		print("resizing mask %d of %d" % (i+1,total),end='\r',file=sys.stdout.flush())
		generate_mask(item,'x'+res)

	print("Completed resizing at x%s resolution" % res)	