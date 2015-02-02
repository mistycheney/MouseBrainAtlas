### Version 5 - currently allows the ability to crop, rotate, and flip the images ###
#Updated paths


import os
import argparse
import re
import subprocess


# Add file name argument of the image file

parser = argparse.ArgumentParser()
parser.add_argument("stack_res",type=str,help="choose what stack of images to crop and resolution, ex: RS141_x5")
parser.add_argument("-d","--img_directory",type=str,help="directory of where the image is stored, ex:",default = os.getcwd())
parser.add_argument("-o","--output_directory",type=str,help="directory of where the image is stored, ex:",default = os.getcwd())
parser.add_argument("-r","--rotation",type=str,help="how each slice will be rotated",default = None)
parser.add_argument("-m","--mirror",type=str,help="to mirror horizontal type 'flop', for vertical type 'flip'",default=None)

args = parser.parse_args()

stack_res = args.stack_res
img_dir = args.img_directory
out_dir = args.output_directory
rot_deg = args.rotation  
mirror = args.mirror


stack = stack_res.split("_")[0]
res = stack_res.split("_")[1]

img_list = subprocess.check_output(['ls',img_dir]).split()
m = [re.search(stack+".*",i) for i in img_list]
img_files = [item.group(0) for item in m if item!=None]

print img_files
def preprocess(img_file):
    ### Set's up the correct path for bounding data and output ###
    img_path = os.path.join(img_dir,img_file)
    txt_name = re.sub('_x.+','.txt',img_file)
    path2txt = re.sub('/DavidData.+','/Brain/preprocessing/bounding_box_data/'+stack+"/"+txt_name,img_path)
   
    

    ### Creates the stack folder ###
    stack_folder = os.path.join(out_dir,stack)
    if not os.path.exists(stack_folder):
        os.system("mkdir "+stack_folder)
    
    ### Create resolution folder ###
    res_folder = os.path.join(stack_folder,res)
    if not os.path.exists(res_folder):
		os.system("mkdir "+res_folder)
	

        
    #creates crop_dim ;list of tuples from txt_name.txt.
    crop_dim=[tuple(line.split()) for line in open(path2txt)]
    

    #uses subprocess with imagemagick to determine size and stores as w,h as a
    #string variable.

    img_id = subprocess.check_output(["identify",img_path]).split()
    tot_w = img_id[2].split('x')[0]
    tot_h = img_id[2].split('x')[1]
    
    #List of all images already cropped to avoid overwriting previously cropped images in the folder
   
    i = 0

    for (x,y,w,h) in crop_dim:
        (x,y,w,h) = (int(int(tot_w)*float(x)),int(int(tot_h)*float(y)),int(int(tot_w)*float(w)),int(int(tot_h)*float(h)))

        
	### creates subfolders 0000,0001,0002,...,etc ###
	path2folder = os.path.join(res_folder,str(i).zfill(4))
	if not os.path.exists(path2folder):
	    os.system("mkdir "+path2folder)
	else: 
	    while os.path.exists(path2folder):
	        i+=1
	        path2folder = os.path.join(res_folder,str(i).zfill(4))
	    os.system("mkdir "+path2folder)

	crop_img_name = stack_res + '_' + str(i).zfill(4) +'.tif'	
	crop_path = os.path.join(path2folder,crop_img_name)
        
        geom =str(w)+'x'+str(h)+'+'+str(x)+'+'+str(y)
	i+=1
        
        
        ### This crops the image according to bounding box data ###
        cmd1 = "convert %s -crop %s %s" % (img_path,geom,crop_path)           
        os.system(cmd1)
        
        ### This rotates the cropped image if specified ### 
        if rot_deg!=None:
            cmd2 = "convert %s -page +0+0 -rotate %s %s" % (crop_path,rot_deg,crop_path)
            os.system(cmd2)
            
        ### This reflects the rotated image if specified ###   
        if mirror!=None:
            cmd3 = "convert %s -%s %s" % (crop_path,mirror,crop_path)
            os.system(cmd3)
            
            
        print "Processed " + crop_img_name

### Processes each image starting with the correct stack name ###
for raw_image in img_files:
    preprocess(raw_image)
