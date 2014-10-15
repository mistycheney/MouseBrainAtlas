from os import system
from subprocess import check_output
import argparse

#setup argparse
parser = argparse.ArgumentParser(
formatter_class=argparse.RawDescriptionHelpFormatter,
description='Run feature_extraction_pipeline_v2.py on entire stack of images',
epilog ="""An example to run feature_extraction_pipeline_v2.py on the images in subdirectory RS141_x5.
python feature_extract_stack.py RS141_x5 nissl324 -o /my/output/folder -p /my/param/folder
This will run the feature extraction pipeline on everyimage in specified subdirectory. It will use
the parameter file from /my/param/folder on gordon and save the output to /my/output/folder on gordon.""")

parser.add_argument("img_subdir", type=str, help="subdirectory where the images are located")
parser.add_argument("param", type=str, help = "the parameter file you want to use on the images")
parser.add_argument("-o","--output_dir", type =str, help="directory of where you want the output saved",default='/oasis/scratch/csd181/yuncong/output')
parser.add_argument("-p","--param_dir", type =str ,help = "directory of where you will look for the parameter file",default='/oasis/projects/nsf/csd181/yuncong/Brain/params')

args = parser.parse_args()
img_subdir = args.img_subdir
output_dir = args.output_dir
param_dir = args.param_dir
param = args.param


#Creates strings to correctly form str_cmd
data_dir = "/oasis/projects/nsf/csd181/yuncong/DavidData/"
img_dir = data_dir + img_subdir
script_dir="/oasis/projects/nsf/csd181/yuncong/Brain/scripts/"
setup_cmd ="source /oasis/projects/nsf/csd181/yuncong/Brain/setup.sh"
feature_script = script_dir+"feature_extraction_pipeline_v2.py"

#Creates two list --hostnames(available computers on gordon) and
#a list of images in the stack, which is specified by img_subdir
str1 = open('hostname.list','r').read()
hostnames = str1.splitlines()
str2 = open(img_subdir+".list",'r').read()
images = str2.splitlines()

#while loop which assigns a computer to each image and then calls the correct ssh cmd
hostname = hostnames[0]
i=0
while True: 
    #fail condition (each image has been assigned a computer which will break the loop)
    if ',' in images[0]:
        break
    else:
        img_name = images[0]
        computer = hostnames[i]
        images.append(images[0]+','+hostnames[i])
        images.remove(images[0])
        hostnames.append(hostnames[i])
	i=i+1 #this evenly distributes the image and computer assignment
        ssh_cmd = "ssh "+computer+" "
        run_feature_cmd = "python "+feature_script+" -o "+output_dir+" -p "+param_dir+" "+img_dir+"/"+img_name+" "+param
        notify = "echo Finished processing " + img_name + " on " + computer
	date = check_output('date').replace(" ","",2).replace(" ","_").replace("\n","")
	log = "%s" % img_name.replace('.tif',"_"+date+".log")
	#setting up full_cmd in string format for os.system
	full_cmd = ssh_cmd + "'" + setup_cmd + " && " + "("+run_feature_cmd + ")&> "+output_dir+"/"+log+" && " +notify + "'"+ " &"
        system(full_cmd) #currently works properly
	#print full_cmd      #for testing purposes only

	
