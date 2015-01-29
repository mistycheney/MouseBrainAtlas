#Quick way to generate necessary .list files
#setup argparse
import argparse
parser = argparse.ArgumentParser(description='Create image list')
parser.add_argument("img_subdir", type=str, help="subdirectory where the images are located")
parser.add_argument("num", type=int, help = "number of images in the subdirectory")

args = parser.parse_args()

seq=[]
for i in range(args.num):
    im = args.img_subdir+"_"+str(i).zfill(4)+".tif"
    seq.append(im+"\n")
f = open(args.img_subdir+'.list','w')
f.writelines(seq)
f.close()
