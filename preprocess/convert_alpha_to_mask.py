import os
import sys
import subprocess
import argparse

parser = argparse.ArgumentParser(description="Generate masks from the alpha channels of tif files")
parser.add_argument("img_dir", help="directory containing the tif images")
args = parser.parse_args()

# img_dir = sys.argv[1]
os.chdir(args.img_dir)

for img in os.listdir('.'):

	img = os.path.realpath(img)

	name, ext = img.split('.')

	cmd = 'convert  %s -channel Alpha -black-threshold 100%% -separate %s_mask.png ' % (img, name)
	print cmd
	subprocess.call(cmd, shell=True)

	cmd = 'convert  %s_mask.png -black-threshold 100%% %s_mask.png ' % (name, name)
	print cmd
	subprocess.call(cmd, shell=True)