import os
import sys
import subprocess

img_dir = sys.argv[1]
os.chdir(img_dir)

for img in os.listdir('.'):

	img = os.path.realpath(img)

	name, ext = img.split('.')

	cmd = 'convert  %s -channel Alpha -separate %s_mask.png ' % (img, name)
	print cmd
	subprocess.call(cmd, shell=True)