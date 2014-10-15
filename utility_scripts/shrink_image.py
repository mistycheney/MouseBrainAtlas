import os
import sys


orig_img = sys.argv[1]
name, ext = orig_img.split('.')
resized_img = name + '_resized.' + ext

cmd = 'convert -resize %%20 %s %s'%(orig_img, resized_img)
print cmd
# subprocess.call(cmd, shell=True)