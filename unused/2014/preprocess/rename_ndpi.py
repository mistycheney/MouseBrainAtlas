import os
import re
import shutil
import sys

# $ rename in_dir out_dir

in_dir = sys.argv[1]
out_dir = sys.argv[2]

if not os.path.exists(out_dir):
   os.makedirs(out_dir)

os.chdir(in_dir)
#names = {}
for fn in os.listdir('.'):
   if not fn.endswith('ndpi'): continue
   try:
	stack, slide_str, _,_,_,_,_,_ = re.split('\W+', fn[:-5])
   except:
	continue

   if slide_str[0].isalpha():
	slide_ind = int(slide_str[1:])   # slide_str is Gxx where xx is slide num
   else:
	slide_ind = int(slide_str)

#   names[slide_ind] = fn

   new_name = '%s_%02d.ndpi' % (out_dir, slide_ind)
   shutil.copy(fn, '../'+out_dir+'/'+new_name)

