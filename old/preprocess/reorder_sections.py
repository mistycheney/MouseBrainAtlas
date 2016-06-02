import numpy as np
import sys
import re
import os

if __name__ == '__main__':
	in_dir = sys.argv[1]
	out_dir = sys.argv[2]
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	c = 0
	files = sorted(os.listdir(in_dir))
	for fn in files:
		stack, resol, slide_ind, sect_ind = os.path.splitext(fn)[0].split('_')
		c += 1
		new_fn = '_'.join([stack, resol, '%04d'%c]) + '.tif'
		os.system('cp %s %s' % (os.path.join(in_dir, fn), os.path.join(out_dir, new_fn)))