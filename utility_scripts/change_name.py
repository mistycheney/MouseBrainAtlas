import re
import os
import sys


os.chdir(sys.argv[1])
for f in os.listdir('.'):
	res = re.findall('^(.*)_param(.*)$', f)
	if len(res) > 0:
		newname = res[0][0] + res[0][1]
		print f, '->', newname
		os.rename(f, newname)

