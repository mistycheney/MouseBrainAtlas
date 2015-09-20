import os, sys


stack = sys.argv[1]
input_dir = sys.argv[2]
output_dir = sys.argv[3]
argfile = sys.argv[4]
suffix = sys.argv[5]

with open(argfile, 'r') as f:
	fns = map(lambda x: x.split(), f.readlines())
	L = [(row[0], int(row[1]), int(row[2]), int(row[3])) for row in fns]

for fn, r, g, b in L:
	d = {
	'fn': fn,
	'r': r,
	'g': g,
	'b': b,
	'input_dir': input_dir,
	'output_dir': output_dir,
	'suffix': suffix
	}

	os.system("""convert %(input_dir)s/%(fn)s_%(suffix)s.tif -trim -fuzz 4%% -fill "rgb(%(r)s,%(g)s,%(b)s)" -opaque white -compress lzw %(output_dir)s/%(fn)s_%(suffix)s.tif"""%d)