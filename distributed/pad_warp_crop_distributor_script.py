import os
import time

from preprocess_utility import *

t = time.time()

first_sec = int(sys.argv[1])
last_sec = int(sys.argv[2])

script_root = os.environ['GORDON_REPO_DIR']+'/notebooks/'
arg_tuples = [[i] for i in range(first_sec, last_sec+1)]
run_distributed3(script_root+'/pad_warp_crop.py', arg_tuples)

print 'total', time.time() - t, 'seconds'
