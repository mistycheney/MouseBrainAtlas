import os
import time

from preprocess_utility import *

t = time.time()


script_root = os.environ['GORDON_REPO_DIR']+'/notebooks/'
arg_tuples = [[i] for i in range(7)]
run_distributed3(script_root+'/filter_image_with_landmark_template_executable.py', arg_tuples)


print 'total', time.time() - t, 'seconds'
