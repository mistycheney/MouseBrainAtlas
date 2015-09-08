import os
import time

from preprocess_utility import *

t = time.time()

DATA_DIR = '/home/yuncong/csd395/CSHL_data'

arg_tuples = [
('MD579', DATA_DIR+'/MD579', DATA_DIR+'/MD579_renamed'),
('MD585', DATA_DIR+'/MD585', DATA_DIR+'/MD585_renamed'),
('MD592', DATA_DIR+'/MD592', DATA_DIR+'/MD592_renamed'),
('MD593', DATA_DIR+'/MD593', DATA_DIR+'/MD593_renamed'),
('MD594', DATA_DIR+'/MD594', DATA_DIR+'/MD594_renamed')
]

run_distributed3('/home/yuncong/csd395/CSHL_data/rename.py', arg_tuples)

print 'total', time.time() - t, 'seconds'
