import numpy as np
import sys

np.save(sys.argv[1], np.load(sys.argv[1]).astype(np.int16))

