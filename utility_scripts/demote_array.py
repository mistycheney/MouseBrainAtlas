import numpy as np

np.save(sys.argv[1], np.load(sys.argv[1]).astype(np.int16))

