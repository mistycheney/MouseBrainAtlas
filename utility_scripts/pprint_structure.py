import cPickle as pickle
from pprint import pprint
import sys

d = pickle.load(open(sys.argv[1], 'r'))
pprint(d)
