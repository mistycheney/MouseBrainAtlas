import sys
from pprint import pprint
import cPickle as pickle

a = pickle.load(open(sys.argv[1], 'r'))
pprint(a)
