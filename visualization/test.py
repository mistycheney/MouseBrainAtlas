from utilities import *
from pprint import pprint

paths = [('a',1), ('a_1',2), ('b_2',4), ('c',4), ('b',1), ('c_6', 3), ('c_6_2', 9), ('b_2_3',0)]

d = labeled_paths_to_tree(paths)
