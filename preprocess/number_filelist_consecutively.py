#!/usr/bin/env python

stack = sys.argv[1]
list_fn = '/home/yuncong/CSHL_data_processed/' + stack + '_filename_map.txt'

with open(list_fn, 'r') as f:
    fns = [l.split()[0] for l in f.readlines()]
    
with open(list_fn, 'w') as f:
    for i, fn in enumerate(fns):
        f.write(fn + ' ' + str(i+1) + '\n')
