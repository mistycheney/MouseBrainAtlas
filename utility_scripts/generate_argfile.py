hostids = range(31,39) + range(41,49)

with open('argfile', 'w') as f:
	for i in range(len(hostids)):
		f.write('gcn-20-%d.sdsc.edu %04d\n'%(hostids[i], 2+i))