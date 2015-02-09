from subprocess import call
import os

def run_distributed(script_name, arg_tuples):

	hostids = range(31,39) + range(41,49)
	n_hosts = len(hostids)

	with open('/tmp/argfile', 'w') as f:
		for i, args in enumerate(arg_tuples):
			hostid = 'gcn-20-%d.sdsc.edu' % hostids[i%n_hosts]
			f.write(' '.join([hostid] + map(str, args)) + '\n')

	n_args = len(arg_tuples[0])

	args_str = ' '.join(['{%d}'%i for i in range(2, 2 + n_args)])

	cmd = "parallel --colsep ' ' ssh yuncong@{1} 'python %s %s' :::: /tmp/argfile" % (script_name, args_str)
	print cmd
	call(cmd, shell=True)


def create_if_not_exists(path):
	if not os.path.exists(path):
		os.makedirs(path)
	return path

def execute_command(cmd):
	try:
		retcode = call(cmd, shell=True)
		if retcode < 0:
			print >>sys.stderr, "Child was terminated by signal", -retcode
		else:
			print >>sys.stderr, "Child returned", retcode
	except OSError as e:
		print >>sys.stderr, "Execution failed:", e
		raise e
