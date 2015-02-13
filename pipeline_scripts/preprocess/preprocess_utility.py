from subprocess import call
import os
import sys

def run_distributed3(script_name, arg_tuples):
	hostids = range(31, 34) + [35] + range(37, 39) + range(41,49)
	n_hosts = len(hostids)

	temp_script = '/tmp/runall.sh'

	with open(temp_script, 'w') as f:
		for i, args in enumerate(arg_tuples):
			hostid = 'gcn-20-%d.sdsc.edu' % hostids[i%n_hosts]
			line = "ssh yuncong@%s python %s %s &" % (hostid, script_name, ' '.join(map(str, args)))
			f.write(line + '\n')

		f.write('wait\n')
		f.write('echo all jobs are done!\n')

	call('chmod u+x ' + temp_script, shell=True)
	call(temp_script, shell=True)

	# execute_command('/tmp/runall.sh')


def run_distributed(script_name, arg_tuples):
	hostids = range(31, 36) + range(37, 39) + range(41,49)
	n_hosts = len(hostids)

	with open('/tmp/argfile', 'w') as f:
		for i, args in enumerate(arg_tuples):
			hostid = 'gcn-20-%d.sdsc.edu' % hostids[i%n_hosts]
			f.write(' '.join([hostid] + map(str, args)) + '\n')

	n_args = len(arg_tuples[0])
	args_str = ' '.join(['{%d}'%i for i in range(2, 2 + n_args)])

	cmd = """parallel --colsep ' ' "ssh yuncong@{1} 'python %s %s' &" :::: /tmp/argfile""" % (script_name, args_str)
	print cmd
	call(cmd, shell=True)

def run_distributed2(script_name, arg_tuples):
	
	with open('/tmp/argfile', 'w') as f:
		for i, args in enumerate(arg_tuples):
			f.write(' '.join(map(str, args)) + '\n')

	n_args = len(arg_tuples[0])
	args_str = ' '.join(['{%d}'%i for i in range(1, 1 + n_args)])

	cmd = """parallel --sshloginfile parallel_sshlogins  --colsep ' '  "hostname; python %s %s" :::: /tmp/argfile"""  % (script_name, args_str)
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
