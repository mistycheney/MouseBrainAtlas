from subprocess import call
import os
import sys

def detect_responsive_nodes():

	# hostids = range(31,39)+range(41,49)
	hostids = range(31,39)+range(41,49) # parallel jobs on gcn-33 executes forever for some reason 
	n_hosts = len(hostids)

	import paramiko
	paramiko.util.log_to_file("filename.log")

	ssh = paramiko.SSHClient()
	ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

	up_hostids = []
	for h in hostids:
		hostname = 'gcn-20-%d.sdsc.edu' % h
		try:
			ssh.connect(hostname, timeout=5)
			up_hostids.append(h)
		except:
			print hostname, 'is down'

	return up_hostids

# untested
def run_distributed4(script_name, arg_tuples):
	from pssh import ParallelSSHClient
	hostids = detect_responsive_nodes()

	hosts = ['gcn-20-%d.sdsc.edu'%i for i in hostids]
	client = ParallelSSHClient(hosts, timeout=5)
	output = client.run_command('hostname')
	for host in output:
	  for line in output[host]['stdout']:
	    print "Host %s - output: %s" % (host, line)


def run_distributed3(script_name, arg_tuples):

	hostids = detect_responsive_nodes()
	n_hosts = len(hostids)

	temp_script = '/tmp/runall.sh'

	n_jobs = len(arg_tuples)

	with open(temp_script, 'w') as f:
		arg_tuple_batches = [[arg_tuples[j] for j in range(n_hosts*i, min(n_hosts*(i+1), n_jobs))] for i in range(n_jobs/n_hosts+1)]

		for arg_tuple_batch in arg_tuple_batches:
			for i, args in enumerate(arg_tuple_batch):
				hostid = 'gcn-20-%d.sdsc.edu' % hostids[i%n_hosts]
				line = "ssh yuncong@%s python %s %s &" % (hostid, script_name, ' '.join(map(str, args)))
				f.write(line + '\n')
			f.write('wait\n')
			f.write('echo =================\n')
			f.write('echo a batch is done!\n')
			f.write('echo =================\n')
		f.write('echo =================\n')
		f.write('echo all jobs are done!\n')
		f.write('echo =================\n')

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

def execute_command(cmd, dryrun=False):
	print cmd

	try:
		if dryrun: return

		retcode = call(cmd, shell=True)
		if retcode < 0:
			print >>sys.stderr, "Child was terminated by signal", -retcode
		else:
			print >>sys.stderr, "Child returned", retcode
	except OSError as e:
		print >>sys.stderr, "Execution failed:", e
		raise e
