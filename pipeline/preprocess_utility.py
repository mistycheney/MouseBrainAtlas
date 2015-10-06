from subprocess import call
import os
import sys

def first_last_tuples_distribute_over(first_sec, last_sec, n_host):
    secs_per_job = (last_sec - first_sec + 1)/float(n_host)
    if secs_per_job < 1:
    	first_last_tuples = [(i,i) for i in range(first_sec, last_sec+1)]
    else:
    	first_last_tuples = [(int(first_sec+i*secs_per_job), int(first_sec+(i+1)*secs_per_job-1) if i != n_host - 1 else last_sec) for i in range(n_host)]
    return first_last_tuples

def detect_responsive_nodes(exclude_nodes=[]):

	print ['gcn-20-%d.sdsc.edu'%i for i in exclude_nodes], 'are excluded'

	hostids = [i for i in range(31,39)+range(41,49) if i not in exclude_nodes]
	# hostids = range(31,33) + range(34,39) + range(41,49)
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


def run_distributed3(command, first_sec, last_sec, stdout=None, exclude_nodes=[]):

	hostids = detect_responsive_nodes(exclude_nodes=exclude_nodes)
	n_hosts = len(hostids)

	temp_script = '/tmp/runall.sh'

	with open(temp_script, 'w') as temp_f:
		for i, (f, l) in enumerate(first_last_tuples_distribute_over(first_sec, last_sec, n_hosts)):
			line = "ssh yuncong@%(hostname)s \"%(generic_launcher_path)s \'%(command)s\' %(f)d %(l)d\" &" % \
					{'hostname': 'gcn-20-%d.sdsc.edu' % hostids[i%n_hosts],
					'generic_launcher_path': os.environ['GORDON_PIPELINE_SCRIPT_DIR'] + '/generic_controller.py',
					'command': command,
					'f': f, 
					'l': l
					}
			temp_f.write(line + '\n')
		temp_f.write('wait\n')
		temp_f.write('echo =================\n')
		temp_f.write('echo all jobs are done!\n')
		temp_f.write('echo =================\n')

	os.chmod(temp_script, 0o777)
	call(temp_script, shell=True, stdout=stdout)

# def run_distributed3(script_name, arg_tuples, stdout=None, exclude_nodes=[]):

# 	hostids = detect_responsive_nodes(exclude_nodes=exclude_nodes)
# 	n_hosts = len(hostids)

# 	temp_script = '/tmp/runall.sh'

# 	n_jobs = len(arg_tuples)

# 	with open(temp_script, 'w') as f:
# 		# arg_tuple_batches = [[arg_tuples[j] for j in range(n_hosts*i, min(n_hosts*(i+1), n_jobs))] for i in range(n_jobs/n_hosts+1)]

# 		first_secs = range(0, n_jobs, n_hosts)
# 		arg_tuple_batches = [arg_tuples[n_hosts*i : min(n_hosts*(i+1)+1, n_jobs)] for i in first_secs]

# 		for arg_tuple_batch in arg_tuple_batches:
# 			for i, args in enumerate(arg_tuple_batch):
# 				hostid = 'gcn-20-%d.sdsc.edu' % hostids[i%n_hosts]
# 				line = "ssh yuncong@%s python %s %s &" % (hostid, script_name, ' '.join(map(str, args)))
# 				f.write(line + '\n')
# 			f.write('wait\n')
# 			f.write('echo =================\n')
# 			f.write('echo a batch is done!\n')
# 			f.write('echo =================\n')
# 		f.write('echo =================\n')
# 		f.write('echo all jobs are done!\n')
# 		f.write('echo =================\n')

# 	call('chmod u+x ' + temp_script, shell=True)
# 	call(temp_script, shell=True, stdout=stdout)

# 	# execute_command('/tmp/runall.sh')

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
