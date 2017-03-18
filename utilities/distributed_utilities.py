from subprocess import call
import subprocess
import boto3
import os
import sys
import cPickle as pickle
import json
from utilities2015 import execute_command
from metadata import *

def delete_file_or_directory(fp):
    execute_command("rm -rf %s" % fp)

def transfer_data(from_fp, to_fp, from_hostname, to_hostname, is_dir, include_only=None):
    assert from_hostname in ['localhost', 'oasis', 's3', 'ec2'], 'from_hostname must be one of localhost, oasis, s3 or ec2.'
    assert to_hostname in ['localhost', 'oasis', 's3', 'ec2'], 'to_hostname must be one of localhost, oasis, s3 or ec2.'

    to_parent = os.path.dirname(to_fp)
    oasis = 'oasis-dm.sdsc.edu'

    if from_hostname in ['localhost', 'ec2']:
        # upload
        if to_hostname == 's3':
            if is_dir:
                execute_command('aws s3 cp --recursive %(from_fp)s s3://%(to_fp)s' % \
            dict(from_fp=from_fp, to_fp=to_fp))
            else:
                execute_command('aws s3 cp %(from_fp)s s3://%(to_fp)s' % \
            dict(from_fp=from_fp, to_fp=to_fp))
        else:
            execute_command("ssh %(to_hostname)s 'rm -rf %(to_fp)s && mkdir -p %(to_parent)s' && scp -r %(from_fp)s %(to_hostname)s:%(to_fp)s" % \
                    dict(from_fp=from_fp, to_fp=to_fp, to_hostname=oasis, to_parent=to_parent))
    elif to_hostname in ['localhost', 'ec2']:
        # download
        if from_hostname == 's3':
            
            # Clear existing folder/file
            execute_command('rm -rf %(to_fp)s && mkdir -p %(to_parent)s' % dict(to_parent=to_parent, to_fp=to_fp))
            
            # Download from S3 using aws commandline interface.
            if is_dir:
                if include_only is not None:
                    execute_command('aws s3 cp --recursive s3://%(from_fp)s %(to_fp)s --include \"%(include)s\"' % dict(from_fp=from_fp, to_fp=to_fp, include=include_only))
                else:
                    execute_command('aws s3 cp --recursive s3://%(from_fp)s %(to_fp)s' % dict(from_fp=from_fp, to_fp=to_fp))
            else:
                execute_command('aws s3 cp s3://%(from_fp)s %(to_fp)s' % dict(from_fp=from_fp, to_fp=to_fp))
        else:
            execute_command("scp -r %(from_hostname)s:%(from_fp)s %(to_fp)s" % dict(from_fp=from_fp, to_fp=to_fp, from_hostname=oasis))
    else:
        # log onto another machine and perform upload from there.
        execute_command("ssh %(from_hostname)s \"ssh %(to_hostname)s \'rm -rf %(to_fp)s && mkdir -p %(to_parent)s && scp -r %(from_fp)s %(to_hostname)s:%(to_fp)s\'\"" % \
                        dict(from_fp=from_fp, to_fp=to_fp, from_hostname=from_hostname, to_hostname=to_hostname, to_parent=to_parent))

default_root = dict(localhost='/home/yuncong', oasis='/home/yuncong/csd395', s3=S3_DATA_BUCKET, ec2='/shared')

def transfer_data_synced(fp_relative, from_hostname, to_hostname, is_dir, from_root=None, to_root=None, include_only=None):
    if from_root is None:
        from_root = default_root[from_hostname]
    if to_root is None:
        to_root = default_root[to_hostname]

    from_fp = os.path.join(from_root, fp_relative)
    to_fp = os.path.join(to_root, fp_relative)
    transfer_data(from_fp=from_fp, to_fp=to_fp, from_hostname=from_hostname, to_hostname=to_hostname, is_dir=is_dir, include_only=include_only)

def first_last_tuples_distribute_over(first_sec, last_sec, n_host):
    secs_per_job = (last_sec - first_sec + 1)/float(n_host)
    if secs_per_job < 1:
        first_last_tuples = [(i,i) for i in range(first_sec, last_sec+1)]
    else:
        first_last_tuples = [(int(first_sec+i*secs_per_job), int(first_sec+(i+1)*secs_per_job-1) if i != n_host - 1 else last_sec) for i in range(n_host)]
    return first_last_tuples

def detect_responsive_nodes_aws(exclude_nodes=[], use_nodes=None):
    def get_ec2_avail_instances(region):
        ins = []
        ec2_conn = boto3.client('ec2', region)
        #reservations = ec2_conn.get_all_reservations()
        response = ec2_conn.describe_instances()
        myid = subprocess.check_output(['wget', '-qO', '-', 'http://instance-data/latest/meta-data/instance-id'])
        for reservation in response["Reservations"]:
            for instance in reservation["Instances"]:
                if instance['State']['Name'] != 'running' or instance['InstanceType'] != 'm4.4xlarge':
                    continue
                if instance['InstanceId'] != myid:
                    ins.append(instance['PublicDnsName'])
                else:
                    ins.append('127.0.0.1')
        return ins
    all_nodes = get_ec2_avail_instances('us-west-1')

    if use_nodes is not None:
        hostids = use_nodes
    else:
        #for node in exclude_nodes:
        #    print(node)
        hostids = [node for node in all_nodes if node not in exclude_nodes]
    n_hosts = len(hostids)
    return hostids

def detect_responsive_nodes(exclude_nodes=[], use_nodes=None):

    all_nodes = range(31,39)+range(41,49)

    if use_nodes is not None:
        hostids = use_nodes
    else:
        print ['gcn-20-%d.sdsc.edu'%i for i in exclude_nodes], 'are excluded'
        hostids = [i for i in all_nodes if i not in exclude_nodes]

    # hostids = range(31,33) + range(34,39) + range(41,49)
    n_hosts = len(hostids)

    import paramiko
    # paramiko.util.log_to_file("filename.log")

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

def run_distributed(command, kwargs_list, stdout=open('/tmp/log', 'ab+'), exclude_nodes=[], use_nodes=None, argument_type='list', cluster_size=None):
    if ON_AWS:
        run_distributed5(command, kwargs_list, cluster_size, stdout, argument_type)
    else:
        run_distributed4(command, kwargs_list, stdout, exclude_nodes, use_nodes, argument_type)


def run_distributed5(command, kwargs_list, cluster_size, stdout=open('/tmp/log', 'ab+'), argument_type='list'):
    """
    Distributed executing a command on AWS.
    """

    if cluster_size is None:
        raise Exception('Must specify cluster_size.')

    import time

    n_hosts = subprocess.check_output('qhost').count('\n') - 3

    if n_hosts < cluster_size:
        autoscaling_description = json.loads(subprocess.check_output('aws autoscaling describe-auto-scaling-groups'.split()))
        asg = autoscaling_description[u'AutoScalingGroups'][0]['AutoScalingGroupName']
        subprocess.call("aws autoscaling set-desired-capacity --auto-scaling-group-name %s --desired-capacity %d" % (asg, cluster_size), shell=True)
        print "Setting autoscaling group %s capaticy to %d\n" % (asg, cluster_size)
        print "Wait for SGE to know all nodes...\n"

        # Wait for SGE to know all nodes. Timeout = 10 mins
        success = False
        for _ in range(600/5):
            n_hosts = (subprocess.check_output('qhost')).count('\n') - 3
            if n_hosts == cluster_size:
                success = True
                break
            time.sleep(5)

        if not success:
            raise Exception('SGE does not receive all host information in 300 seconds. Abort.')

    assert n_hosts == cluster_size

    temp_script = '/tmp/runall.sh'

    if isinstance(kwargs_list, dict):
        keys, vals = zip(*kwargs_list.items())
        kwargs_list_as_list = [dict(zip(keys, t)) for t in zip(*vals)]
        kwargs_list_as_dict = kwargs_list
    else:
        kwargs_list_as_list = kwargs_list
        keys = kwargs_list[0].keys()
        vals = [t.values() for t in kwargs_list]
        kwargs_list_as_dict = dict(zip(keys, vals))
      
    assert argument_type in ['single', 'partition', 'list', 'list2'], 'argument_type must be one of single, partition, list, list2.'

    for i, (fi, li) in enumerate(first_last_tuples_distribute_over(0, len(kwargs_list_as_list)-1, cluster_size)):
        if argument_type == 'partition':
            # For cases with partition of first section / last section
            line = command % {'first_sec': kwargs_list_as_dict['sections'][fi], 'last_sec': kwargs_list_as_dict['sections'][li]}
        elif argument_type == 'list':
        # Specify kwargs_str
            line = command % {'kwargs_str': json.dumps(kwargs_list_as_list[fi:li+1])}
        elif argument_type == 'list2':
        # Specify {key: list}
            line = command % {key: json.dumps(vals[fi:li+1]) for key, vals in kwargs_list_as_dict.iteritems()}
        elif argument_type == 'single':
            line = "%(generic_launcher_path)s \"%(command_template)s\" \"%(kwargs_list_str)s\"" % \
            {'generic_launcher_path': os.path.join(os.environ['REPO_DIR'], 'utilities', 'sequential_dispatcher.py'),
            'command_template': command,
            'kwargs_list_str': json.dumps(kwargs_list_as_list[fi:li+1]).replace('"','\\"').replace("'",'\\"')
            }
            
        temp_f = open(temp_script, 'w')
        temp_f.write(line)
        temp_f.close()
        os.chmod(temp_script, 0o777)
        call('qsub -V -l mem_free=60G -o %(stdout_log)s -e %(stderr_log)s %(script)s' % \
             dict(script=temp_script, stdout_log='/home/ubuntu/stdout_%d.log' % i, stderr_log='/home/ubuntu/stderr_%d.log' % i),
             shell=True, stdout=stdout)

    # Wait for qsub to complete.
    success = False
    for _ in range(0, 1200):
        op = subprocess.check_output('qstat')
        if "runall.sh" not in op:
            sys.stderr.write('qsub returned.\n')
            success = True
            break
        time.sleep(5)

    if not success:
        raise Exception('qsub does not return in 6000 seconds. Abort.')


def run_distributed4(command, kwargs_list, stdout=open('/tmp/log', 'ab+'), exclude_nodes=[], use_nodes=None, argument_type='list'):
    """
    There should be only one ssh connection to each node.
    """

    hostids = detect_responsive_nodes(exclude_nodes=exclude_nodes, use_nodes=use_nodes)
    print 'Using nodes:', ['gcn-20-%d.sdsc.edu'%i for i in hostids]
    n_hosts = len(hostids)

    temp_script = '/tmp/runall.sh'

    if isinstance(kwargs_list, dict):
        keys, vals = zip(*kwargs_list.items())
        kwargs_list_as_list = [dict(zip(keys, t)) for t in zip(*vals)]
        kwargs_list_as_dict = kwargs_list
    else:
        kwargs_list_as_list = kwargs_list
        keys = kwargs_list[0].keys()
        vals = [t.values() for t in kwargs_list]
        kwargs_list_as_dict = dict(zip(keys, vals))

    # for k,v in kwargs_list_as_dict.iteritems():
    #     sys.stderr.write(k+'\n')
    #     sys.stderr.write(str(v)+'\n')

    with open(temp_script, 'w') as temp_f:

        for i, (fi, li) in enumerate(first_last_tuples_distribute_over(0, len(kwargs_list_as_list)-1, n_hosts)):

            if argument_type == 'partition':
                # For cases with partition of first section / last section
                line = "ssh yuncong@%(hostname)s \"%(command)s\" &" % \
                        {'hostname': 'gcn-20-%d.sdsc.edu' % hostids[i%n_hosts],
                        'command': command % {'first_sec': kwargs_list_as_dict['sections'][fi], 'last_sec': kwargs_list_as_dict['sections'][li]}
                        }
            elif argument_type == 'list':
                # Specify kwargs_str
                line = "ssh yuncong@%(hostname)s \"%(command)s\" &" % \
                        {'hostname': 'gcn-20-%d.sdsc.edu' % hostids[i%n_hosts],
                        'command': command % {'kwargs_str': json.dumps(kwargs_list_as_list[fi:li+1]).replace('"','\\"').replace("'",'\\"')}
                        }
            elif argument_type == 'list2':
                # Specify {key: list}
                line = "ssh yuncong@%(hostname)s \"%(command)s\" &" % \
                        {'hostname': 'gcn-20-%d.sdsc.edu' % hostids[i%n_hosts],
                        'command': command % {key: json.dumps(vals[fi:li+1]).replace('"','\\"').replace("'",'\\"')
                                            for key, vals in kwargs_list_as_dict.iteritems()}
                        }
            elif argument_type == 'single':
                # the command takes only one element of the list
                line = "ssh yuncong@%(hostname)s \"%(generic_launcher_path)s \'%(command_template)s\' \'%(kwargs_list_str)s\' \" &" % \
                        {'hostname': 'gcn-20-%d.sdsc.edu' % hostids[i%n_hosts],
                        'generic_launcher_path': os.environ['REPO_DIR'] + '/utilities/sequential_dispatcher.py',
                        'command_template': command,
                        'kwargs_list_str': json.dumps(kwargs_list_as_list[fi:li+1]).replace('"','\\"').replace("'",'\\"')
                        }
            else:
                raise Exception('argument_type %s not recognized.' % argument_type)

            temp_f.write(line + '\n')

        temp_f.write('wait\n')
        temp_f.write('echo =================\n')
        temp_f.write('echo all jobs are done!\n')
        temp_f.write('echo =================\n')

    os.chmod(temp_script, 0o777)
    call(temp_script, shell=True, stdout=stdout)
