from subprocess import call
from metadata import *
import subprocess
import boto3
import os
import sys
import cPickle as pickle
import json

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import execute_command

def delete_file_or_directory(fp):
    execute_command("rm -rf %s" % fp)


def download_from_s3(local_path, s3_path = None):
    #downloading 500 files of 1Mb each
    #boto3 - 36 seconds
    #aws cli - 5 seconds
    s3_connection = boto3.resource('s3')
    if s3_path == None:
        s3_path = DataManager.map_local_filename_to_s3(local_path)
    bucket, file_to_download= s3_path.split("s3://")[1].split("/", 1)
    #file_to_download = file_to_download.split("/", 1)[1]
    
    bucket = s3_connection.Bucket(bucket)
    create_parent_dir_if_not_exists(local_path)
    if len(list(bucket.objects.filter(Prefix=file_to_download))) > 1:
        execute_command('aws s3 cp --recursive %s %s' % (s3_path, local_path))
        #subprocess.call(["aws", "s3", "cp", s3_path, local_path, "--recursive"], stdout = open(os.devnull, 'w'))
    else:
        bucket.download_file(file_to_download, local_path)
    return local_path

def upload_to_s3(local_path, s3_path = None, output = False):
    #uploading 500 files of 1Mb each
    #boto3 - 1 minute 24  seconds
    #aws cli - 7 seconds
    if s3_path == None:
        s3_path = map_local_filename_to_s3(local_path)
    execute_command('aws s3 cp --recursive %s %s' % (local_path, s3_path))
    # if output == True:
    #     subprocess.call(["aws", "s3", "cp", local_path, s3_path, "--recursive"])
    # else:
    #     subprocess.call(["aws", "s3", "cp", local_path, s3_path, "--recursive"], stdout = open(os.devnull, 'w'))

    
def transfer_data_s3(from_fp, to_fp, from_hostname='localhost', to_bucket=S3_DATA_BUCKET):
    to_parent = os.path.dirname(to_fp)
    s3_path = 's3://' + to_bucket + '/' + to_fp
    if from_hostname == 'localhost':
        # upload
        execute_command('aws s3 cp --recursive %s %s' % (local_path, s3_path))
    elif to_hostname == 'localhost':
        # download
        execute_command('aws s3 cp --recursive %(s3_path)s %(local_path)s' % dict(s3_path=s3_path, local_path=local_path))
        # execute_command("rm -rf %(to_fp)s && mkdir -p %(to_parent)s && scp -r %(from_hostname)s:%(from_fp)s %(to_fp)s" % \
        #                     dict(from_fp=from_fp, to_fp=to_fp, from_hostname=from_hostname, to_parent=to_parent))
    else:
        # log onto another machine and perform upload from there.
        # execute_command("ssh %(from_hostname)s \"ssh %(to_hostname)s \'rm -rf %(to_fp)s && mkdir -p %(to_parent)s && scp -r %(from_fp)s %(to_hostname)s:%(to_fp)s\'\"" % \
        #                 dict(from_fp=from_fp, to_fp=to_fp, from_hostname=from_hostname, to_hostname=to_hostname, to_parent=to_parent))
        raise Exception('Not implemented.')


def transfer_data(from_fp, to_fp, from_hostname='localhost', to_hostname='oasis-dm.sdsc.edu'):
    to_parent = os.path.dirname(to_fp)
    if from_hostname == 'localhost':
        # upload
        execute_command("ssh %(to_hostname)s 'rm -rf %(to_fp)s && mkdir -p %(to_parent)s' && scp -r %(from_fp)s %(to_hostname)s:%(to_fp)s" % \
                    dict(from_fp=from_fp, to_fp=to_fp, to_hostname=to_hostname, to_parent=to_parent))
    elif to_hostname == 'localhost':
        # download
        execute_command("rm -rf %(to_fp)s && mkdir -p %(to_parent)s && scp -r %(from_hostname)s:%(from_fp)s %(to_fp)s" % \
                        dict(from_fp=from_fp, to_fp=to_fp, from_hostname=from_hostname, to_parent=to_parent))
    else:
        # log onto another machine and perform upload from there.
        execute_command("ssh %(from_hostname)s \"ssh %(to_hostname)s \'rm -rf %(to_fp)s && mkdir -p %(to_parent)s && scp -r %(from_fp)s %(to_hostname)s:%(to_fp)s\'\"" % \
                        dict(from_fp=from_fp, to_fp=to_fp, from_hostname=from_hostname, to_hostname=to_hostname, to_parent=to_parent))

# def upload_to_remote(fp_local, fp_remote, remote_hostname='oasis-dm.sdsc.edu'):
#     execute_command("scp -r %(fp_local)s %(remote_hostname)s:%(fp_remote)s" % \
#                     dict(fp_remote=fp_remote, fp_local=fp_local, remote_hostname=remote_hostname))

# def download_from_remote(fp_remote, fp_local, remote_hostname='oasis-dm.sdsc.edu'):
#     execute_command("scp -r %(remote_hostname)s:%(fp_remote)s %(fp_local)s" % \
#                     dict(fp_remote=fp_remote, fp_local=fp_local, remote_hostname=remote_hostname))

default_root_mapping = dict(localhost='/home/yuncong/CSHL_data_processed', dm='/home/yuncong/csd395/CSHL_data_processed')

def transfer_data_synced(fp_relative, from_hostname='localhost', to_hostname='dm', from_root=None, to_root=None):
    if from_root is None:
        from_root = default_root_mapping[from_hostname]
    if to_root is None:
        to_root = default_root_mapping[to_hostname]

    from_fp = os.path.join(from_root, fp_relative)
    to_fp = os.path.join(to_root, fp_relative)
    transfer_data(from_fp=from_fp, to_fp=to_fp, from_hostname=from_hostname, to_hostname=to_hostname)

# def upload_to_remote_synced(fp_relative, stack=None, remote_root='/home/yuncong/csd395/CSHL_data_processed', local_root='/home/yuncong/CSHL_data_processed'):
#     if stack is not None:
#         remote_fp = os.path.join(remote_root, stack, fp_relative)
#         local_fp = os.path.join(local_root, stack, fp_relative)
#     else:
#         remote_fp = os.path.join(remote_root, fp_relative)
#         local_fp = os.path.join(local_root, fp_relative)
#     create_if_not_exists(os.path.dirname(local_fp))
#     execute_command("scp -r %(fp_local)s oasis-dm.sdsc.edu:%(fp_remote)s" % \
#                     dict(fp_remote=remote_fp, fp_local=local_fp))

# def download_from_remote_synced(fp_relative, remote_root='/home/yuncong/csd395/CSHL_data_processed', local_root='/home/yuncong/CSHL_data_processed'):
#     remote_fp = os.path.join(remote_root, fp_relative)
#     local_fp = os.path.join(local_root, fp_relative)
#     create_if_not_exists(os.path.dirname(local_fp))
#     execute_command("scp -r oasis-dm.sdsc.edu:%(fp_remote)s %(fp_local)s" % \
#                     dict(fp_remote=remote_fp, fp_local=local_fp))

def first_last_tuples_distribute_over(first_sec, last_sec, n_host):
    secs_per_job = (last_sec - first_sec + 1)/float(n_host)
    if secs_per_job < 1:
        first_last_tuples = [(i,i) for i in range(first_sec, last_sec+1)]
    else:
        first_last_tuples = [(int(first_sec+i*secs_per_job), int(first_sec+(i+1)*secs_per_job-1) if i != n_host - 1 else last_sec) for i in range(n_host)]
    return first_last_tuples

#def download_dir(client, resource, dist, local='/tmp', bucket='your_bucket'):
#    paginator = client.get_paginator('list_objects')
#    for result in paginator.paginate(Bucket=bucket, Delimiter='/', Prefix=dist):
#        if result.get('CommonPrefixes') is not None:
#            for subdir in result.get('CommonPrefixes'):
#                download_dir(client, resource, subdir.get('Prefix'), local)
#        if result.get('Contents') is not None:
#            for file in result.get('Contents'):
#                if not os.path.exists(os.path.dirname(local + os.sep + file.get('Key'))):
#                     os.makedirs(os.path.dirname(local + os.sep + file.get('Key')))
#                resource.meta.client.download_file(bucket, file.get('Key'), local + os.sep + file.get('Key'))

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
        
        # Wait for SGE to know all nodes. Timeout = 5 mins
        success = False
        for _ in range(60):
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
    if argument_type == 'single':
        for arg in kwargs_list_as_list:
            line = command % arg
    elif argument_type in ['partition', 'list', 'list2']:
        for i, (fi, li) in enumerate(first_last_tuples_distribute_over(0, len(kwargs_list_as_list)-1, cluster_size)):
            if argument_type == 'partition':
                # For cases with partition of first section / last section
                line = "%(command)s " % \
                    {
                    'command': command % {'first_sec': kwargs_list_as_dict['sections'][fi], 'last_sec': kwargs_list_as_dict['sections'][li]}
                    }
            elif argument_type == 'list':
            # Specify kwargs_str
                line = "%(command)s " % \
                {
                'command': command % {'kwargs_str': json.dumps(kwargs_list_as_list[fi:li+1])}
                }
            elif argument_type == 'list2':
            # Specify {key: list}
                line = "%(command)s\" &" % \
                {
                'command': command % {key: json.dumps(vals[fi:li+1]) for key, vals in kwargs_list_as_dict.iteritems()}
                }
            print(line)
            temp_f = open(temp_script, 'w')
            temp_f.write(line + '\n')
            temp_f.close()
            os.chmod(temp_script, 0o777)            
            call('qsub -V -l mem_free=60G -o %(stdout_log)s -e %(stderr_log)s %(script)s' % \
                 dict(script=temp_script, stdout_log='/home/ubuntu/stdout.log', stderr_log='/home/ubuntu/stderr.log'), 
                 shell=True, stdout=stdout)
    else:
        raise Exception('argument_type %s not recognized.' % argument_type)
        
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



# def run_distributed3(command, first_sec=None, last_sec=None, interval=1, section_list=None, stdout=None, exclude_nodes=[], take_one_section=True):
#     """
#     if take_one_section is True, command must contain %%(secind)d;
#     otherwise, command must contain %%(f)d %%(l)d
#     """
#
#     hostids = detect_responsive_nodes(exclude_nodes=exclude_nodes)
#     n_hosts = len(hostids)
#
#     temp_script = '/tmp/runall.sh'
#
#     # assign a job to each machine
#     # each line is a job that processes several sections
#     with open(temp_script, 'w') as temp_f:
#         if section_list is not None:
#             for i, (fi, li) in enumerate(first_last_tuples_distribute_over(0, len(section_list)-1, n_hosts)):
#                 if take_one_section:
#                     line = "ssh yuncong@%(hostname)s \"%(generic_launcher_path)s \'%(command)s\' --list %(seclist_str)s \" &" % \
#                             {'hostname': 'gcn-20-%d.sdsc.edu' % hostids[i%n_hosts],
#                             'generic_launcher_path': os.environ['REPO_DIR'] + '/distributed/generic_launcher.py',
#                             'command': command,
#                             # 'seclist_str': '+'.join(map(str, section_list[fi:li+1]))
#                             'seclist_str': pickle.dumps(section_list[fi:li+1])
#                             }
#                     temp_f.write(line + '\n')
#         else:
#             for i, (f, l) in enumerate(first_last_tuples_distribute_over(first_sec, last_sec, n_hosts)):
#                 if take_one_section:
#                     line = "ssh yuncong@%(hostname)s \"%(generic_launcher_path)s \'%(command)s\' -f %(f)d -l %(l)d -i %(interval)d\" &" % \
#                             {'hostname': 'gcn-20-%d.sdsc.edu' % hostids[i%n_hosts],
#                             'generic_launcher_path': os.environ['REPO_DIR'] + '/distributed/generic_launcher.py',
#                             'command': command,
#                             'f': f,
#                             'l': l,
#                             'interval': interval
#                             }
#                 else:
#                     line = "ssh yuncong@%(hostname)s %(command)s &" % \
#                             {'hostname': 'gcn-20-%d.sdsc.edu' % hostids[i%n_hosts],
#                             'command': command % {'f': f, 'l': l}
#                             }
#
#                 temp_f.write(line + '\n')
#         temp_f.write('wait\n')
#         temp_f.write('echo =================\n')
#         temp_f.write('echo all jobs are done!\n')
#         temp_f.write('echo =================\n')
#
#     os.chmod(temp_script, 0o777)
#     call(temp_script, shell=True, stdout=stdout)
