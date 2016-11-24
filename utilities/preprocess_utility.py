from subprocess import call
from metadata import *
import subprocess
import boto.ec2
import os
import sys
import cPickle as pickle
import json

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
        ec2_conn = boto.ec2.connect_to_region(region)
        reservations = ec2_conn.get_all_reservations()
        myid = subprocess.check_output(['wget', '-qO', '-', 'http://instance-data/latest/meta-data/instance-id'])
        for reservation in reservations:
            for instance in reservation.instances: 
                if instance.state != 'running' or instance.instance_type != 'm4.4xlarge': 
                    continue
                if instance.id != myid:
                    ins.append(instance.public_dns_name) 
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

def run_distributed_aws(command, kwargs_list, stdout=open('/tmp/log', 'ab+'), exclude_nodes=[], use_nodes=None, argument_type='list'):
    hostids = detect_responsive_nodes_aws(exclude_nodes=exclude_nodes, use_nodes=use_nodes)
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
                line = "ssh -i '/home/ubuntu/KeyCompute.pem' ubuntu@%(hostname)s \"%(command)s\" &" % \
                        {'hostname': hostids[i%n_hosts],
                        'command': command % {'first_sec': kwargs_list_as_dict['sections'][fi], 'last_sec': kwargs_list_as_dict['sections'][li]}
                        }
            elif argument_type == 'list':
                # Specify kwargs_str
                line = "ssh -i '/home/ubuntu/KeyCompute.pem' ubuntu@%(hostname)s \"%(command)s\" &" % \
                        {'hostname': hostids[i%n_hosts],
                        'command': command % {'first_sec': kwargs_list_as_dict['sections'][fi], 'last_sec': kwargs_list_as_dict['sections'][li]}
                        }
#            elif argument_type == 'list':
#                # Specify kwargs_str
#                line = "ssh -i '/home/ubuntu/KeyCompute.pem' ubuntu@%(hostname)s \"%(command)s\" &" % \
#                        {'hostname': hostids[i%n_hosts],
#                        'command': command % {'kwargs_str': json.dumps(kwargs_list_as_list[fi:li+1]).replace('"','\\"').replace("'",'\\"')}
#                        }
            elif argument_type == 'list2':
                # Specify {key: list}
                line = "ssh -i '/home/ubuntu/KeyCompute.pem' ubuntu@%(hostname)s \"%(command)s\" &" % \
                        {'hostname': hostids[i%n_hosts],
                        'command': command % {key: json.dumps(vals[fi:li+1]).replace('"','\\"').replace("'",'\\"')
                                            for key, vals in kwargs_list_as_dict.iteritems()}
                        }
            elif argument_type == 'single':
                line = "ssh -i '/home/ubuntu/KeyCompute.pem' ubuntu@%(hostname)s \"%(generic_launcher_path)s \'%(command_template)s\' \'%(kwargs_list_str)s\' \" &" % \
                        {'hostname': hostids[i%n_hosts],
                        'generic_launcher_path': os.environ['REPO_DIR'] + '/utilities/sequential_dispatcher.py',
                        'command_template': command,
                        'kwargs_list_str': json.dumps(kwargs_list_as_list[fi:li+1]).replace('"','\\"').replace("'",'\\"')
                        }
                print line
            else:
                raise Exception('argument_type %s not recognized.' % argument_type)
            temp_f.write(line + '\n')

        temp_f.write('wait\n')
        temp_f.write('echo =================\n')
        temp_f.write('echo all jobs are done!\n')
        temp_f.write('echo =================\n')

    os.chmod(temp_script, 0o777)
    call(temp_script, shell=True, stdout=stdout)

def run_distributed4(command, kwargs_list, stdout=open('/tmp/log', 'ab+'), exclude_nodes=[], use_nodes=None, argument_type='list'):
    """
    There should be only one ssh connection to each node.
    """

    if on_aws:
        hostids = detect_responsive_nodes_aws(exclude_nodes=exclude_nodes, use_nodes=use_nodes)
        auth_str = "-i '/home/ubuntu/KeyCompute.pem' ubuntu"
        print 'Using nodes:', hostids
    else:
        hostids = detect_responsive_nodes(exclude_nodes=exclude_nodes, use_nodes=use_nodes)
        hostids = ['gcn-20-%d.sdsc.edu'%i for i in hostids]
        auth_str = "yuncong"
        print 'Using nodes:', hostids
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
                line = "ssh %(authen_str)s@%(hostname)s \"%(command)s\" &" % \
                        {'authen_str': auth_str,
                        'hostname': hostids[i%n_hosts],
                        'command': command % {'first_sec': kwargs_list_as_dict['sections'][fi], 'last_sec': kwargs_list_as_dict['sections'][li]}
                        }
            elif argument_type == 'list':
                # Specify kwargs_str
                line = "ssh %(authen_str)s@%(hostname)s \"%(command)s\" &" % \
                        {'authen_str': auth_str,
                        'hostname': hostids[i%n_hosts],
                        'command': command % {'kwargs_str': json.dumps(kwargs_list_as_list[fi:li+1]).replace('"','\\"').replace("'",'\\"')}
                        }
            elif argument_type == 'list2':
                # Specify {key: list}
                line = "ssh %(authen_str)s@%(hostname)s \"%(command)s\" &" % \
                        {'authen_str': auth_str,
                        'hostname': hostids[i%n_hosts],
                        'command': command % {key: json.dumps(vals[fi:li+1]).replace('"','\\"').replace("'",'\\"')
                                            for key, vals in kwargs_list_as_dict.iteritems()}
                        }
            elif argument_type == 'single':
                line = "ssh %(authen_str)s@%(hostname)s \"%(generic_launcher_path)s \'%(command_template)s\' \'%(kwargs_list_str)s\' \" &" % \
                        {'authen_str': auth_str,
                        'hostname': hostids[i%n_hosts],
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
