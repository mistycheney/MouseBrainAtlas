import paramiko
host = 'gcn-20-33.sdsc.edu'
username = 'yuncong'
password = 'One2Three4'

gcn = paramiko.SSHClient()
gcn.set_missing_host_key_policy(paramiko.AutoAddPolicy())
gcn.connect(host, username=username, password=password)

(stdin, stdout, stderr) = gcn.exec_command('rm test')
print("\nstdout is:\n" + stdout.read() + "\nstderr is:\n" + stderr.read())