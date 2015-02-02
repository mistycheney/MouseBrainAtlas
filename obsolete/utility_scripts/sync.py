import subprocess


def sync(stack, resol, section, )

	remote_file = '/home/yuncong/DavidData/%(stack)s/%(resol)s/%(slice)s/pipelineResults/%(obj)s'%d

	cmd = 'scp gcn-20-33.sdsc.edu:%s %s'%(remote_file, local_dir)
	print cmd
	subprocess.call(cmd, shell=True)

rsync -azP --exclude=*2* gcn:/oasis/projects/nsf/csd181/yuncong/DavidData2014v4/RS141/x5/0006 .