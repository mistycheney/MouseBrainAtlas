CSD181=/oasis/projects/nsf/csd181/yuncong

export PYTHONPATH=$CSD181/opencv/release/lib/python2.7/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/python/lib:$LD_LIBRARY_PATH:$CSD181/KDU74_Demo_Apps_for_Linux-x86-64_140513:$CSD181/opencv/release/lib
export PATH=/oasis/projects/nsf/csd181/yuncong/virtualenv-1.9.1/yuncongve/bin:$PATH:$CSD181/KDU74_Demo_Apps_for_Linux-x86-64_140513
source /opt/intel/composer_xe_2013_sp1.2.144/bin/compilervars.sh intel64

export GORDON_DATA_DIR=$CSD181/DavidData2014
export LOCAL_DATA_DIR=/home/yuncong/BrainLocal/DavidData_v4
export GORDON_REPO_DIR=$HOME/Brain
export LOCAL_REPO_DIR=$HOME/Brain
