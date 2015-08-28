CSD181=/oasis/projects/nsf/csd181/yuncong
CSD395=/oasis/projects/nsf/csd395/yuncong

export DATASET_VERSION=2014

export PYTHONPATH=$CSD181/opencv-2.4.9/release/lib/python2.7/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/python/lib:$LD_LIBRARY_PATH:$CSD181/KDU74_Demo_Apps_for_Linux-x86-64_140513:$CSD181/opencv/release/lib
export PATH=/oasis/projects/nsf/csd181/yuncong/virtualenv-1.9.1/yuncongve/bin:$PATH:$CSD181/KDU74_Demo_Apps_for_Linux-x86-64_140513
#source /opt/intel/composer_xe_2013_sp1.2.144/bin/compilervars.sh intel64

export GORDON_DATA_DIR=$CSD181/DavidData2014tif
export GORDON_REPO_DIR=$CSD395/Brain
export GORDON_RESULT_DIR=$CSD395/DavidData2014results
export GORDON_LABELING_DIR=$CSD181/DavidData2014labelings
export GORDON_SLIDEDATA_DIR=$CSD181/DavidData2014slides
export GORDON_NDPI_DIR=$CSD181/DavidData2014ndpi

export GORDON_NDPISPLIT_PROGRAM=$CSD181/ndpisplit

export MSNAKES_PATH=$GORDON_REPO_DIR/pipeline_scripts/preprocess/morphsnakes

export LOCAL_DATA_DIR=$HOME/DavidData2014tif
export LOCAL_SLIDEDATA_DIR=$HOME/DavidData2014slides/
export LOCAL_SECTIONDATA_DIR=$HOME/DavidData2014sections/
export LOCAL_REPO_DIR=$HOME/Brain
export LOCAL_RESULT_DIR=$HOME/DavidData2014results
export LOCAL_LABELING_DIR=$HOME/DavidData2014labelings


alias sync_result='function _sr(){ cd $LOCAL_RESULT_DIR; rsync -azP --delete --include="*/" --include="00*/$2" --exclude="*" -m  yuncong@gordon.sdsc.edu:$GORDON_RESULT_DIR/$1 . ; cd - ;}; _sr'

alias extract_result='function _er(){ mkdir $3; find $LOCAL_RESULT_DIR/$1 -regex .*/$2 -type f -print0 | xargs -0 cp -t $3; }; _er'

alias killall_gordon_python='for i in {31..38} {41..48}; do ssh gcn-20-$i.sdsc.edu "killall python"; done'
