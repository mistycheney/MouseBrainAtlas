#! /bin/bash

CSD181=/oasis/projects/nsf/csd181/yuncong
CSD395=/oasis/projects/nsf/csd395/yuncong

export DATASET_VERSION=2015

if [[ $(hostname) = "yuncong-MacbookPro" ]]; then 
        export DATA_DIR=$HOME/CSHL_data_processed
	export REPO_DIR=$HOME/Brain
	#export RESULT_DIR=$HOME/CSHL_data_results
	export LABELING_DIR=$HOME/CSHL_data_labelings_losslessAlignCropped
	export MXNET_DIR=$HOME/mxnet
	export MODEL_DIR=$HOME/mxnet_models
	export PYTHONPATH=$MXNET_DIR/python:$PYTHONPATH
	export ELASTIX_DIR=$HOME/elastix_linux64_v4.7
	export OPENCV_DIR=$HOME/opencv-2.4.11
        export OPENCV_LIBDIR=$OPENCV_DIR/release/lib/python2.7/dist-packages


elif [[ $(hostname) = "yuncong-Precision-WorkStation-T7500" ]]; then
        export DATA_DIR=$HOME/CSHL_data_processed
        export REPO_DIR=$HOME/Brain
        #export RESULT_DIR=$HOME/CSHL_data_results
        export LABELING_DIR=$HOME/CSHL_data_labelings_losslessAlignCropped
        export MXNET_DIR=$HOME/mxnet
        export MODEL_DIR=$HOME/mxnet_models
        export OPENCV_DIR=$HOME/opencv
        export OPENCV_LIBDIR=$OPENCV_DIR/release/lib/python2.7/dist-packages
        export CAFFE_DIR=$HOME/caffe-rc3

else
	CSD181=/oasis/projects/nsf/csd181/yuncong
	CSD395=/oasis/projects/nsf/csd395/yuncong
	export DATA_DIR=$CSD395/CSHL_data_processed
	export REPO_DIR=$CSD395/Brain
	export PIPELINE_SCRIPT_DIR=$GORDON_REPO_DIR/pipeline
	#export RESULT_DIR=$CSD395/CSHL_data_results
	export LABELING_DIR=$CSD395/CSHL_data_labelings_losslessAlignCropped
	export MXNET_DIR=$CSD395/mxnet
	export MODEL_DIR=$CSD395/jiaxuzhu/model/

	# openblas
	export LD_LIBRARY_PATH=$CSD395/OpenBLAS-release/lib:$LD_LIBRARY_PATH
	
	#export PATH=/oasis/projects/nsf/csd181/yuncong/virtualenv-1.9.1/yuncongve/bin:$PATH
	export LD_LIBRARY_PATH=/opt/python/lib:$LD_LIBRARY_PATH
	
	# elastix
	export ELASTIX_DIR=$CSD395/elastix_linux64_v4.7
	
	# geos-svn
	export LD_LIBRARY_PATH=$HOME/csd395/geos-svn/release/lib/:$LD_LIBRARY_PATH
	
	export OPENCV_DIR=$HOME/csd181/opencv-2.4.9
	export OPENCV_LIBDIR=$OPENCV_DIR/release/lib/python2.7/site-packages
fi

# mxnet
export PYTHONPATH=$MXNET/python:$PYTHONPATH
# elastix
export ELASTIX_BIN=$ELASTIX_DIR/bin/elastix
# opencv
export LD_LIBRARY_PATH=$OPENCV_DIR/release/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$OPENCV_LIBDIR:$PYTHONPATH
export PKG_CONFIG_PATH=$OPENCV_DIR/release/lib/pkgconfig/:$PKG_CONFIG_PATH
# caffe
export PYTHONPATH=$CAFFE_DIR/python:$PYTHONPATH

alias sync_result='function _sr(){ cd $LOCAL_RESULT_DIR; rsync -azP --delete --include="*/" --include="0*/*$2*" --exclude="*" -m  yuncong@gordon.sdsc.edu:$GORDON_RESULT_DIR/$1 . ; cd - ;}; _sr'
alias extract_result='function _er(){ mkdir $3; find $LOCAL_RESULT_DIR/$1 -regex .*/.*$2.* -type f -print0 | xargs -0 cp -t $3; }; _er'
alias killall_gordon_python='for i in {31..38} {41..48}; do ssh -o ConnectTimeout=10  gcn-20-$i.sdsc.edu "pkill -9 python"; done'
alias apply_all_gcn='function _aag(){ for i in {31..38} {41..48}; do ssh gcn-20-$i.sdsc.edu $1; done; }; _aag'

