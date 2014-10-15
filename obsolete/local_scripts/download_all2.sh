
#/bin/bash

[ $# -ne 4 ] && { echo "Usage: ./download_all2.sh <orig_image> <result_name> <local_dir> <ssh_username>
For example, the following command logs in gordon as yuncong, and downloads the relevant data that are computed using parameter set 10, on image 244 in the dataset PMD1305_region0_reduce2. Data is stored in the directory named output.
./download_all2.sh /oasis/projects/nsf/csd181/yuncong/ParthaData/PMD1305_region0_reduce2/PMD1305_region0_reduce2_0244.tif PMD1305_region0_reduce2_0244_param10 output yuncong"; 
if [[ "$BASH_SOURCE" == "$0" ]]; then
  exit 1; 
else
  return;
fi; }


CWD=$(pwd)
data_dir=/oasis/projects/nsf/csd181/yuncong/Brain/data/
result_name=$2
cache_dir=/oasis/scratch/csd181/yuncong/output
result_dir="$cache_dir/$result_name"
ssh_username=$4
gcn="gcn-20-32.sdsc.edu"
tarfile="${result_name}_data.tar.gz"
local_dir="$3/${result_name}_data"
orig_img="$1"
ssh $ssh_username@$gcn /bin/bash << EOF
  if [[ ! -f "${cache_dir}/{$tarfile}" ]]; then
    echo "${orig_img}" > tmp.txt
    echo "${result_dir}/${result_name}_segmentation.npy" >> tmp.txt
    echo "$result_dir/${result_name}_sp_texton_hist_normalized.npy" >> tmp.txt
    echo "$result_dir/${result_name}_sp_dir_hist_normalized.npy" >> tmp.txt
    echo "$result_dir/${result_name}_segmentation.tif" >> tmp.txt
    echo "$result_dir/${result_name}_textonmap.tif" >> tmp.txt
    echo "$result_dir/${result_name}_texton_saliencymap.tif" >> tmp.txt
    cat tmp.txt | xargs tar cfz "$cache_dir/$tarfile" --transform='s|.*/||g'
    rm tmp.txt
  fi
EOF

#if hash globusconnect 2>/dev/null; then
#  echo globusconnect found, using it
# globusconnect may give significant speedup over regular scp (with my home internet)
# Reference for setting up globus connect:
# https://support.globus.org/entries/24078973-Installing-Globus-Connect-Personal-for-Linux-Using-the-Command-Line
# https://support.globus.org/entries/30058643-Using-the-Command-Line-Interface-CLI-
#  globusconnect -start &
#  ssh cli.globusonline.org scp "xsede#gordon:$cache_dir/$tarfile yuncong#macbook:$CWD"
# kill $!
#else
#  echo globusconnect not found, using scp 
  # use regular scp
scp $gcn:"$cache_dir/$tarfile" .
#fi

mkdir -p $local_dir
tar xfz "$tarfile" -C $local_dir
