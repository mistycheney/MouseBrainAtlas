for F in $(cat filelist.txt) ; do
  wget $F
done;
