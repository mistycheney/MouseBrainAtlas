#Install vtk 7.1.1 (Cannot use newest version 8 because the window crashes immediately after it launches)
#cd /home/yuncong
#awget https://www.vtk.org/files/release/7.1/vtkpython-7.1.1-Linux-64bit.tar.gz
#tar xfz vtkpython-7.1.1-Linux-64bit.tar.gz
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/yuncong/vtkpython-7.1.1-Linux-64bit/lib/
export PYTHONPATH=$PYTHONPATH:/home/yuncong/vtkpython-7.1.1-Linux-64bit/lib/python2.7/site-packages/


