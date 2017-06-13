unset SSH_ASKPASS
. /home/system/Env_base
#. /opt/aics/hpcu/env.sh
export PATH=$HOME/install_x64/usr/local/bin:/opt/local/bin:$PATH
export LD_LIBRARY_PATH=$HOME/install/usr/local/lib:/scratch/ra000022/boost/lib:$PWD/fast-rcnn-k/fast-rcnn/caffe-fast-rcnn/build/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/FJSVtclang/GM-1.2.0-19/lib:/opt/FJSVtclang/GM-1.2.0-19/lib64:$HOME/install/lib:/opt/aics/netcdf/k-serial-noszip/lib

export CMAKE_ROOT=$HOME/install_x64/usr/local/share/cmake-3.5
