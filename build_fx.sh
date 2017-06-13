#!/bin/bash

mkfifo fifo
CUR_DIR=$PWD
trap 'rm $CUR_DIR/fifo' EXIT
tee build.log < fifo &
exec > fifo 2>&1

mkdir -p $HOME/install
mkdir -p $HOME/install_x64

###Cblas
cd cblas
if [ ! -e $HOME/install/usr/local/lib/libcblas_K.a ]; then
  make
  cp include/* $HOME/install/usr/local/include/
fi

###cmake
cd ../cmake-3.5.0/
if [ ! -e $HOME/install_x64/usr/local/bin/cmake ]; then
  chmod +x bootstrap
  chmod +x configure
  ./bootstrap 
  make -j8
  make DESTDIR=$HOME/install_x64 install
fi

### GFlags (Requires CMAKE)
cd ../gflags-master/
if [ ! -e $HOME/install/usr/local/lib/libgflags.a ]; then
    mkdir -p build && cd build
    cmake .. -DCMAKE_TOOLCHAIN_FILE=../Platforms/fujitsu.cmake
    cmake .. -DCMAKE_TOOLCHAIN_FILE=../Platforms/fujitsu.cmake
    make -j8 && make DESTDIR=$HOME/install install
    cd ..
fi

### GLog
cd ../glog-0.3.3_xcomp/
if [ ! -e $HOME/install/usr/local/lib/libglog.a ]; then
  make -j8 && make DESTDIR=$HOME/install install
fi

### Google Test (Requires CMAKE)
cd ../googletest/googletest/
mkdir -p build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=../platforms/fujitsu.cmake
make -j8 && make DESTDIR=$HOME/install install

### Snappy
cd ../../../snappy/
./configure CXX="FCCpx -Xg -pthread" CC="fccpx -Xg -pthread " CXFLAGS="-Kfast" CFLAGS="-Kfast" cross_compiling=yes
make -j8 && make DESTDIR=$HOME/install install

### LevelDB (Requires Snappy)
cd ../leveldb/
make CXX="FCCpx -Kfast -mt -Xg -fPIC -pthread -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACRO -DSNAPPY  -I$HOME/install/usr/local/include -L$HOME/install/usr/local/lib" CC="fccpx -Kfast -mt -Xg -fPIC -pthread -DSNAPPY -I$HOME/install/usr/local/include -L$HOME/install/usr/local/lib"   TARGET_OS=K PLATFORM_LIBS="-lsnappy"
cp include/* $HOME/install/usr/local/include/ -r
cp out-shared/* $HOME/install/usr/local/lib/ -r
cp out-static/* $HOME/install/usr/local/lib/ -r

### lmdb
cd ../lmdb/libraries/liblmdb/
make -j8 CXX="FCCpx -Xg -pthread" CC="fccpx -Xg -pthread " CXFLAGS="-Kfast" CFLAGS="-Kfast" cross_compiling=yes
make DESTDIR=$HOME/install install

### OpenCV 2.4.12 (Requires CMAKE)
cd ../../../opencv-2.4.12/
mkdir -p build && cd build
cmake -DWITH_OPENMP=ON -DWITH_OPENCL=OFF -DWITH_TIFF=ON -DBUILD_TIFF=ON -DWITH_JPEG=ON -DBUILD_JPEG=ON -DWITH_PNG=PNG -DBUILD_PNG=ON -DWITH_JASPER=ON -DBUILD_JASPER=ON -DWITH_ZLIB=ON -DBUILD_ZLIB=ON -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_GTK=OFF -DWITH_WEBP=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_opencv_videostab=OFF -DBUILD_opencv_gpu=OFF -DBUILD_opencv_video=OFF -DBUILD_opencv_flann=OFF -DBUILD_opencv_objdetect=OFF -DCMAKE_TOOLCHAIN_FILE=../platforms/fujitsu/fujitsu.cmake ..
make -j8 && make install
cp install/* $HOME/install/usr/local/ -rf
cp ../include/extra/ $HOME/install/usr/local/include/extra -rf

### Protobuf
##### Compilation of `protoc` on host (x64)
cd ../../protobuf/
./autogen.sh
./configure --prefix=$HOME/install_x64/usr/local
make -j8 && make install
make clean

##### Compilation of `protoc` for SPARC
#./configure CXX="FCCpx -Xg -pthread" CC="fccpx -Xg -pthread " CXXFLAGS="-Kfast" CFLAGS="-Kfast" cross_compiling=yes --host=sparc-linux --target=sparc64-linux --with-protoc=$HOME/install_x64/usr/local/bin/protoc --prefix=$HOME/install/usr/local
cp Makefile_SPARC Makefile
cp src/Makefile_SPARC src/Makefile
cp libtool_SPARC libtool
make -j8 && make install
cp -p src/google/protobuf/stubs/atomicops_internals_fujitsu_sparc.h $HOME/install/usr/local/include/google/protobuf/stubs/


## Caffe
#cd ../fast-rcnn-k/fast-rcnn/caffe-fast-rcnn/
#make -j8
