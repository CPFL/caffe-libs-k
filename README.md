# caffe-libs-k
# Caffe on K Sparc
---
## Dependencies
* BLAS
* CMake >= 2.8
* Google Flags
* Google Log
* Google Test
* LevelDB
* LMDB
* OpenCV >=2.4.10
* Google Protobuf
* Snappy

```
git clone https://github.com/CPFL/caffe-libs-k.git
```

### Environment
* Since we do not have root access, we create custom installation directories.
```
$ mkdir $HOME/install
$ mkdir $HOME/install_x64
```
* Add the following paths to LD_LIBRARY_PATH in `~/.bashrc`. For example:

 `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/FJSVtclang/GM-1.2.0-19/lib:/opt/FJSVtclang/GM-1.2.0-19/lib64:/home/ra000022/a03330/install/lib:/opt/rist/boost-1.47.0/lib:/opt/aics/netcdf/k-serial-noszip/lib
`
3. Add `unset SSH_ASKPASS` to  `~/.bashrc`
4. Add `export PATH=$HOME/install_x64/usr/local/bin:/opt/local/bin:$PATH` to your PATH env var.
5. Source it `$ . ~/.bashrc`

## Dependencies compilation
### BLAS
While on the base directory of the cloned repository, execute:
```
$ cd cblas
$ make
$ cp include/* $HOME/install/usr/local/include/
```
This will install the library automatically in `$HOME/install/lib`.

### CMAKE 3.5
While on the base directory of the cloned repository, execute:
```
$ cd cmake-3.5.0/
$ chmod +x bootstrap
$ chmod +x configure
$ ./bootstrap 
$ make 
$ make DESTDIR=$HOME/install_x64 install
```
* Add `export CMAKE_ROOT=$HOME/install_x64/usr/local/share/cmake-3.5` to `~/.bashrc`
* Also add `$HOME/install_x64/usr/local/bin` to your $PATH and source it `$ . ~/.bashrc`. For example: `export PATH=$HOME/install_x64/usr/local/bin:$PATH`
* If you run `$ cmake /V` you should read: `cmake version 3.5.0`

### GFlags (Requires CMAKE)
While on the base directory of the cloned repository, execute:
```
$ cd gflags-master/
$ mkdir build && cd build
$ cmake .. -DCMAKE_TOOLCHAIN_FILE=../Platforms/fujitsu.cmake
$ make && make DESTDIR=$HOME/install install
```

### GLog
While on the base directory of the cloned repository, execute:
```
$ cd glog-0.3.3_xcomp/
$ ./configure CXX="FCCpx -Xg -pthread" CC="fccpx -Xg -pthread " CXFLAGS="-Kfast" CFLAGS="-Kfast" cross_compiling=yes
$ make && make DESTDIR=$HOME/install install
```

### Google Test (Requires CMAKE)
While on the base directory of the cloned repository, execute:
```
$ cd googletest/googletest/
$ mkdir build && cd build
$ cmake .. -DCMAKE_TOOLCHAIN_FILE=../platforms/fujitsu.cmake
$ make && make DESTDIR=$HOME/install install
```

### Snappy
While on the base directory of the cloned repository, execute:
```
$ cd snappy/
$ ./configure CXX="FCCpx -Xg -pthread" CC="fccpx -Xg -pthread " CXFLAGS="-Kfast" CFLAGS="-Kfast" cross_compiling=yes
$ make && make DESTDIR=$HOME/install install
```

### LevelDB (Requires Snappy)
While on the base directory of the cloned repository, execute:
```
$ cd leveldb/
$ make CXX="FCCpx -Kfast -mt -Xg -fPIC -pthread -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACRO -DSNAPPY  -I$HOME/install/usr/local/include -L$HOME/install/usr/local/lib" CC="fccpx -Kfast -mt -Xg -fPIC -pthread -DSNAPPY -I$HOME/install/usr/local/include -L$HOME/install/usr/local/lib"   TARGET_OS=K PLATFORM_LIBS="-lsnappy"
$ cp include/* $HOME/install/usr/local/include/ -r
$ cp out-shared/* $HOME/install/usr/local/lib/ -r
$ cp out-static/* $HOME/install/usr/local/lib/ -r
```

### lmdb
While on the base directory of the cloned repository, execute:
```
$ cd lmdb/libraries/liblmdb/
$ make CXX="FCCpx -Xg -pthread" CC="fccpx -Xg -pthread " CXFLAGS="-Kfast" CFLAGS="-Kfast" cross_compiling=yes
$ make DESTDIR=$HOME/install install
```

### OpenCV 2.4.12 (Requires CMAKE)
While on the base directory of the cloned repository, execute:
```
$ cd opencv-2.4.12/
$ mkdir build && cd build
$ cmake -DWITH_OPENMP=ON -DWITH_OPENCL=OFF -DWITH_TIFF=ON -DBUILD_TIFF=ON -DWITH_JPEG=ON -DBUILD_JPEG=ON -DWITH_PNG=PNG -DBUILD_PNG=ON -DWITH_JASPER=ON -DBUILD_JASPER=ON -DWITH_ZLIB=ON -DBUILD_ZLIB=ON -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_GTK=OFF -DWITH_WEBP=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_opencv_videostab=OFF -DBUILD_opencv_gpu=OFF -DBUILD_opencv_video=OFF -DCMAKE_TOOLCHAIN_FILE=../platforms/fujitsu/fujitsu.cmake ..
$ make && make install
$ cp install/* $HOME/install/usr/local/ -rf
$ cp ../include/extra/ $HOME/install/usr/local/include/extra -rf
```

### Protobuf
##### Compilation of `protoc` on host (x64)
While on the base directory of the cloned repository, execute:
```
$ cd protobuf/
$ ./autogen.sh
$ ./configure --prefix=$HOME/install_x64/usr/local
$ make && make install
$ make clean
```
##### Compilation of `protoc` for SPARC
```
$ ./configure CXX="FCCpx -Xg -pthread" CC="fccpx -Xg -pthread " CXXFLAGS="-Kfast" CFLAGS="-Kfast" cross_compiling=yes --host=sparc-linux --target=sparc64-linux --with-protoc=$HOME/install_x64/usr/local/bin/protoc --prefix=$HOME/install/usr/local
$ make && make install
$  cp src/google/protobuf/stubs/atomicops_internals_fujitsu_sparc.h $HOME/install/usr/local/include/google/protobuf/stubs/
```


## Caffe
---
While on the base directory of the cloned repository, execute:
```
$ cd fast-rcnn-k/fast-rcnn/caffe-fast-rcnn/
$ make
$ make distribute
```
if you want to change the compilation type from RELEASE to DEBUG, open the `Makefile.config` and remove the comment in the line `DEBUG := 1` near the bottom of the file

To run Caffe, Please set as following:
    $ . /home/system/Env_base
    $ . /opt/aics/hpcu/env.sh
    $ export LD_LIBRARY_PATH=$HOME/install/usr/local/lib:/scratch/ra000022/boost/lib:$LD_LIBRARY_PATH
