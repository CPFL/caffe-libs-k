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
4. Source it `$ . ~/.bashrc`

## Dependencies compilation
### BLAS
While on the base directory of the cloned repository, execute:
```
$ cd cblas
$ make
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
$ ./configure CXX="FCCpx -Xg -pthread" CC="fccpx -Xg -pthread " CXFLAGS="-Kfast" CFLAGS="-Kfast" cross_compiling=yes
$ make
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
```
---
### TODO Protobuf x64 and SPARC
---
