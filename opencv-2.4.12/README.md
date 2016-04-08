### OpenCV: Open Source Computer Vision Library

Obtained from [https://github.com/Itseez/opencv/archive/2.4.12.zip](https://github.com/Itseez/opencv/archive/2.4.12.zip)

##How to build

While in the project directory execute:

```
$ mkdir build && cd build
$ cmake -DWITH_OPENMP=ON -DWITH_OPENCL=OFF -DWITH_TIFF=ON -DBUILD_TIFF=ON -DWITH_JPEG=ON -DBUILD_JPEG=ON -DWITH_PNG=PNG -DBUILD_PNG=ON -DWITH_JASPER=ON -DBUILD_JASPER=ON -DWITH_ZLIB=ON -DBUILD_ZLIB=ON -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_GTK=OFF -DWITH_WEBP=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_opencv_videostab=OFF -DBUILD_opencv_gpu=OFF -DBUILD_opencv_video=OFF -DCMAKE_TOOLCHAIN_FILE=../platforms/fujitsu/fujitsu.cmake ..

$ make
```

Extra files added for SPARC can be found inside `include/extra` and in `modules/core/include/opencv2/core/operations.hpp:58`

cross compilation script can be found in `platforms/fujitsu/fujitsu.cmake`
