# CMake generated Testfile for 
# Source directory: /home/ra000022/a03330/caffe/cmake-3.5.0/Tests/FindThreads
# Build directory: /home/ra000022/a03330/caffe/cmake-3.5.0/Tests/FindThreads
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(FindThreads.C-only "/home/ra000022/a03330/caffe/cmake-3.5.0/bin/ctest" "--build-and-test" "/home/ra000022/a03330/caffe/cmake-3.5.0/Tests/FindThreads/C-only" "/home/ra000022/a03330/caffe/cmake-3.5.0/Tests/FindThreads/C-only" "--build-generator" "Unix Makefiles" "--build-project" "FindThreads_C-only" "--build-options" "-DCMAKE_MAKE_PROGRAM:FILEPATH=/usr/bin/gmake" "--test-command" "/home/ra000022/a03330/caffe/cmake-3.5.0/bin/ctest" "-V")
add_test(FindThreads.CXX-only "/home/ra000022/a03330/caffe/cmake-3.5.0/bin/ctest" "--build-and-test" "/home/ra000022/a03330/caffe/cmake-3.5.0/Tests/FindThreads/CXX-only" "/home/ra000022/a03330/caffe/cmake-3.5.0/Tests/FindThreads/CXX-only" "--build-generator" "Unix Makefiles" "--build-project" "FindThreads_CXX-only" "--build-options" "-DCMAKE_MAKE_PROGRAM:FILEPATH=/usr/bin/gmake" "--test-command" "/home/ra000022/a03330/caffe/cmake-3.5.0/bin/ctest" "-V")
