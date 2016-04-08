# CMake generated Testfile for 
# Source directory: /home/ra000022/a03330/caffe/cmake-3.5.0/Utilities/cmcurl
# Build directory: /home/ra000022/a03330/caffe/cmake-3.5.0/Utilities/cmcurl
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(curl "LIBCURL" "http://open.cdash.org/user.php")
subdirs(lib)
