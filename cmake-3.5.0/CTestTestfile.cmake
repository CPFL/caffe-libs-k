# CMake generated Testfile for 
# Source directory: /home/ra000022/a03330/caffe/cmake-3.5.0
# Build directory: /home/ra000022/a03330/caffe/cmake-3.5.0
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
include("/home/ra000022/a03330/caffe/cmake-3.5.0/Tests/EnforceConfig.cmake")
add_test(SystemInformationNew "/home/ra000022/a03330/caffe/cmake-3.5.0/bin/cmake" "--system-information" "-G" "Unix Makefiles")
subdirs(Source/kwsys)
subdirs(Utilities/KWIML)
subdirs(Utilities/cmzlib)
subdirs(Utilities/cmcurl)
subdirs(Utilities/cmcompress)
subdirs(Utilities/cmbzip2)
subdirs(Utilities/cmliblzma)
subdirs(Utilities/cmlibarchive)
subdirs(Utilities/cmexpat)
subdirs(Utilities/cmjsoncpp)
subdirs(Source/CursesDialog/form)
subdirs(Source)
subdirs(Utilities)
subdirs(Tests)
subdirs(Auxiliary)
