# This script will be executed with ctest -S

# Check that the system name is determined correctly:
set(CMAKE_CMAKE_SYSTEM_NAME "Linux")

if (NOT "${CMAKE_SYSTEM_NAME}" STREQUAL "${CMAKE_CMAKE_SYSTEM_NAME}")
   message(FATAL_ERROR "Error: CMAKE_SYSTEM_NAME is \"${CMAKE_SYSTEM_NAME}\", but should be \"Linux\"")
endif()

# this seems to be necessary, otherwise ctest complains that these
# variables are not set:
set(CTEST_COMMAND "\"/home/ra000022/a03330/caffe/cmake-3.5.0/bin/ctest\"")
set(CTEST_SOURCE_DIRECTORY "/home/ra000022/a03330/caffe/cmake-3.5.0/Tests/CTestScriptMode/")
set(CTEST_BINARY_DIRECTORY "/home/ra000022/a03330/caffe/cmake-3.5.0/Tests/CTestScriptMode/")