# Install script for directory: /home/ra000022/a03330/opencv-2.4.12/modules/flann

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/ra000022/a03330/opencv-2.4.12/cmake/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "libs")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/home/ra000022/a03330/opencv-2.4.12/cmake/modules/flann/CMakeFiles/CMakeRelink.dir/libopencv_flann.so.2.4.12"
    "/home/ra000022/a03330/opencv-2.4.12/cmake/modules/flann/CMakeFiles/CMakeRelink.dir/libopencv_flann.so.2.4"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_flann.so.2.4.12"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_flann.so.2.4"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/ra000022/a03330/opencv-2.4.12/cmake/modules/flann/CMakeFiles/CMakeRelink.dir/libopencv_flann.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_flann.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_flann.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_flann.so")
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/flann" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/modules/flann/include/opencv2/flann/flann.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/flann" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/modules/flann/include/opencv2/flann/flann_base.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/flann" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/modules/flann/include/opencv2/flann/miniflann.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/flann" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/modules/flann/include/opencv2/flann/all_indices.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/flann" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/modules/flann/include/opencv2/flann/allocator.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/flann" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/modules/flann/include/opencv2/flann/any.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/flann" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/modules/flann/include/opencv2/flann/autotuned_index.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/flann" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/modules/flann/include/opencv2/flann/composite_index.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/flann" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/modules/flann/include/opencv2/flann/config.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/flann" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/modules/flann/include/opencv2/flann/defines.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/flann" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/modules/flann/include/opencv2/flann/dist.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/flann" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/modules/flann/include/opencv2/flann/dummy.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/flann" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/modules/flann/include/opencv2/flann/dynamic_bitset.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/flann" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/modules/flann/include/opencv2/flann/general.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/flann" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/modules/flann/include/opencv2/flann/ground_truth.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/flann" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/modules/flann/include/opencv2/flann/hdf5.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/flann" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/modules/flann/include/opencv2/flann/heap.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/flann" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/modules/flann/include/opencv2/flann/hierarchical_clustering_index.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/flann" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/modules/flann/include/opencv2/flann/index_testing.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/flann" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/modules/flann/include/opencv2/flann/kdtree_index.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/flann" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/modules/flann/include/opencv2/flann/kdtree_single_index.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/flann" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/modules/flann/include/opencv2/flann/kmeans_index.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/flann" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/modules/flann/include/opencv2/flann/linear_index.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/flann" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/modules/flann/include/opencv2/flann/logger.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/flann" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/modules/flann/include/opencv2/flann/lsh_index.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/flann" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/modules/flann/include/opencv2/flann/lsh_table.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/flann" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/modules/flann/include/opencv2/flann/matrix.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/flann" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/modules/flann/include/opencv2/flann/nn_index.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/flann" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/modules/flann/include/opencv2/flann/object_factory.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/flann" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/modules/flann/include/opencv2/flann/params.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/flann" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/modules/flann/include/opencv2/flann/random.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/flann" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/modules/flann/include/opencv2/flann/result_set.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/flann" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/modules/flann/include/opencv2/flann/sampling.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/flann" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/modules/flann/include/opencv2/flann/saving.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/flann" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/modules/flann/include/opencv2/flann/simplex_downhill.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/flann" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/modules/flann/include/opencv2/flann/timer.h")
endif()

