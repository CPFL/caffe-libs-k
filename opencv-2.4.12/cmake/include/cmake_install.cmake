# Install script for directory: /home/ra000022/a03330/opencv-2.4.12/include

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

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv" TYPE FILE FILES
    "/home/ra000022/a03330/opencv-2.4.12/include/opencv/cv.h"
    "/home/ra000022/a03330/opencv-2.4.12/include/opencv/cv.hpp"
    "/home/ra000022/a03330/opencv-2.4.12/include/opencv/cvaux.h"
    "/home/ra000022/a03330/opencv-2.4.12/include/opencv/cvaux.hpp"
    "/home/ra000022/a03330/opencv-2.4.12/include/opencv/cvwimage.h"
    "/home/ra000022/a03330/opencv-2.4.12/include/opencv/cxcore.h"
    "/home/ra000022/a03330/opencv-2.4.12/include/opencv/cxcore.hpp"
    "/home/ra000022/a03330/opencv-2.4.12/include/opencv/cxeigen.hpp"
    "/home/ra000022/a03330/opencv-2.4.12/include/opencv/cxmisc.h"
    "/home/ra000022/a03330/opencv-2.4.12/include/opencv/highgui.h"
    "/home/ra000022/a03330/opencv-2.4.12/include/opencv/ml.h"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/include/opencv2/opencv.hpp")
endif()

