# Install script for directory: /home/ra000022/a03330/opencv-2.4.12/doc

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

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "docs")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/OpenCV/doc" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/doc/opencv-logo.png")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "docs")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/OpenCV/doc" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/doc/opencv-logo2.png")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "docs")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/OpenCV/doc" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/doc/opencv-logo-white.png")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "docs")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/OpenCV/doc" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/doc/opencv.ico")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "docs")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/OpenCV/doc" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/doc/pattern.png")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "docs")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/OpenCV/doc" TYPE FILE FILES "/home/ra000022/a03330/opencv-2.4.12/doc/acircles_pattern.png")
endif()

