# the name of the target operating system
set(CMAKE_SYSTEM_NAME Linux CACHE STRING "Cross-compiling for Fujitsu Sparc64")
# Set the identification to the same value we would get on the nodes (uname -m)
set(CMAKE_SYSTEM_PROCESSOR "s64fx")

set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS FALSE)

# set the compiler
set(CMAKE_C_COMPILER fccpx)
set(CMAKE_CXX_COMPILER FCCpx)

set(CMAKE_C_FLAGS "-Xg -pthread -Kfast")
set(CMAKE_CXX_FLAGS "-Xg -pthread -std=c++11 -Kfast")

set(FUJITSU_PATH "/opt/FJSVtclang/GM-1.2.0-19/")

include_directories(${FUJITSU_PATH}/include)

# Prevent CMake from adding GNU-specific linker flags (-rdynamic)
# A patch has been submitted to make CMake itself handle this in the future
set(CMAKE_C_COMPILER_ID "Fujitsu" CACHE STRING "Fujitsu C cross-compiler" FORCE)
set(CMAKE_CXX_COMPILER_ID "Fujitsu" CACHE STRING "Fujitsu C++ cross-compiler" FORCE)

# FindOpenMP.cmake does not try -Kopenmp,but the package will try specific
# flags based on the compier ID.
set(OMP_FLAG_Fujitsu "-Kopenmp")

