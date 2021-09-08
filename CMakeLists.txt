cmake_minimum_required(VERSION 3.18)
project(diss)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)
set(HDF5_USE_STATIC_LIBRARIES OFF)
find_package(HDF5 COMPONENTS HL REQUIRED)

set(READDY_BUILD_SHARED_COMBINED ON)
add_subdirectory(lib/readdy)
add_subdirectory(src)
