cmake_minimum_required(VERSION 3.2)
project(CVPR2018)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-Wall -O3")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

# OpenCV Settings
SET(OpenCV_DIR "/home/siyuan/libraries/opencv/opencv-3.1.0/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Boost Settings
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost REQUIRED COMPONENTS system filesystem serialization)
include_directories(${Boost_INCLUDE_DIRS})

# Ubuntu 16.04 eigen3
include_directories("/usr/include/eigen3")

## GTest Settings
#find_package(GTest REQUIRED)
#include_directories(${GTEST_INCLUDE_DIRS})

# CGAL for polygon intersection
find_package(CGAL QUIET COMPONENTS Core )
if ( CGAL_FOUND )
    include( ${CGAL_USE_FILE} )
    include( CGAL_CreateSingleSourceCGALProgram )
    # create_single_source_cgal_program( "main.cpp" )
else()
    message(STATUS "This program requires the CGAL library, and will not be compiled.")
endif()

# OpenMP settings
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Include path for the current folder
set(BASEPATH ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${BASEPATH})

# Source files
file(GLOB SOURCE_FILES *.h *.cpp)
file(GLOB_RECURSE LIB_SOURCE_FILES lib/*.h lib/*.hpp lib/*.cc lib/*.cpp)
set(SOURCE_FILES ${SOURCE_FILES} ${LIB_SOURCE_FILES})

# Add executable and link target
add_executable(cpp ${SOURCE_FILES})
target_link_libraries(cpp ${OpenCV_LIBS} ${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY} ${Boost_SERIALIZATION_LIBRARY})

