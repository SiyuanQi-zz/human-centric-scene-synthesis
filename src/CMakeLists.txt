cmake_minimum_required(VERSION 3.2)
project(CVPR2018)

set(CMAKE_CXX_FLAGS "-Wall -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_subdirectory(cpp)
