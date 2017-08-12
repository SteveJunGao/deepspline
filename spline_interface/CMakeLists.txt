cmake_minimum_required(VERSION 3.5)
project(spline)

set( CMAKE_EXPORT_COMPILE_COMMANDS ON )
set (CMAKE_CXX_STANDARD 11)

add_executable(spline main.cpp bspline.h bspline.cpp)
