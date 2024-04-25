cmake_minimum_required(VERSION 3.5)
project(Bundle_Adjustment)
set(CMAKE_CXX_STANDARD 11)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

add_executable(Test_lm_optimize lm_optimize.cpp)