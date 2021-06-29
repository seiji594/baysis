cmake_minimum_required(VERSION 3.19)
project(Baysis)

set(CMAKE_CXX_STANDARD 14)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(Baysis experiments/main.cpp)
target_link_libraries (Baysis Eigen3::Eigen)
