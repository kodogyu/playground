cmake_minimum_required(VERSION 3.16)
project(foo)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(foo SHARED src/foo.cpp)