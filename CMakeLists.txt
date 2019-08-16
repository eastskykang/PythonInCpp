cmake_minimum_required (VERSION 3.13)
project (tfbench)

# python path
set(Python_ROOT_DIR /home/donghok/anaconda3/envs/tf-benchmark)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

message(STATUS ${Python_EXECUTABLE})
message(STATUS ${Python_INCLUDE_DIRS})
message(STATUS ${Python_LIBRARIES})

set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})
set(PYTHON_INCLUDE_DIR ${Python_INCLUDE_DIRS})
set(PYTHON_LIBRARY ${Python_LIBRARIES})
#
# pybind
add_subdirectory(pybind11)
# or
# find_package(pybind11 REQUIRED)

#pybind11_add_module(test src/test.cpp)
add_executable(test src/test.cpp)
target_link_libraries(test PRIVATE pybind11::embed)