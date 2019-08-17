cmake_minimum_required (VERSION 3.13)
project (tfbench)

# python path
set(Python3_ROOT_DIR /home/donghok/anaconda3/envs/tfbench)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

#message(STATUS ${Python3_EXECUTABLE})
#message(STATUS ${Python3_INCLUDE_DIRS})
#message(STATUS ${Python3_LIBRARIES})

set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
set(PYTHON_INCLUDE_DIR ${Python3_INCLUDE_DIRS})
set(PYTHON_LIBRARIES ${Python3_LIBRARIES})

# pybind
add_subdirectory(pybind11)
# or
# find_package(pybind11 REQUIRED)

add_executable(test src/test.cpp)
target_link_libraries(test PRIVATE pybind11::embed)