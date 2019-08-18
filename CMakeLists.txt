cmake_minimum_required (VERSION 3.13)
project (tfbench)

# python path
set(Python3_ROOT_DIR /home/donghok/anaconda3/envs/tfbench)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
set(PYTHON_INCLUDE_DIR ${Python3_INCLUDE_DIRS})
set(PYTHON_LIBRARY ${Python3_LIBRARY})

# dependencies
find_package(Eigen3 REQUIRED NO_MODULE)

add_subdirectory(deps)

# python code
add_subdirectory(python)

# executable
add_executable(test src/test.cpp)
target_link_libraries(test PRIVATE
        pybind11::embed
        yaml-cpp
        Eigen3::Eigen)
target_include_directories(test PUBLIC ${CMAKE_BINARY_DIR}/python)