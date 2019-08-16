cmake_minimum_required (VERSION 3.5)
project (tfbench)

# interpreter path
# TODO no hard coding!
set(PYTHON_EXECUTABLE /home/donghok/anaconda3/envs/tf-benchmark/bin/python)
set(PYTHON_INCLUDE_DIR /home/donghok/anaconda3/envs/tf-benchmark/include/python3.7m)
set(PYTHON_LIBRARY /home/donghok/anaconda3/envs/tf-benchmark/lib/libpython3.7m.so)

# pybind
add_subdirectory(pybind11)
# or
# find_package(pybind11 REQUIRED)

#pybind11_add_module(test src/test.cpp)
add_executable(test src/test.cpp)
target_link_libraries(test PRIVATE pybind11::embed)