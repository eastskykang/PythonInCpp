# tfbench

## Dependencies
- python >= 3.5
- tensorflow == 1.14.0

## Steps

- Strongly recommend to use conda or virtualenv.
```sh
$ conda env create -n <ENV NAME> python=3.5
$ conda activate <ENV NAME>
```
- Install tensorflow by pip 
```sh
# option1: GPU
$ pip install tensorflow-gpu
# option2: CPU
$ pip install tensorflow-gpu  
```
- Build by cmake 
```sh
$ mkdir build && cd build 
$ cmake ../ ```-DPython3_ROOT_DIR=<YOUR PYTHON3 INTERPRETER PATH>```
```
- Run test