# tfbench

## Dependencies

- Python
    - python >= 3.5
    - tensorflow == 1.14.0
- C++
    - eigen3

## Steps

- Install eigen by apt-get
```bash
$ sudo apt-get install libeigen3-dev
```
- Strongly recommend to use conda or virtualenv.
```bash
$ conda env create -n <ENV NAME> python=3.5
$ conda activate <ENV NAME>
```
- Install tensorflow by pip 
```bash
# option1: GPU
$ pip install tensorflow-gpu
# option2: CPU
$ pip install tensorflow-gpu  
```
- Build by cmake 
```bash
$ mkdir build && cd build 
$ cmake ../ ```-DPython3_ROOT_DIR=<YOUR PYTHON3 INTERPRETER PATH>```
```
- Run test