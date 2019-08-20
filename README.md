# tfbench

## Dependencies

- Python
    - python >= 3.5
    - tensorflow == 1.14.0
- C++
    - Boost
    - Eigen3

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
- Install tensorflow  
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
```bash
$ ./test --yaml_path=<YOUR TEST CONFIGURATION YAML PATH> 
```


<!---
## Docker 

- (optional) install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) for GPU support 


```bash
$ docker pull tensorflow/tensorflow:nightly-devel-gpu-py3
```

```bash
$ docker run --runtime=nvidia -it -w /tensorflow -v $PWD:/mnt -e HOST_PERMS="$(id -u):$(id -g)" \
    tensorflow/tensorflow:nightly-devel-gpu-py3 bash
```
-->