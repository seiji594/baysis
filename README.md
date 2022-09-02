#  Baysis project
Baysis is a C++ library for Bayesian Sampling and Inference for State space models. It is the accompanying code for the MSc in Data Analytics dissertation at Queen Mary's University London.

### Building
Baysis is a CMake project and can be built with cmake:
>mkdir build\
>cd build\
>cmake ../\
>make install

The `bin` directory contains pre-built binaries for Darwin architecture.

### Dependencies
* Eigen3
* HDF5 10.7

Make sure you set the HDF5_DIR enviroment variable to the path where the CMake config of HDF5 is located (usually share/cmake/hdf5 of the HDF5 CMake installation directory).