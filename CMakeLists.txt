# https://stackoverflow.com/questions/51907755/building-a-pybind11-module-with-cpp-and-cuda-sources-using-cmake
# https://stackoverflow.com/questions/53859271/cmake-cannot-find-cuda-could-not-find-cmake-module-file-cmakedeterminecudacom

cmake_minimum_required(VERSION 3.5)
# set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(my_impl)
# include(FindCUDA)
set(CMAKE_CXX_STANDARD 14)

find_package(CUDA REQUIRED)
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30)

# set(PYBIND11_CPP_STANDARD -std=c++11)
add_subdirectory(pybind11)

cuda_add_library(my_impl_cu SHARED gaussian_blur.cu )

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -fPIC") 

add_library(my_impl MODULE my_impl.cpp)

set_target_properties(my_impl PROPERTIES 
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

target_link_libraries(my_impl PRIVATE my_impl_cu)
target_link_libraries(my_impl PRIVATE pybind11::module)

# find_package(CUDA REQUIRED)
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -ccbin /usr/bin/clang )

# file( GLOB  hdr *.hpp *.h )
# file( GLOB  cu  *.cu)

# CUDA_ADD_EXECUTABLE(helper ${helper_files} ${hdr} ${cu})


# add_library(my_impl MODULE my_impl.cpp)
# cuda_add_library(my_impl my_impl.cpp gaussian_blur.cu)

# target_include_directories(my_impl PUBLIC ${CUDA_INCLUDE_DIRS})

# pybind11_add_module(my_impl my_impl.cpp)

# set_property(TARGET my_impl PROPERTY CXX_STANDARD 11)

# target_link_libraries(my_impl PRIVATE pybind11::module)
