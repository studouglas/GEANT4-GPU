cmake_minimum_required (VERSION 2.8)
project(MainExe)

find_package (CUDA REQUIRED)

include_directories(/usr/local/cuda/include)

cuda_add_library (TestingCudaSquareArrayLib DeviceMain.h DeviceMain.cu)
add_executable (MainExe HostMain.cpp)

LIST(APPEND CMAKE_CXX_FLAGS "-std=c++0x -O3 -ffast-math -Wall")

LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)

target_link_libraries(MainExe TestingCudaSquareArrayLib /usr/local/cuda/lib/libcudart_static.a)
