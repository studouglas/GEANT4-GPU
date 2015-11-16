cmake_minimum_required (VERSION 2.8)

find_package (CUDA REQUIRED)

include_directories(/usr/local/cuda/include)

cuda_add_library (CUDA_G4NeutronHPVector SHARED CUDA_G4NeutronHPVector.h CUDA_G4NeutronHPVector.cu)

#LIST(APPEND CMAKE_CXX_FLAGS "-std=c++0x -O3 -ffast-math -Wall")
LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)

#target_link_libraries(Geant4 CUDA_G4NeutronHPVector /usr/local/cuda/lib/libcudart_static.a)
