cmake_minimum_required (VERSION 2.8)

option (USE_GPU "Use CUDA to run simulations on the GPU" ON)
add_subdirectory(Hadr04)
	
if (USE_GPU)
	find_package (CUDA REQUIRED)

	include_directories(/usr/local/cuda/include)
	cuda_add_library (CUDA_G4NeutronHPVector SHARED CUDA_G4NeutronHPVector.h CUDA_G4NeutronHPVector.cu)

	LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
	LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
endif()