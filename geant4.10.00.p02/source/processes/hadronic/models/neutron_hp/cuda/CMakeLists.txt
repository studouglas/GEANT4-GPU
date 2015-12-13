cmake_minimum_required (VERSION 2.8)

option (GEANT4_ENABLE_CUDA "Use CUDA to run simulations in parallel on the GPU" ON)
	
if (GEANT4_ENABLE_CUDA)
	find_package (CUDA REQUIRED)

	include_directories(/usr/local/cuda/include)
	cuda_add_library (CUDA_G4NeutronHPVector SHARED CUDA_G4NeutronHPVector.h CUDA_G4NeutronHPVector.cu)

	LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
	LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)

	target_link_libraries(CUDA_G4NeutronHPVector ${GEANT4_LIBRARIES})
endif()