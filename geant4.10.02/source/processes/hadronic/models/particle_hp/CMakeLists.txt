#------------------------------------------------------------------------------
# CMakeLists.txt
# Module : G4had_par_hp
# Package: Geant4.src.G4processes.G4hadronic.G4hadronic_models.G4had_par_hp
#
# CMakeLists.txt for building a single granular library.
#
# Generated on : 24/9/2010
#
# $Id: CMakeLists.txt 76349 2013-11-08 15:50:19Z tkoi $
#
#------------------------------------------------------------------------------

if(GEANT4_BUILD_GRANULAR_LIBS)
    include(Geant4MacroLibraryTargets)
    GEANT4_GRANULAR_LIBRARY_TARGET(COMPONENT sources.cmake)
endif()

if(GEANT4_ENABLE_CUDA)
    set(ENABLE_CUDA 1)
    message(STATUS "CUDA Acceleration ENABLED")
else()
    set(ENABLE_CUDA 0)
    message(STATUS "CUDA Acceleration DISABLED")
endif()
configure_file("${PROJECT_SOURCE_DIR}/source/externals/cuda/CUDAGPU.hh"
     "${PROJECT_SOURCE_DIR}/source/processes/hadronic/models/particle_hp/include/CUDAGPU.hh")
