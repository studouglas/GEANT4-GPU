#------------------------------------------------------------------------------
# CMakeLists.txt
# Module : G4had_neu_hp
# Package: Geant4.src.G4processes.G4hadronic.G4hadronic_models.G4had_neu_hp
#
# CMakeLists.txt for building a single granular library.
#
# Generated on : 24/9/2010
#
# $Id: CMakeLists.txt 81879 2014-06-06 12:32:01Z gcosmo $
#
#------------------------------------------------------------------------------

#add_subdirectory(cuda)

if(GEANT4_BUILD_GRANULAR_LIBS)
    include(Geant4MacroLibraryTargets)
    GEANT4_GRANULAR_LIBRARY_TARGET(COMPONENT sources.cmake)
endif()

