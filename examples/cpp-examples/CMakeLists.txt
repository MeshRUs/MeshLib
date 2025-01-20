cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(cpp-examples CXX)

find_package(MeshLib CONFIG REQUIRED)
find_package(TBB REQUIRED)

set(EXAMPLES
  FreeFormDeformation
  GlobalRegistration
  LaplacianDeformation
  MeshBoolean
  MeshDecimate
  MeshExport
  MeshExtrude
  MeshFillHole
  MeshFromText
  MeshICP
  MeshLoadSave
  MeshModification
  MeshOffset
  MeshResolveDegenerations
  MeshStitchHole
  NoiseDenoise
  Triangulation
)

foreach(TARGET ${EXAMPLES})
  add_executable(${TARGET} ${TARGET}.dox.cpp)
  target_include_directories(${TARGET} PUBLIC ${MESHLIB_INCLUDE_DIR} ${MESHLIB_THIRDPARTY_INCLUDE_DIR})
  target_link_libraries(${TARGET} PRIVATE MeshLib::MRMesh TBB::tbb)
  target_link_directories(${TARGET} PUBLIC ${MESHLIB_THIRDPARTY_LIB_DIR})
endforeach()
# MeshFromText additionally depends on MRSymbolMesh
target_link_libraries(MeshFromText PRIVATE MeshLib::MRSymbolMesh)
# MeshOffset and Triangulation additionally depend on MRVoxels
target_link_libraries(MeshOffset PRIVATE MeshLib::MRVoxels)
target_link_libraries(Triangulation PRIVATE MeshLib::MRVoxels)
# MeshLoadSave additionally depends on MRIOExtras
target_link_libraries(MeshLoadSave PRIVATE MeshLib::MRIOExtras)

find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)

add_executable(MRBasedMainFunc MRBasedMainFunc.dox.cpp)
target_compile_definitions(MRBasedMainFunc PRIVATE SPDLOG_FMT_EXTERNAL)
target_include_directories(MRBasedMainFunc PUBLIC ${MESHLIB_INCLUDE_DIR} ${MESHLIB_THIRDPARTY_INCLUDE_DIR})
target_link_libraries(MRBasedMainFunc PRIVATE MeshLib::MRMesh MeshLib::MRViewer fmt::fmt spdlog::spdlog)
target_link_directories(MRBasedMainFunc PUBLIC ${MESHLIB_THIRDPARTY_LIB_DIR})
