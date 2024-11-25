cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(cmake-c-examples C)

find_package(MeshLib CONFIG REQUIRED)

set(EXAMPLES
  GlobalRegistration
  MeshBoolean
  MeshDecimate
  MeshFillHole
  MeshLoadSave
  MeshOffset
  MeshResolveDegenerations
)

foreach(TARGET ${EXAMPLES})
  add_executable(${TARGET} ${TARGET}.dox.c)
  target_include_directories(${TARGET} PUBLIC ${MESHLIB_INCLUDE_DIR} ${MESHLIB_THIRDPARTY_INCLUDE_DIR})
  target_link_libraries(${TARGET} PRIVATE MeshLib::MRMeshC)
endforeach()
