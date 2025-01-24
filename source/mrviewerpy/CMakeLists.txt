cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD ${MR_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(mrviewerpy CXX)

file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.h")

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

set_target_properties(${PROJECT_NAME} PROPERTIES
  PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/meshlib"
)

if(APPLE)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
  target_link_libraries(${PROJECT_NAME} PRIVATE
    MRViewer
    MRMesh
    MRPython
    fmt
    glfw
    spdlog
    pybind11::module
    pybind11nonlimitedapi_stubs
    ${CMAKE_DL_LIBS}
  )
ELSE()
  target_link_libraries(${PROJECT_NAME} PRIVATE
    MRViewer
    MRMesh
    MRPython
    fmt
    glfw
    spdlog
    tbb
    pybind11nonlimitedapi_stubs
    ${CMAKE_DL_LIBS}
  )
ENDIF()

# `dladdr` in our patched `PYBIND11_MODULE(...)` needs this.
IF(NOT WIN32)
  target_compile_definitions(${PROJECT_NAME} PRIVATE
    _GNU_SOURCE
  )
ENDIF()

# Don't emit undefined references. We need this because we don't want to link to a specific Python version.
# The flag we're removing here is something we only set on Linux, in `CompilerOptions.cmake`.
string(REPLACE "-Wl,-z,defs" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")

IF(MESHLIB_BUILD_VOXELS)
  IF(APPLE)
    target_link_libraries(${PROJECT_NAME} PRIVATE
      MRVoxels
      OpenVDB
    )
  ELSE()
    target_link_libraries(${PROJECT_NAME} PRIVATE
      MRVoxels
      openvdb
    )
  ENDIF()
ELSE()
  target_compile_definitions(${PROJECT_NAME} PRIVATE MESHLIB_NO_VOXELS)
ENDIF()

install(TARGETS ${PROJECT_NAME} DESTINATION "${MR_PY_LIB_DIR}")

IF(MR_PCH)
  TARGET_PRECOMPILE_HEADERS(${PROJECT_NAME} REUSE_FROM MRPch)
ENDIF()
