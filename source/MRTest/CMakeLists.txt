cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD ${MR_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(MRTest CXX)

file(GLOB SOURCES "*.cpp")

add_executable(${PROJECT_NAME} ${SOURCES})

IF(MR_EMSCRIPTEN)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --emrun")
ENDIF()

IF(APPLE)
  target_link_libraries(${PROJECT_NAME} PRIVATE
    MRMesh
    MRIOExtras
    MRPython
    MRViewer
    jsoncpp
    fmt
    gtest
    spdlog
    Python::Python
    pybind11nonlimitedapi_stubs
    ${CMAKE_DL_LIBS}
  )
ELSE()
  IF(NOT MR_EMSCRIPTEN)
    target_link_libraries(${PROJECT_NAME} PRIVATE
      MRMesh
      MRIOExtras
      MRPython
      MRViewer
      jsoncpp
      fmt
      gtest
      spdlog
      Python::Python
      pybind11nonlimitedapi_stubs
      ${CMAKE_DL_LIBS}
    )
  ELSE()
    set(CMAKE_EXECUTABLE_SUFFIX ".html")
    target_link_libraries(${PROJECT_NAME} PRIVATE
      gtest
      zip
      jsoncpp
      -Wl,--whole-archive
      MRMesh
      MRIOExtras
      MRViewer
      -Wl,--no-whole-archive
    )
  ENDIF()
ENDIF()

# `dladdr` in our patched `PYBIND11_MODULE(...)` needs this.
IF(NOT WIN32)
  target_compile_definitions(${PROJECT_NAME} PRIVATE
    _GNU_SOURCE
  )
ENDIF()

add_test(
  NAME ${PROJECT_NAME}
  COMMAND ${PROJECT_NAME}
)

IF(MR_PCH)
  TARGET_PRECOMPILE_HEADERS(${PROJECT_NAME} REUSE_FROM MRPch)
ENDIF()
