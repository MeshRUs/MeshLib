# to force update thirdparty touch this line
cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(MeshLib_thirdparty CXX)

# all binaries will be located in ./build/Release/bin
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set(MESHLIB_THIRDPARTY_DIR "${PROJECT_SOURCE_DIR}")
set(MESHLIB_THIRDPARTY_INCLUDE_DIR "include")

# Inhibit all warning messages
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w")

IF(MR_EMSCRIPTEN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s USE_BOOST_HEADERS=1")
  IF(NOT MR_EMSCRIPTEN_SINGLETHREAD)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread")
  ENDIF()
ENDIF()

#imgui build
file(GLOB IMGUI_SOURCES "./imgui/*.cpp" "./../source/imgui/*.cpp" "./imgui/misc/freetype/*.cpp" "./imgui/misc/cpp/*.cpp" "./imgui/backends/imgui_impl_opengl3.cpp" "./imgui/backends/imgui_impl_glfw.cpp")
file(GLOB IMGUI_HEADERS "./imgui/*.h" "./../source/imgui/*.h" "./imgui/misc/freetype/*.h" "./imgui/backends/imgui_impl_opengl3.h" "./imgui/backends/imgui_impl_glfw.h" "./imgui/backends/imgui_impl_opengl3_loader.h")
include_directories("../source/")
include_directories("../thirdparty/imgui/")
include_directories("../thirdparty/parallel-hashmap/")
add_compile_definitions(IMGUI_USER_CONFIG=\"imgui/MRCustomImGuiConfig.h\")
add_compile_definitions(IMGUI_ENABLE_FREETYPE)
add_library(imgui SHARED ${IMGUI_SOURCES} ${IMGUI_HEADERS})
IF(MR_EMSCRIPTEN)
  add_compile_definitions(IMGUI_IMPL_OPENGL_ES3)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s USE_FREETYPE=1")
ELSE()
  set(BUILD_SHARED_LIBS ON)
  find_package(glfw3 CONFIG REQUIRED)
  find_package(Freetype REQUIRED)
  include_directories(${FREETYPE_INCLUDE_DIRS})
  IF(APPLE)
    target_link_libraries(imgui PRIVATE ${FREETYPE_LIBRARIES} glfw)
    set(INSTALL_GTEST ON)
    add_subdirectory(./googletest)
  ELSE()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable")
  ENDIF()
ENDIF()

IF(APPLE OR MR_EMSCRIPTEN)
  add_subdirectory(./expected ./expected)
ELSE()
  # TODO: correct OpenVDB module search
  list(APPEND CMAKE_MODULE_PATH
    "/usr/lib/cmake/OpenVDB/"
    "/usr/lib/x86_64-linux-gnu/cmake/OpenVDB/"
    "/usr/lib/aarch64-linux-gnu/cmake/OpenVDB/"
    "/usr/lib64/cmake/OpenVDB/"
  )
  find_package(OpenVDB 9 COMPONENTS openvdb)
  IF(NOT OpenVDB_FOUND)
    set(OPENVDB_ENABLE_UNINSTALL OFF)
    set(OPENVDB_ENABLE_INSTALL OFF)
    set(OPENVDB_CORE_SHARED "ON")
    set(OPENVDB_CORE_STATIC "OFF")
    set(OPENVDB_BUILD_BINARIES OFF)
    set(OPENVDB_BUILD_VDB_PRINT OFF)
    set(USE_EXPLICIT_INSTANTIATION OFF)

    # Ubuntu 20.04 compatibility
    find_package(Boost REQUIRED)
    find_package(TBB REQUIRED)
    IF(Boost_VERSION VERSION_LESS 1.73 OR TBB_VERSION VERSION_LESS 2020.2)
      add_subdirectory(./openvdb/v9/openvdb)
    ELSE()
      add_subdirectory(./openvdb/v10/openvdb)
    ENDIF()
  ENDIF()

  set(CPR_ENABLE_UNINSTALL "OFF")
  set(CPR_ENABLE_INSTALL "OFF")
  set(CPR_USE_SYSTEM_GTEST "ON")
  set(CPR_BUILD_TESTS "OFF")
  set(CPR_BUILD_TESTS_SSL "OFF")
  add_subdirectory(./cpr)
ENDIF()

add_subdirectory(./OpenCTM ./OpenCTM)
option(PHMAP_INSTALL "" ON)
add_subdirectory(./parallel-hashmap)

IF(MR_EMSCRIPTEN)
  set(GDCM_BUILD_APPLICATIONS OFF)
  set(GDCM_BUILD_SHARED_LIBS OFF)
  set(GDCM_BUILD_TESTING OFF)
  set(GDCM_BUILD_DOCBOOK_MANPAGES OFF)
  add_subdirectory(./GDCM)

  add_compile_definitions(GTEST_HAS_CXXABI_H_=0)
  set(INSTALL_GTEST ON)
  add_subdirectory(./googletest)

  set(ENABLE_PROGRAMS OFF CACHE BOOL "")
  set(ENABLE_TESTING OFF CACHE BOOL "")
  add_subdirectory(./mbedtls)
  install(DIRECTORY mbedtls/include/mbedtls DESTINATION ${MESHLIB_THIRDPARTY_INCLUDE_DIR})

  # Use custom FindMbedTLS.cmake to override the one included in ziplib, that does not work
  list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

  set(BUILD_TOOLS OFF CACHE BOOL "")
  set(BUILD_REGRESS OFF CACHE BOOL "")
  set(BUILD_EXAMPLES OFF CACHE BOOL "")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -s USE_ZLIB=1")
  set(LIBZIP_DO_INSTALL ON)
  add_subdirectory(./libzip)

  set(CMAKE_CXX_FLAGS_BACKUP "${CMAKE_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s NO_DISABLE_EXCEPTION_CATCHING=1")
  set(JSONCPP_WITH_TESTS OFF)
  set(JSONCPP_WITH_POST_BUILD_UNITTEST OFF)
  set(BUILD_STATIC_LIBS OFF)
  add_subdirectory(./jsoncpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_BACKUP}")

  add_subdirectory(./fmt)
  #install(DIRECTORY fmt/include/fmt DESTINATION ${MESHLIB_THIRDPARTY_INCLUDE_DIR})

  set(SPDLOG_INSTALL ON)
  set(SPDLOG_FMT_EXTERNAL ON)
  add_compile_definitions(SPDLOG_WCHAR_FILENAMES) # hack to make it work with new version of fmt
  add_subdirectory(./spdlog)

  option(TBB_TEST "Enable testing" OFF)
  set(CMAKE_SYSTEM_PROCESSOR_BACKUP "${CMAKE_SYSTEM_PROCESSOR}")
  set(CMAKE_SYSTEM_PROCESSOR "unknown")
  add_subdirectory(./onetbb)
  set(CMAKE_SYSTEM_PROCESSOR "${CMAKE_SYSTEM_PROCESSOR_BACKUP}")
  install(DIRECTORY onetbb/include/tbb DESTINATION ${MESHLIB_THIRDPARTY_INCLUDE_DIR})

  add_subdirectory(./tinyxml2)

  set(BUILD_TESTS OFF CACHE BOOL "")
  set(BUILD_FUZZERS OFF CACHE BOOL "")
  set(BUILD_BENCHMARKS OFF CACHE BOOL "")
  set(BUILD_STATIC OFF CACHE BOOL "")
  set(BLOSC_INSTALL ON)
  add_subdirectory(./c-blosc)
ELSE() # not MR_EMSCRIPTEN
  set(CLIP_EXAMPLES OFF)
  set(CLIP_TESTS OFF)
  set(CLIP_X11_WITH_PNG OFF)
  set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
  add_subdirectory(./clip)
  file(GLOB CLIP_HEADERS "clip/*.h")
  install(FILES ${CLIP_HEADERS} DESTINATION ${MESHLIB_THIRDPARTY_INCLUDE_DIR}/clip)

  set(E57_XML_PARSER TinyXML2 CACHE STRING "")
  add_subdirectory(./libE57Format)
ENDIF()

set(GLAD_INSTALL ON)
set(GLAD_REPRODUCIBLE ON)
add_subdirectory(./glad)
IF(NOT MR_EMSCRIPTEN)
  set(PYBIND11_INSTALL ON)
  add_subdirectory(./pybind11)
ENDIF()
set(TINYGLTF_INSTALL ON)
add_subdirectory(./tinygltf)

set(WITH_TESTS OFF CACHE BOOL "")
add_subdirectory(./laz-perf)

file(GLOB IMGUI_HEADERS "imgui/*.h")
install(FILES ${IMGUI_HEADERS} DESTINATION ${MESHLIB_THIRDPARTY_INCLUDE_DIR}/imgui)
install(DIRECTORY imgui/backends DESTINATION ${MESHLIB_THIRDPARTY_INCLUDE_DIR}/imgui)
install(DIRECTORY imgui/misc DESTINATION ${MESHLIB_THIRDPARTY_INCLUDE_DIR}/imgui)
install(DIRECTORY eigen/Eigen DESTINATION ${MESHLIB_THIRDPARTY_INCLUDE_DIR})
install(DIRECTORY eigen/unsupported DESTINATION ${MESHLIB_THIRDPARTY_INCLUDE_DIR})
