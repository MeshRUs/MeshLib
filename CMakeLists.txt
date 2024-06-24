cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")
include(CompilerOptions)

IF(NOT MR_EMSCRIPTEN AND NOT APPLE AND EXISTS /usr/bin/time)
  # print compile time of each translation unit in the CSV format (user time, system time, command)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "/usr/bin/time -f \",%U,%S,%C\" -a -o compile_timings.txt")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "/usr/bin/time -f \",%U,%S,%C\" -a -o link_timings.txt")
ENDIF()

project(MeshLib CXX)
add_compile_definitions(MR_PROJECT_NAME=\"MeshLib\")
add_compile_definitions(MR_FRAMEWORK)
add_compile_definitions(IMGUI_USER_CONFIG=\"imgui/MRCustomImGuiConfig.h\")

# all binaries will be located in ./build/Release/bin
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(MESHLIB_CUSTOM_INSTALL_PREFIX "Install into directory defined by CMAKE_INSTALL_PREFIX" OFF)
option(MESHLIB_PYTHON_SUPPORT "Python support" ON)
option(MESHLIB_BUILD_MRVIEWER "Build MRViewer library and application" ON)
option(MESHLIB_BUILD_PYTHON_MODULES "Build Python modules" ON)
option(MESHLIB_BUILD_MESHCONV "Build meshconv utility" ON)
option(MESHLIB_BUILD_MRCUDA "Build MRCuda library" ON)

include(CTest)

set(MESHLIB_THIRDPARTY_DIR "${PROJECT_SOURCE_DIR}/thirdparty")

set(MESHLIB_THIRDPARTY_ROOT_DIR "${PROJECT_SOURCE_DIR}" CACHE PATH "Thirdparty library location")
IF(NOT MESHLIB_CUSTOM_INSTALL_PREFIX)
  IF(NOT EXISTS "${MESHLIB_THIRDPARTY_ROOT_DIR}")
    message(FATAL_ERROR "thirdparty build directory not found! You can build thirdparty with ./scripts/build_thirdparty.sh")
  ENDIF()
  include_directories(${MESHLIB_THIRDPARTY_ROOT_DIR}/include)
  link_directories(${MESHLIB_THIRDPARTY_ROOT_DIR}/lib)
ENDIF()

IF(APPLE)
  # need to hardcode some path to fix wheel (no option to pass path)
  # TODO: revise
  set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/..;@loader_path/../lib;@loader_path/../lib/lib;@loader_path/meshlib;${MESHLIB_THIRDPARTY_ROOT_DIR}/lib;${CMAKE_BINARY_DIR}/bin")
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

  IF(BUILD_TESTING)
    set(GTest_DIR ${MESHLIB_THIRDPARTY_ROOT_DIR}/lib/cmake/GTest)
    find_package(GTest REQUIRED CONFIG)
    include_directories(${GTEST_INCLUDE_DIRS})
  ENDIF()
ENDIF()

include(DetectPlatform)
IF(MESHLIB_PYTHON_SUPPORT AND NOT MR_EMSCRIPTEN)
  # TODO: drop Python version hard-coding; set -DMESHLIB_PYTHON_VERSION explicitly in the CI config if needed
  IF(APPLE)
    set(MESHLIB_PYTHON_VERSION "3.10")
  ELSEIF(DEFINED MR_LINUX_DISTRO)
    IF(${MR_LINUX_DISTRO} STREQUAL "Ubuntu")
      set(MESHLIB_PYTHON_VERSION "3.8")
      IF(${MR_PLATFORM} STREQUAL "Ubuntu_22")
        set(MESHLIB_PYTHON_VERSION "3.10")
      ELSEIF(${MR_PLATFORM} STREQUAL "Ubuntu_24")
        set(MESHLIB_PYTHON_VERSION "3.12")
      ENDIF()
    ELSEIF(${MR_LINUX_DISTRO} STREQUAL "Fedora")
      set(MESHLIB_PYTHON_VERSION "3.11")
      IF(${MR_PLATFORM} STREQUAL "Fedora_39")
        set(MESHLIB_PYTHON_VERSION "3.12")
      ENDIF()
    ENDIF()
  ENDIF()

  IF($ENV{MESHLIB_PYTHON_VERSION})
    set(MR_PYTHON_VERSION $ENV{MESHLIB_PYTHON_VERSION})
  ENDIF()

  IF(DEFINED MESHLIB_PYTHON_VERSION)
    find_package(Python ${MESHLIB_PYTHON_VERSION} EXACT REQUIRED COMPONENTS Interpreter Development)
  ELSE()
    find_package(Python REQUIRED COMPONENTS Interpreter Development)
  ENDIF()

  IF(APPLE)
    # TODO: revise
    link_directories(${HOMEBREW_PREFIX}/opt/python@${MESHLIB_PYTHON_VERSION}/Frameworks/Python.framework/Versions/${MESHLIB_PYTHON_VERSION}/lib/)
    find_package(pybind11 REQUIRED)
  ENDIF()
ENDIF()

set(PROJECT_SOURCE_DIR ./source)
include_directories(${PROJECT_SOURCE_DIR})

IF(NOT MR_EMSCRIPTEN)
  find_package(PkgConfig REQUIRED)

  pkg_check_modules(JSONCPP jsoncpp)
  include_directories(${JSONCPP_INCLUDE_DIRS})

  find_package(fmt)
  find_package(spdlog REQUIRED)
  IF(spdlog_VERSION VERSION_GREATER_EQUAL "1.7" AND fmt_VERSION VERSION_GREATER_EQUAL "7.0")
    add_compile_definitions(SPDLOG_FMT_EXTERNAL)
  ENDIF()
  add_compile_definitions(SPDLOG_COMPILED_LIB SPDLOG_SHARED_LIB)
ENDIF() # NOT MR_EMSCRIPTEN

IF(NOT $ENV{MR_VERSION} STREQUAL "")
  set(MESHLIB_VERSION $ENV{MR_VERSION})
ELSE()
  set(MESHLIB_VERSION 0.0.0)
ENDIF()
message("version: ${MESHLIB_VERSION}")

IF(MESHLIB_CUSTOM_INSTALL_PREFIX)
  include(GNUInstallDirs)
  set(MR_BIN_DIR "${CMAKE_INSTALL_BINDIR}")
  set(MR_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}")
  set(MR_MAIN_LIB_DIR "${CMAKE_INSTALL_LIBDIR}")
  set(MR_PY_LIB_DIR "${CMAKE_INSTALL_LIBDIR}/meshlib")
  set(MR_RESOURCES_DIR "${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}")
  set(MR_FONTS_DIR "${CMAKE_INSTALL_DATAROOTDIR}/fonts")
  set(MR_CONFIG_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
ELSEIF(NOT APPLE)
  set(MR_ROOT "./usr/local")
  set(MR_BIN_DIR "${MR_ROOT}/bin")
  set(MR_INCLUDE_DIR "${MR_ROOT}/include/MeshLib")
  set(MR_MAIN_LIB_DIR "${MR_ROOT}/lib/MeshLib")
  set(MR_PY_LIB_DIR "${MR_MAIN_LIB_DIR}/meshlib")
  set(MR_RESOURCES_DIR "${MR_ROOT}/etc/MeshLib")
  set(MR_FONTS_DIR "${MR_ROOT}/share/fonts")
  set(MR_CONFIG_DIR "${MR_MAIN_LIB_DIR}/cmake")
ELSE()
  set(MR_ROOT "./Library/Frameworks/MeshLib.framework/Versions/${MESHLIB_VERSION}")
  set(MR_BIN_DIR "${MR_ROOT}/bin")
  set(MR_INCLUDE_DIR "${MR_ROOT}/include")
  set(MR_MAIN_LIB_DIR "${MR_ROOT}/lib")
  set(MR_PY_LIB_DIR "${MR_MAIN_LIB_DIR}/meshlib")
  set(MR_RESOURCES_DIR "${MR_ROOT}/Resources")
  set(MR_FONTS_DIR "${MR_ROOT}/Resources/fonts")
  set(MR_CONFIG_DIR "${MR_ROOT}/Resources/cmake")
ENDIF()

# MRMesh library is always built
add_subdirectory(${PROJECT_SOURCE_DIR}/MRPch ./MRPch)
add_subdirectory(${PROJECT_SOURCE_DIR}/MRMesh ./MRMesh)

IF(MESHLIB_BUILD_MRVIEWER)
  set(IMGUI_DIR ${MESHLIB_THIRDPARTY_ROOT_DIR}/include/imgui)
  include_directories(${IMGUI_DIR})

  IF(NOT MR_EMSCRIPTEN)
    find_package(glfw3 CONFIG REQUIRED)
    pkg_check_modules(GTKMM gtkmm-3.0)
  ENDIF()

  IF(MR_EMSCRIPTEN)
    include(EmscriptenHelpers)
    mr_emscripten_pack_directory("${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/assets" "/")
    IF(NOT MR_DISABLE_EMSCRIPTEN_ASYNCIFY)
      mr_emscripten_set_async_func_list("${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/wasm_async_func_list.txt")
    ENDIF()
  ENDIF()

  add_subdirectory(${PROJECT_SOURCE_DIR}/MRViewer ./MRViewer)
  add_subdirectory(${PROJECT_SOURCE_DIR}/MRCommonPlugins ./MRCommonPlugins)
  add_subdirectory(${PROJECT_SOURCE_DIR}/MRViewerApp ./MRViewerApp)
ENDIF()

IF(NOT MESHLIB_PYTHON_SUPPORT)
  set(MESHLIB_BUILD_PYTHON_MODULES OFF)
ENDIF()
IF(NOT MR_EMSCRIPTEN)
  IF(MESHLIB_BUILD_PYTHON_MODULES)
    add_subdirectory(${PROJECT_SOURCE_DIR}/mrmeshpy ./mrmeshpy)
    add_subdirectory(${PROJECT_SOURCE_DIR}/mrmeshnumpy ./mrmeshnumpy)
    IF(MESHLIB_BUILD_MRVIEWER)
      add_subdirectory(${PROJECT_SOURCE_DIR}/mrviewerpy ./mrviewerpy)
    ENDIF()
  ENDIF()
ENDIF()

IF(NOT MR_EMSCRIPTEN)
  IF(MESHLIB_BUILD_MESHCONV)
    add_subdirectory(${PROJECT_SOURCE_DIR}/meshconv ./meshconv)
  ENDIF()
ENDIF()

IF(NOT MR_EMSCRIPTEN AND NOT APPLE)
  IF(MESHLIB_BUILD_MRCUDA)
    add_subdirectory(${PROJECT_SOURCE_DIR}/MRCuda ./MRCuda)
  ELSE()
    message(STATUS "MESHLIB_BUILD_MRCUDA is ${MESHLIB_BUILD_MRCUDA}")
  ENDIF()
ENDIF()

IF(BUILD_TESTING)
  enable_testing()
  add_subdirectory(${PROJECT_SOURCE_DIR}/MRTest ./MRTest)
ENDIF()

include(CMakePackageConfigHelpers)
configure_package_config_file(meshlib-config.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/meshlib-config.cmake
  INSTALL_DESTINATION ${MR_CONFIG_DIR}
  PATH_VARS MR_INCLUDE_DIR MR_MAIN_LIB_DIR
)

write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/meshlib-config-version.cmake
  VERSION ${MESHLIB_VERSION}
  COMPATIBILITY ExactVersion
)

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/meshlib-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/meshlib-config-version.cmake
  DESTINATION ${MR_CONFIG_DIR}
)

set(CPACK_GENERATOR "DRAGNDROP")
include(CPack)
