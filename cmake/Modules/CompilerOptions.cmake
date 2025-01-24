set(MR_CXX_STANDARD 20 CACHE STRING "Version of the C++ standard used to compile the project")
IF(DEFINED ENV{MR_USE_CPP_23} AND "$ENV{MR_USE_CPP_23}" STREQUAL "ON")
  message("MR_USE_CPP_23 variable is deprecated; consider setting MR_CXX_STANDARD to 23")
  set(MR_CXX_STANDARD 23)
ENDIF()
set(CMAKE_CXX_STANDARD ${MR_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MR_PCH_DEFAULT OFF)
# for macOS, GCC, and Clang<15 builds: PCH not only does not give any speedup, but even vice versa
IF(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 15)
  set(MR_PCH_DEFAULT ON)
#ELSEIF(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
#  set(MR_PCH_DEFAULT ON)
ENDIF()
set(MR_PCH ${MR_PCH_DEFAULT} CACHE BOOL "Enable precompiled headers")
IF(MR_PCH AND NOT MR_EMSCRIPTEN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
ENDIF()
message("MR_PCH=${MR_PCH}")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG -DDEBUG")
# turn on warnings as errors
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-missing-field-initializers -Wno-unknown-pragmas -Wno-sign-compare -Werror -fvisibility=hidden -pedantic-errors -DIMGUI_DISABLE_OBSOLETE_FUNCTIONS -DIMGUI_ENABLE_FREETYPE")

IF(NOT MR_EMSCRIPTEN_SINGLETHREAD)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
ENDIF() # NOT MR_EMSCRIPTEN_SINGLETHREAD

IF(WIN32 AND MINGW)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wa,-mbig-obj")
ENDIF()

# make link to fail if there are unresolved symbols (GCC and Clang)
IF(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,defs")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,defs")
ENDIF()

# This allows us to share bindings for C++ types across compilers (across GCC and Clang). Otherwise Pybind refuses
# to share them because the compiler name and the ABI version number are different, even when there's no actual ABI incompatibility in practice.
add_compile_definitions(PYBIND11_COMPILER_TYPE=\"_meshlib\")
add_compile_definitions(PYBIND11_BUILD_ABI=\"_meshlib\")

# Things for our patched pybind: --- [

# It's a good idea to have this match `PYTHON_MIN_VERSION` in `scripts/mrbind/generate.mk`.
# Here `0x030800f0` corresponds to 3.8 (ignore the `f0` suffix at the end, it just means a release version as opposed to alpha/beta/etc).
add_compile_definitions(Py_LIMITED_API=0x030800f0)

# It's a good idea to have this match the value specified in `scripts/mrbind/generate.mk`. See that file for the explanation.
add_compile_definitions(PYBIND11_INTERNALS_VERSION=5)

# This affects the naming of our pybind shims.
add_compile_definitions(PYBIND11_NONLIMITEDAPI_LIB_SUFFIX_FOR_MODULE=\"meshlib\")

# ] --- end things for our patched pybind

# Warn about ABI incompatibilities.
# GCC 12 fixed a bug, and this fix affects the ABI: https://github.com/gcc-mirror/gcc/commit/a37e8ce3b66325f0c6de55c80d50ac1664c3d0eb
# Because of this fix GCC 11 and older are incompatible with GCC 12+, and also with Clang that we use the build the Python bindings.
# This breaks the bindings on Ubuntu 20.04 (where we use GCC 10).
# This ABI change affects inheriting from certain classes with trailing padding, and the fix is always to add a dummy member variable at the end (mark it with
#   MR_BIND_IGNORE to hide from the bindings) to make sure there's no trailing padding. This affects only those bases that are aggregates and have default
#   member initializers.
# We can remove this flag when we stop supporting Ubuntu 20.04. In theory, Ubuntu 22.04 also uses GCC 11 by default, but it also has GCC 12, and on it we
#   use GCC 12 for Meshlib, so the resulting MeshLib library is probably incompatible with GCC 11 anyway, so we don't care about this bug there.
# We only enable this on GCC 12, because the next versions introduce more ABI changes that warn here, and we don't care about them (about much newer GCCs being
#   incompatible with GCC 11.)
IF(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12 AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wabi=16")
ENDIF()
# complitely ignore "maybe-uninitialized" for GCC because of false positives
# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=109561
# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=116090
IF(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-maybe-uninitialized")
ENDIF()

# TODO: __aarch64__ ?
IF(NOT APPLE AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
  message("CMAKE_SYSTEM_PROCESSOR is ${CMAKE_SYSTEM_PROCESSOR}")
  add_compile_definitions(__ARM_CPU__)
  message("ARM cpu detected")
ENDIF()

IF(MR_EMSCRIPTEN)
  # reference: https://github.com/emscripten-core/emscripten/blob/main/src/settings.js
  string(JOIN " " CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS}"
    "-s EXPORTED_RUNTIME_METHODS=[ccall]"
    "-s ALLOW_MEMORY_GROWTH=1"
    "-s MAXIMUM_MEMORY=4GB"
    "-s LLD_REPORT_UNDEFINED=1"
    "-s USE_WEBGL2=1"
    "-s USE_GLFW=3"
    "-s USE_ZLIB=1"
    "-s FULL_ES3=1"
    "-s USE_LIBPNG=1"
  )

  IF(MR_EMSCRIPTEN_SINGLETHREAD)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s ENVIRONMENT=web")
  ELSE()
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s ENVIRONMENT=web,worker -pthread -s PTHREAD_POOL_SIZE_STRICT=0 -s PTHREAD_POOL_SIZE=navigator.hardwareConcurrency")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-pthreads-mem-growth") # look https://github.com/emscripten-core/emscripten/issues/8287
  ENDIF() # NOT MR_EMSCRIPTEN_SINGLETHREAD

  IF(NOT MR_DISABLE_EMSCRIPTEN_ASYNCIFY)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s ASYNCIFY -Wno-limited-postlink-optimizations")
    add_compile_definitions(MR_EMSCRIPTEN_ASYNCIFY)
  ENDIF() # NOT MR_DISABLE_EMSCRIPTEN_ASYNCIFY

  add_compile_definitions(SPDLOG_FMT_EXTERNAL)
  add_compile_definitions(SPDLOG_WCHAR_FILENAMES) # hack to make it work with new version of fmt

  # FIXME: comment required
  add_compile_definitions(EIGEN_STACK_ALLOCATION_LIMIT=0)
ENDIF() # MR_EMSCRIPTEN
