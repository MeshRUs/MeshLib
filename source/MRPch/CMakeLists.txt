cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD ${MR_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(MRPch CXX)

option(MR_PCH_USE_EXTRA_HEADERS "Add frequently used MeshLib headers to the precompiled header" OFF)

file(GLOB HEADERS "*.h")

install(
  FILES ${HEADERS}
  DESTINATION "${MR_INCLUDE_DIR}/${PROJECT_NAME}"
)

IF(MR_PCH)
  IF(MR_EMSCRIPTEN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s USE_BOOST_HEADERS=1")

    add_library(${PROJECT_NAME} STATIC "MRPch.cpp" ${HEADERS})

    target_link_libraries(${PROJECT_NAME} PRIVATE
      fmt
      jsoncpp
      tbb
    )
  ELSE()
    find_package(Boost COMPONENTS REQUIRED)
    find_package(TBB REQUIRED)
    pkg_check_modules(JSONCPP jsoncpp)
    include_directories(${JSONCPP_INCLUDE_DIRS})

    add_library(${PROJECT_NAME} STATIC "MRPch.cpp" ${HEADERS})

    IF(APPLE)
      # fix Boost.Stacktrace
      target_compile_definitions(${PROJECT_NAME} PRIVATE
        _GNU_SOURCE
      )
    ENDIF()

    target_link_libraries(${PROJECT_NAME} PRIVATE
      fmt
      spdlog
      Boost::boost
      jsoncpp
      tbb
    )
  ENDIF()

  IF(MR_PCH_USE_EXTRA_HEADERS)
    target_compile_definitions(${PROJECT_NAME} PRIVATE MR_PCH_USE_EXTRA_HEADERS)
  ENDIF()
  IF(NOT MESHLIB_BUILD_MRVIEWER)
    target_compile_definitions(${PROJECT_NAME} PRIVATE MESHLIB_NO_VIEWER)
  ENDIF()

  target_precompile_headers(${PROJECT_NAME} PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CURRENT_SOURCE_DIR}/MRPch.h>"
  )

  # `dladdr` in our patched `PYBIND11_MODULE(...)` needs this.
  IF(NOT WIN32)
    target_compile_definitions(${PROJECT_NAME} PRIVATE
      _GNU_SOURCE
    )
  ENDIF()
ENDIF()
