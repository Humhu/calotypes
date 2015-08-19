# calotypes CMake files
# Humphrey Hu, 2015

project(calotypes)

cmake_minimum_required(VERSION 2.8)

# Defaults to release build
if (NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  message(STATUS "No build type selected, default to Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif ()
message(STATUS "Build configuration: ${CMAKE_BUILD_TYPE}")

# Set install paths
set (LIBRARY_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/lib" CACHE PATH
  "Target directory for all libraries.")
set (EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin" CACHE PATH
  "Target directory for all executables.")

add_definitions(-Wall -std=c++0x)

set(calotypes_VERSION_MAJOR 1)
set(calotypes_VERSION_MINOR 0)
set(calotypes_VERSION "${calotypes_VERSION_MAJOR}.${calotypes_VERSION_MINOR}")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

# Boost utils
find_package( Boost REQUIRED )
include_directories( ${Boost_INCLUDE_DIRS} )

# OpenCV for camera interface, looks for a symlink
find_package(OpenCV)
message(STATUS "OpenCV found at: ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})
list(APPEND calotypes_LIBS ${OpenCV_LIBS})
if( OpenCV_VERSION VERSION_EQUAL "3")
	add_definitions("-DOPENCV3=1")
	list(APPEND calotypes_DEFS "-DOPENCV3=1")
endif()

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
list(APPEND calotypes_INCLUDES ${EIGEN3_INCLUDE_DIR} )

find_package( YamlCpp REQUIRED )
include_directories( ${YAMLCPP_INCLUDE_DIR} )
list(APPEND calotypes_INCLUDES ${YAMLCPP_INCLUDE_DIR} ) # TODO Clean up!
list( APPEND calotypes_LIBS ${YAMLCPP_LIBRARY} )

# Camera drivers
find_package(V4L2 REQUIRED)
include_directories(${V4L2_INCLUDE_DIRS})
list(APPEND calotypes_LIBS ${V4L2_LIBRARIES})

# headers
include_directories("${PROJECT_SOURCE_DIR}/include")

set(INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/include/calotypes")
set(LIBRARY_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib")
set(BINARY_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/bin")

add_subdirectory(src)
add_subdirectory(app)

export(TARGETS calotypes
    FILE "${PROJECT_BINARY_DIR}/calotypesTargets.cmake")

configure_file(calotypesConfig.cmake.in
    "${PROJECT_BINARY_DIR}/calotypesConfig.cmake" @ONLY)

configure_file(calotypesConfigVersion.cmake.in
    "${PROJECT_BINARY_DIR}/calotypesConfigVersion.cmake" @ONLY)

install(FILES
    "${PROJECT_BINARY_DIR}/calotypesConfig.cmake"
    "${PROJECT_BINARY_DIR}/calotypesConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_PREFIX}" COMPONENT dev)

# install(EXPORT calotypesTargets
#     DESTINATION "${CMAKE_INSTALL_PREFIX}" COMPONENT dev)
