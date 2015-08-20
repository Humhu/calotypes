cmake_minimum_required(VERSION 2.8)

set( SRCS CalibrationLog.cpp
		  CameraCalibration.cpp )

add_library( calotypes SHARED ${SRCS} )

file( GLOB HEADERS "${PROJECT_SOURCE_DIR}/include/calotypes/*.h"
      "${PROJECT_SOURCE_DIR}/include/calotypes/*.hpp")

set_target_properties( calotypes PROPERTIES
					   OUTPUT_NAME calotypes
					   PUBLIC_HEADER "${HEADERS}" )

install( TARGETS calotypes
		 EXPORT calotypesTargets
		 LIBRARY DESTINATION "${LIBRARY_INSTALL_DIR}" COMPONENT shlib
		 PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}"
		 COMPONENT dev )

target_link_libraries( calotypes ${calotypes_LIBS} )
