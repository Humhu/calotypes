cmake_minimum_required( VERSION 2.6 )

add_executable( simple_calibration SimpleCalibration.cpp )
target_link_libraries( simple_calibration 
					   calotypes
					   ${calotypes_LIBS} )

add_executable( threaded_calibration ThreadedCalibration.cpp )
target_link_libraries( threaded_calibration 
					   calotypes
					   ${calotypes_LIBS} )
					   
add_executable( process_images ProcessImages.cpp )
target_link_libraries( process_images
					   calotypes
					   ${calotypes_LIBS} )

install( TARGETS simple_calibration
		 RUNTIME DESTINATION bin
)
