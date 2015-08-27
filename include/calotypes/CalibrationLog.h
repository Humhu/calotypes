#pragma once

#include <opencv2/core.hpp>

#include "calotypes/CameraCalibration.h"

#include <fstream>

namespace calotypes
{

/*! \brief Records and reads camera training data from a text log. 
 * Log lines are stored as: 
 * [image_path] [image_w] [image_h]
 * [obj_1_x] [obj_1_y] [obj_1_z] ... 
 * [img_1_x] [img_1_y] ... */
class CalibrationLogReader
{
public:
	
	CalibrationLogReader( const std::string& path );
	
	/*! \brief Retrieve the next training datapoint. Returns success. */
	bool GetNext( CameraTrainingData& data );
	
private:
	
	std::ifstream log;
	
};

class CalibrationLogWriter
{
public:
	
	CalibrationLogWriter( const std::string& path );
	
	/*! \brief Writes a training datapoint to the log. */
	void WriteNext( const CameraTrainingData& data );
	
	std::ofstream log;
	
};
	
}
