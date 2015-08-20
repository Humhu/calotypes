#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <boost/foreach.hpp>
#include <boost/bind.hpp>
#include <Eigen/Dense>

#include "calotypes/CrossValidation.hpp"
#include "calotypes/CameraCalibration.h"
#include "calotypes/CalibrationLog.h"

using namespace calotypes;

int main( int argc, char** argv )
{

	if( argc < 2 )
	{
		std::cerr << "Please specify config file path." << std::endl;
		return -1;
	}
	
	std::string configPath( argv[1] );
	CalibrationLogReader reader( configPath );
	
	CameraTrainingData datum;
	std::vector<CameraTrainingData> data;
	std::string path;
	cv::Size imageSize;
	while( reader.GetNext( path, imageSize, datum ) )
	{
		data.push_back( datum );
	}
	
	CameraTrainingParams params;
	params.optimizeAspectRatio = false;
	params.optimizePrincipalPoint = false;
	
	typedef CrossValidator< CameraModel, CameraTrainingData > CameraCV;
	CameraCV::TrainFunc trainer = boost::bind( &TrainCameraModel, _1,
											   imageSize, params ); // HACK heh
	CameraCV::TestFunc tester = boost::bind( &TestCameraModel, _1, _2 );
	CameraCV crossValidator( trainer, tester );
	
	std::vector<CameraModel> models;
	std::vector<double> errors;
	unsigned int numFolds = 4;
	std::cout << "Performing " << numFolds << "-fold cross validation..." << std::endl;
	double mse = crossValidator.PerformCrossValidation( data, numFolds, models, errors );
	std::cout << "CV completed with MSE: " << mse << std::endl;
	std::cout << "Errors: " << std::endl;
	for( unsigned int i = 0; i < numFolds; i++ )
	{
		std::cout << "\tFold " << i << ": " << errors[i] << std::endl;
		std::cout << "\tModel " << models[i] << std::endl;
	}
	
	return 0;
}
