#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <boost/foreach.hpp>
#include <boost/bind.hpp>
#include <Eigen/Dense>

#include "calotypes/CrossValidation.hpp"
#include "calotypes/CameraCalibration.h"
#include "calotypes/CalibrationLog.h"
#include "calotypes/WorkerPool.hpp"
#include "calotypes/RandomDataSelector.hpp"

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
	
	std::vector<CameraTrainingData> subset;
	RandomDataSelector selector;
// 	selector.SelectData( data, data.size(), subset );
	subset = data;
	
	CameraTrainingParams params;
	params.optimizeAspectRatio = true;
	params.optimizePrincipalPoint = true;
	params.enableRadialDistortion[0] = true;
	params.enableRadialDistortion[1] = false;
	
	typedef CrossValidationTask< CameraModel, CameraTrainingData > CameraCV;
	CameraCV::TrainFunc trainer = boost::bind( &TrainCameraModel, _1, _2,
											   imageSize, params ); // HACK heh
	CameraCV::TestFunc tester = boost::bind( &TestCameraModel, _1, _2 );
	CameraCV crossValidation( trainer, tester, subset, 4 );
	
	unsigned int numFolds = 4;
	
	WorkerPool pool( 4 );
	pool.StartWorkers();
	
	std::cout << "Performing " << numFolds << "-fold cross validation..." << std::endl;
	for( unsigned int i = 0; i < numFolds; i++ )
	{
		WorkerPool::Job job = boost::bind( &CameraCV::ValidateFold, &crossValidation, i );
		pool.EnqueueJob( job );
	}
	pool.WaitOnJobs();
	
	std::vector< CameraCV::ValidationResult > results = crossValidation.GetResults();
	
	std::cout << "Errors: " << std::endl;
	for( unsigned int i = 0; i < numFolds; i++ )
	{
		std::cout << "\tFold " << i << " test/train: " << results[i].testError
				  << " " << results[i].trainingError << std::endl;
		std::cout << "\tModel " << results[i].model << std::endl;
	}
	
	return 0;
}
