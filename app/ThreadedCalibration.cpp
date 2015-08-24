#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <boost/foreach.hpp>
#include <boost/bind.hpp>
#include <Eigen/Dense>
#include <fstream>
#include <unistd.h>

#include "calotypes/CrossValidation.hpp"
#include "calotypes/CameraCalibration.h"
#include "calotypes/CalibrationLog.h"
#include "calotypes/WorkerPool.hpp"
#include "calotypes/RandomDataSelector.hpp"

using namespace calotypes;

void PrintHelp()
{
	std::cout << "threaded_calibration: Perform cross-validated camera calibration" << std::endl;
	std::cout << "Arguments:" << std::endl;
	std::cout << "[-d datalogPath] The path to a text file containing the detected corners" << std::endl;
	std::cout << "[-o outputPath] The output log to write cross-validation stats to" << std::endl;
	std::cout << "[-k numFolds (4)] The number of folds for cross-validation" << std::endl;
}

int main( int argc, char** argv )
{
	
	std::string configPath, outputPath;
	unsigned int numFolds = 4;
	char c;
	while( (c = getopt( argc, argv, "hd:o:k:")) != -1 )
	{
		switch( c )
		{
			case 'h':
				PrintHelp();
				return 0;
			case 'd':
				configPath.assign( optarg );
				break;
			case 'o':
				outputPath.assign( optarg );
				break;
			case 'k':
				numFolds = strtol( optarg, NULL, 10 );
				break;
			case '?':
                if (isprint (optopt)) 
				{
                    std::cerr << "Unknown option -" << optopt << std::endl;
                } 
                else 
				{
                    std::cerr << "Unknown option character " << optopt << std::endl;
                }
                return -1;
            default:
				std::cerr << "Unrecognized option." << std::endl;
                return -1;
		}
	}
	
	CalibrationLogReader reader( configPath );
	std::ofstream outputLog( outputPath );
	if( !outputLog.is_open() )
	{
		std::cerr << "Could not open output log at " << outputPath << std::endl;
		return -1;
	}
	
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
	
	outputLog << "cross-validation results for " << configPath << std::endl;
	outputLog << "k = " << numFolds << std::endl;
	outputLog << "[fold_index] [train_error] [test_error]" << std::endl;
	
	std::cout << "Errors: " << std::endl;
	for( unsigned int i = 0; i < numFolds; i++ )
	{
		std::cout << "\tFold " << i << " test/train: " << results[i].testError
				  << " " << results[i].trainingError << std::endl;
		std::cout << "\tModel " << results[i].model << std::endl;
		
		outputLog << i << " " << results[i].trainingError << " " << results[i].testError << std::endl;
	}
	
	return 0;
}
