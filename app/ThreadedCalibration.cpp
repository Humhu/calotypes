#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <boost/foreach.hpp>
#include <boost/bind.hpp>
#include <Eigen/Dense>
#include <fstream>
#include <unistd.h>

#include <boost/circular_buffer.hpp>

#include "calotypes/CrossValidation.hpp"

#include "calotypes/CameraCalibration.h"
#include "calotypes/DatasetFunctions.hpp"
#include "calotypes/SubsampleDataSelector.hpp"
#include "calotypes/GreedyDataSelector.hpp"

#include "calotypes/CalibrationLog.h"
#include "calotypes/WorkerPool.hpp"
#include "calotypes/WeightedSamplers.hpp"

using namespace calotypes;

void PrintHelp()
{
	std::cout << "threaded_calibration: Perform cross-validated camera calibration" << std::endl;
	std::cout << "Arguments:" << std::endl;
	std::cout << "[-d datalogPath] The path to a text file containing the detected corners" << std::endl;
	std::cout << "[-o outputPath] The output log to write cross-validation stats to" << std::endl;
	std::cout << "[-k numFolds (4)] The number of folds for cross-validation" << std::endl;
	std::cout << "[-n numData (30)] Number of images to select from each fold" << std::endl;
	std::cout << "[-m method (gc)] Curation method [ss, ur, gc]" << std::endl;
	std::cout << "[-s subsample (1)] Subsampling to apply to data" << std::endl;
}

int main( int argc, char** argv )
{
	std::string configPath, outputPath;
	std::string method = "gc";
	unsigned int numFolds = 4;
	unsigned int numData = 30;
	unsigned int subsample = 1;
	char c;
	while( (c = getopt( argc, argv, "hd:o:k:n:m:s:")) != -1 )
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
			case 'n':
				numData = strtol( optarg, NULL, 10 );
				break;
			case 'm':
				method.assign( optarg );
				break;
			case 's':
				subsample = strtol( optarg, NULL, 10 );
				break;
			case '?':
                return -1;
            default:
				std::cerr << "Unrecognized option." << std::endl;
                return -1;
		}
	}
	
	if( method != "ss" && method != "ur" && method != "gc")
	{
		std::cerr << "Invalid method. Must be 'subsample (ss), uniform random (ur) or greedy curation (gc)" << std::endl;
		return -1;
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
	unsigned int count = 0;
	while( reader.GetNext( datum ) )
	{
		if( (count % subsample) == 0)
		{
			data.push_back( datum );
		}
		count++;
	}
	
	CameraTrainingParams params;
	params.optimizeAspectRatio = true;
	params.optimizePrincipalPoint = true;
	params.enableRadialDistortion[0] = true;
	params.enableRadialDistortion[1] = true;
	params.enableRadialDistortion[2] = true;
	params.enableRationalDistortion[0] = true;
	params.enableRationalDistortion[1] = true;
	params.enableRationalDistortion[2] = true;
	params.enableTangentialDistortion = true;
	params.enableThinPrism = true;
	
	typedef CrossValidationTask< CameraModel, CameraTrainingData > CameraCV;
	
	WorkerPool pool( 4 );
	pool.StartWorkers();
	
	// Construct selection objects
	CameraModel model( params );

	CameraModel initialModel;
	std::cout << "Training initial model estimate...";
	RandomCameraDataSelector randSelector;
	std::vector<CameraTrainingData> initialData;
	randSelector.SelectData( data, numData, initialData );
	TrainCameraModel( initialModel, initialData, data[0].imageSize );
	std::cout << " complete!" << std::endl;

	std::cout << "Computing data pose estimates...";
	for( unsigned int i = 0; i < data.size(); i++ )
	{
		cv::Mat rvec, tvec;
		cv::solvePnP( data[i].objectPoints, data[i].imagePoints, initialModel.cameraMatrix,
					initialModel.distortionCoefficients, rvec, tvec );
		data[i].pose = Pose3D( tvec, rvec );
	}
	std::cout << " complete!" << std::endl;
	
	// Switch data selection method
	CameraDataSelector::Ptr dataSelector;
	if( method == "ss" )
	{
		// TODO Implement subsampling baseline
		unsigned int subsampleStep = 3;
		dataSelector = std::make_shared< SubsampleDataSelector<CameraTrainingData> >( subsampleStep );
	}
	else if( method == "ur" )
	{
		dataSelector = std::make_shared< RandomDataSelector<CameraTrainingData> >();
	}
	else if( method == "gc" )
	{
		// Unity weights
		KernelFunction<CameraTrainingData>::Ptr distanceFunc = std::make_shared< PoseKernelFunction >();
		// TODO Choose std deviation for kernel
		KernelFunction<CameraTrainingData>::Ptr gaussianKernel = 
			std::make_shared< GaussianKernelAdaptor<CameraTrainingData> >( distanceFunc, 1.0 );
		DatasetFunction<CameraTrainingData>::Ptr ucsd = 
			std::make_shared< UniformCauchySchwarzDivergence<CameraTrainingData> >( gaussianKernel );
			
		dataSelector =
			std::make_shared< GreedyDataSelector<CameraTrainingData> >( ucsd );
	}
	else
	{
		std::cerr << "Invalid method!" << std::endl;
		exit( -1 );
	}
		
	CameraCV::TrainFunc trainer = 
		boost::bind( &SubsetTrainCameraModel, _1, _2, data[0].imageSize, 
					 boost::ref(*dataSelector), numData, params ); // HACK heh
	
	CameraCV::TestFunc tester = boost::bind( &TestCameraModel, _1, _2 );
					 
	
	CameraCV crossValidation( trainer, tester, data, numFolds );
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
	
	double testAcc = 0, trainAcc = 0;
	for( unsigned int i = 0; i < numFolds; i++ )
	{
		std::cout << "\tFold " << i << " test/train: " << results[i].testError
				  << " " << results[i].trainingError << std::endl;
		std::cout << "\tModel " << results[i].model << std::endl;
		
		std::cout << "\tTraining data " << results[i].trainingData.size() << std::endl;
		for( unsigned int j = 0; j < results[i].trainingData.size(); j++ )
		{
			std::cout << "\t\t" << results[i].trainingData[j].name << std::endl;
		}
		
		outputLog << i << " " << results[i].trainingError << " " << results[i].testError << std::endl;
		
		testAcc += results[i].testError;
		trainAcc += results[i].trainingError;
	}
	
	std::cout << "Average test/train error: " << testAcc/numFolds << " " << trainAcc/numFolds << std::endl;
	
	return 0;
}
