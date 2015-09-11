#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <boost/foreach.hpp>
#include <boost/bind.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <Eigen/Dense>
#include <fstream>
#include <unistd.h>

#include "calotypes/CameraCalibration.h"
#include "calotypes/DatasetFunctions.hpp"

#include "calotypes/SubsampleDataSelector.hpp"
#include "calotypes/GreedyDataSelector.hpp"

#include "calotypes/CalibrationLog.h"
#include "calotypes/KernelDensityEstimation.hpp"

using namespace calotypes;
namespace btime = boost::posix_time;

void PrintHelp()
{
	std::cout << "holdout_calibration: Perform holdout camera calibration" << std::endl;
	std::cout << "Arguments:" << std::endl;
	std::cout << "[-d train path] The path to a text file containing the training detections" << std::endl;
	std::cout << "[-t test path] The path to a text file containing the test detections" << std::endl;
	std::cout << "[-n num data (30)] Number of images to select for training" << std::endl;
	std::cout << "[-m method (gc)] Curation method: subsample (ss), uniform random (ur), or greedy curation (gc)" << std::endl;
	std::cout << "[-s subsample (1)] Subsampling to apply to data" << std::endl;
	std::cout << "[-b kernel SD (1.0)] Gaussian kernel standard deviation" << std::endl;
	std::cout << "[-o outputPath (output.txt)] Output file path" << std::endl;

}

// TODO Switch to getopt_long to avoid confusion in train/test flag
int main( int argc, char** argv )
{
	
	std::string trainPath, testPath;
	std::string outputPath = "output.txt";
	std::string method = "gc";
	unsigned int numData = 30;
	unsigned int subsample = 1;
	char c;
	double standardDev = 1.0;
	while( (c = getopt( argc, argv, "hd:t:n:m:s:b:o:")) != -1 )
	{
		switch( c )
		{
			case 'h':
				PrintHelp();
				return 0;
			case 'd':
				trainPath.assign( optarg );
				break;
			case 't':
				testPath.assign( optarg );
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
			case 'b':
				standardDev= strtod( optarg, NULL );
				break;
			case 'o':
				outputPath.assign( optarg );
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

	// Parse training and test files
	CalibrationLogReader trainReader( trainPath );
	CalibrationLogReader testReader( testPath );
	
	CameraTrainingData datum;
	std::vector<CameraTrainingData> trainData, testData;
	
	unsigned int count = 0;
	while( trainReader.GetNext( datum ) )
	{
		if( (count % subsample) == 0) { trainData.push_back( datum ); }
		count++;
	}
	count = 0;
	while( testReader.GetNext( datum ) )
	{
		if( (count % subsample) == 0) { testData.push_back( datum ); }
		count++;
	}
	
	// boost filesystem gives random access only
	std::sort( trainData.begin(), trainData.end(), CompareDataName );
	
	// Open output log and print header
	std::ofstream outputLog( outputPath );
	if( !outputLog.is_open() )
	{
		std::cerr << "Could not open output log at " << outputPath << std::endl;
		return -1;
	}
	
	btime::ptime now = btime::microsec_clock::local_time();
	outputLog << "calibration time: " << btime::to_simple_string( now ) << std::endl;
	outputLog << "training path: " << trainPath << std::endl;
	outputLog << "test path: " << testPath << std::endl;
	outputLog << "training budget: " << numData << std::endl;
	outputLog << "dataset subsample: " << subsample << std::endl;
	outputLog << "method: " << method << std::endl;
	outputLog << "kernel sd: " << standardDev << std::endl;
	outputLog << std::endl;
	
	// Specify camera model complexity
	// TODO Do this dynamically somehow
	CameraTrainingParams params;
	params.optimizeAspectRatio = true;
	params.optimizePrincipalPoint = true;
	params.enableRadialDistortion[0] = true;
	params.enableRadialDistortion[1] = true;
	params.enableRadialDistortion[2] = true;
	params.enableRationalDistortion[0] = false;
	params.enableRationalDistortion[1] = false;
	params.enableRationalDistortion[2] = false;
	params.enableTangentialDistortion = true;
	params.enableThinPrism = false;
	
	// Construct selection objects
	CameraModel model( params );

	CameraModel initialModel;
	std::cout << "Training initial model estimate...";
	RandomCameraDataSelector randSelector;
	std::vector<CameraTrainingData> initialData;
	randSelector.SelectData( trainData, numData, initialData );
	TrainCameraModel( initialModel, initialData, trainData[0].imageSize ); // TODO Train with params here?
	std::cout << " complete!" << std::endl;
	
	outputLog << "initial model: " << std::endl;
	outputLog << initialModel << std::endl;

	std::cout << "Computing data pose estimates...";
	for( unsigned int i = 0; i < trainData.size(); i++ )
	{
		cv::Mat rvec, tvec;
		cv::solvePnP( trainData[i].objectPoints, trainData[i].imagePoints, initialModel.cameraMatrix,
					initialModel.distortionCoefficients, rvec, tvec );
		trainData[i].pose = Pose3D( tvec, rvec );
	}
	std::cout << " complete!" << std::endl;
	
	// Cauchy-Schwarz objects
	// TODO non-unity weights
	KernelFunction<CameraTrainingData>::Ptr distanceFunc = std::make_shared< PoseKernelFunction >();
	// TODO Choose std deviation for kernel
	KernelFunction<CameraTrainingData>::Ptr gaussianKernel = 
		std::make_shared< GaussianKernelAdaptor<CameraTrainingData> >( distanceFunc, standardDev );
	DatasetFunction<CameraTrainingData>::Ptr ucsd = 
		std::make_shared< UniformCauchySchwarzDivergence<CameraTrainingData> >( gaussianKernel );
	
	// Switch data selection method
	CameraDataSelector::Ptr dataSelector;
	if( method == "ss" )
	{
		// TODO Implement subsampling baseline
		unsigned int subsamplePeriod = 30;
		dataSelector = std::make_shared< SubsampleDataSelector<CameraTrainingData> >( subsamplePeriod );
	}
	else if( method == "ur" )
	{
		dataSelector = std::make_shared< RandomDataSelector<CameraTrainingData> >();
	}
	else if( method == "gc" )
	{
		dataSelector = std::make_shared< GreedyDataSelector<CameraTrainingData> >( ucsd );
// 		dataSelector = std::make_shared< RepeatedGreedyDataSelector<CameraTrainingData> >( ucsd, 10 );
	}
	else
	{
		std::cerr << "Invalid method!" << std::endl;
		exit( -1 );
	}
	
	SubsetTrainCameraModel( model, trainData, trainData[0].imageSize, 
							*dataSelector, numData, params ); // HACK
		
	double trainingError = TestCameraModel( model, trainData );
	double testError = TestCameraModel( model, testData );
	double csd = (*ucsd)( trainData );
	
	std::cout << "Training on: " << std::endl;
	for( unsigned int i = 0; i < trainData.size(); i++ )
	{
		std::cout << "\t" << trainData[i].name << std::endl;
	}
	
	std::cout << "Test/training error: " << testError << " " << trainingError << std::endl;
	
	// Print final output results
	outputLog << "final model: " << std::endl;
	outputLog << model << std::endl;
	
	outputLog << "training paths: " << std::endl;
	for( unsigned int i = 0; i < trainData.size(); i++ )
	{
		outputLog << trainData[i].name << std::endl;
	}
	outputLog << std::endl;
	
	outputLog << "training csd: " << csd << std::endl;
	outputLog << "training error: " << trainingError << std::endl;
	outputLog << "test error: " << testError << std::endl;
	
	return 0;
}
