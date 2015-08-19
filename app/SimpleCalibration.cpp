#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <boost/foreach.hpp>
#include <Eigen/Dense>

#include "calotypes/PlanarGridStructure.h"
#include "calotypes/ReprojectionEvaluation.h"

using namespace calotypes;

int main( int argc, char** argv )
{

	if( argc < 2 )
	{
		std::cerr << "Please specify config file path." << std::endl;
		return -1;
	}
	
	std::string configPath( argv[1] );
	YAML::Node config = YAML::LoadFile( configPath );
	
	int cbRows = config["cb_rows"].as<int>();
	int cbCols = config["cb_cols"].as<int>();
	double cbDim = config["cb_dim"].as<double>();
	PlanarGridStructure patternGrid( cbRows, cbCols, cbDim );
	
	typedef std::vector< std::string > Strings;
	Strings imagePaths = config["images"].as< Strings >();
	
	cv::namedWindow( "images" );
	
	std::vector< cv::Mat > images;
	typedef std::vector< cv::Point2f > Corners;
	typedef std::vector< cv::Point3f > Points;
	std::vector< Corners > imagePoints;
	std::vector< Points > gridPoints;
	BOOST_FOREACH( const std::string& path, imagePaths )
	{
		std::cout << "Reading image " << path << std::endl;
		cv::Mat image = cv::imread( path );
		if( image.empty() )
		{
			std::cerr << "Could not read image at " << path << std::endl;
			return -1;
		}
		cv::Mat gray;
		cv::cvtColor( image, gray, cv::COLOR_BGR2GRAY);
		images.push_back( gray );
		
		Corners corners;
		if( !cv::findChessboardCorners( gray, patternGrid.GetSize(), corners,
										cv::CALIB_CB_ADAPTIVE_THRESH | 
										cv::CALIB_CB_NORMALIZE_IMAGE | 
										cv::CALIB_CB_FAST_CHECK ) )
		{
			std::cerr << "Could not find chessboard in " << path << std::endl;
			continue;
		}
		cv::TermCriteria cspCrit( cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.1 );
		cv::cornerSubPix( gray, corners, cv::Size(11,11), cv::Size(-1,-1), cspCrit );
		imagePoints.push_back( corners );
		gridPoints.push_back( patternGrid.GetPoints() );
		
		cv::Mat drawn;
		cv::drawChessboardCorners( image, patternGrid.GetSize(), corners, true );
		cv::imshow( "images", image );
		cv::waitKey( 1 );
	}
	
	cv::destroyWindow( "images" );
	
	// LOO cross-validation
	// TODO Fails for images without checkerboards
	std::vector< Corners > predicted( images.size() );
	for( unsigned int i = 0; i < images.size(); i++ )
	{
		std::vector< Points > gridLOO = gridPoints;
		gridLOO.erase( gridLOO.begin() + i );
		std::vector< Corners > cornersLOO = imagePoints;
		cornersLOO.erase( cornersLOO.begin() + i );
		
		std::cout << "LOOCV iteration " << i << std::endl;
		
		// Basic calibration
		cv::Mat cameraMatrix = cv::Mat::eye( 3, 3, CV_64F );
		cv::Mat distortionCoeffs( 12, 1, CV_64F );
		std::vector< cv::Mat > rvecs, tvecs;
		double residual = cv::calibrateCamera( gridLOO,
											   cornersLOO,
											   images[0].size(), // HACK
											   cameraMatrix,
											   distortionCoeffs,
											   rvecs, 
											   tvecs );//,
// 											   cv::CALIB_RATIONAL_MODEL |
// 											   cv::CALIB_THIN_PRISM_MODEL );

		typedef Eigen::Matrix< double, 3, 3 > CameraMatrix;
		typedef Eigen::Matrix< double, 12, 1 > DistortionVector;
		Eigen::Map< CameraMatrix > cameraMap( cameraMatrix.ptr<double>() );
		Eigen::Map< DistortionVector > distortionMap( distortionCoeffs.ptr<double>() );
		
		std::cout << "Calibration completed with residual " << residual << std::endl;
		std::cout << "Camera matrix: " << std::endl << cameraMap << std::endl;
		std::cout << "Distortion coeffs: " << std::endl << distortionMap.transpose() << std::endl;
		
		std::vector< Corners > prediction;
		std::vector< Corners > observed{ imagePoints[i] };
		EvaluateCalibration( cameraMatrix, distortionCoeffs, patternGrid,
							 observed, prediction );
		predicted[i] = prediction[0];
	}
	
	double mse = ComputeMSE( imagePoints, predicted );
	std::cout << "LOOCV MSE: " << mse << std::endl;
		
	return 0;
}
