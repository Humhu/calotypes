#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <boost/foreach.hpp>

#include "calotypes/CalibrationLog.h"
#include "calotypes/PlanarGridStructure.h"

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
	
	std::string outputPath = config["output_path"].as<std::string>();
	CalibrationLogWriter writer( outputPath );
	
	cv::namedWindow( "images" );
	
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
		
		ImagePoints corners;
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
		
		CameraTrainingData datum;
		datum.imagePoints = corners;
		datum.objectPoints = patternGrid.GetPoints();

		writer.WriteNext( path, image.size(), datum );
		
		cv::Mat drawn;
		cv::drawChessboardCorners( image, patternGrid.GetSize(), corners, true );
		cv::imshow( "images", image );
		cv::waitKey( 1 );
	}
	
	cv::destroyWindow( "images" );
	
}
	
