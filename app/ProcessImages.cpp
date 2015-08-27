#include <opencv2/opencv.hpp>
#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>
#include <unistd.h>
#include "calotypes/CalibrationLog.h"
#include "calotypes/PlanarGridStructure.h"

using namespace calotypes;
namespace bfs = boost::filesystem;

void PrintHelp()
{
	std::cout << "process_images: Generate detection file from images" << std::endl;
	std::cout << "Arguments: " << std::endl;
	std::cout << "[-r rows] Rows in the pattern" << std::endl;
	std::cout << "[-c cols] Columns in the pattern" << std::endl;
	std::cout << "[-s size] Pattern grid dimension" << std::endl;
	std::cout << "[-d dir] Directory of images" << std::endl;
	std::cout << "[-o output] Output file name" << std::endl;
}

int main( int argc, char** argv )
{

	unsigned int cbRows = 6, cbCols = 7;
	double cbDim = 1.0;
	char c;
	std::string imageDir( "./" );
	std::string outputPath( "detections.txt" );
	
	while( (c = getopt( argc, argv, "hr:c:s:d:o:")) != -1 )
	{
		switch( c )
		{
			case 'h':
				PrintHelp();
				return 0;
			case 'r':
				cbRows = strtol( optarg, NULL, 10 );
				break;
			case 'c':
				cbCols = strtol( optarg, NULL, 10 );
				break;
			case 's':
				cbDim = strtod( optarg, NULL );
				break;
			case 'd':
				imageDir.assign( optarg );
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

	PlanarGridStructure patternGrid( cbRows, cbCols, cbDim );
	
	if( *imageDir.end() != '/' ) { imageDir += "/"; }
	
	if( !bfs::exists( imageDir ) || !bfs::is_directory( imageDir ) )
	{
		throw std::runtime_error( "Invalid directory " + imageDir );
	}
	
	std::vector< std::string > imagePaths;
	bfs::directory_iterator iter( imageDir ), endIter;
	while( iter != endIter )
	{
		bfs::directory_entry entry = *iter;
		iter++;
		if( !is_regular_file( entry.path() ) ) { continue; }
		imagePaths.push_back( entry.path().string() );
	}
	
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
		datum.name = path;
		datum.imageSize = image.size();
		datum.imagePoints = corners;
		datum.objectPoints = patternGrid.GetPoints();

		writer.WriteNext( datum );
		
		cv::Mat drawn;
		cv::drawChessboardCorners( image, patternGrid.GetSize(), corners, true );
		cv::imshow( "images", image );
		cv::waitKey( 1 );
	}
	
	cv::destroyWindow( "images" );
	
}
	
