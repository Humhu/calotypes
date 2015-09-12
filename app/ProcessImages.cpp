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
	std::cout << "[-v vis_dir] Directory to output visualized detections" << std::endl;
	std::cout << "[-t mthresh] Movement rejection threshold" << std::endl;
}

double ComputeMovement( const ImagePoints& a, const ImagePoints& b )
{
	if( a.size() != b.size() )
	{
		throw std::runtime_error( "Cannot compare image points of different sizes." );
	}
	
	double acc = 0;
	double dx, dy;
	for( unsigned int i = 0; i < a.size(); i++ )
	{
		dx = a[i].x - b[i].x;
		dy = a[i].y - b[i].y;
		acc += std::sqrt( dx*dx + dy*dy );
	}
	return acc / a.size();
}

struct Entry
{
	std::string path;
	std::pair<std::string,std::string> stem;
};

bool CompareNames( const Entry& a, const Entry& b )
{
	if( a.stem.first.size() == b.stem.first.size() )
	{ 
		return a.stem < b.stem;
	}
	return a.stem.first.size() < b.stem.first.size();
}

int main( int argc, char** argv )
{

	unsigned int cbRows = 6, cbCols = 7;
	double cbDim = 1.0;
	char c;
	std::string imageDir( "./" );
	std::string outputPath( "detections.txt" );
	bool enableVis = false;
	std::string visDir;
	double movementThreshold = 1.0;
	
	while( (c = getopt( argc, argv, "hr:c:s:d:o:v:t:")) != -1 )
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
			case 'v':
				enableVis = true;
				visDir.assign( optarg );
				break;
			case 't':
				movementThreshold = strtod( optarg, NULL );
				break;
			case '?':
                return -1;
            default:
				std::cerr << "Unrecognized option." << std::endl;
                return -1;
		}
	}

	PlanarGridStructure patternGrid( cbRows, cbCols, cbDim );
	
	if( imageDir.back() != '/' ) { imageDir += "/"; }
	if( visDir.back() != '/' ) { visDir += "/"; }
	
	if( !bfs::exists( imageDir ) || !bfs::is_directory( imageDir ) )
	{
		throw std::runtime_error( "Invalid directory " + imageDir );
	}
	
	std::vector< Entry > entries;
// 	std::vector< std::string > imagePaths;
// 	std::vector< std::pair<std::string, std::string> > fileStems;
	bfs::directory_iterator iter( imageDir ), endIter;
	while( iter != endIter )
	{
		bfs::directory_entry entry = *iter;
		iter++;
		if( !is_regular_file( entry.path() ) ) { continue; }
		Entry e;
		e.path = entry.path().string();
// 		imagePaths.push_back( entry.path().string() );
		std::pair<std::string,std::string> stem( entry.path().stem().string(), 
												 entry.path().extension().string() );
// 		fileStems.push_back( stem );
		e.stem = stem;
		entries.push_back( e );
	}
	std::sort( entries.begin(), entries.end(), CompareNames );
	
	CalibrationLogWriter writer( outputPath );
	
	cv::namedWindow( "images" );
	
	ImagePoints corners, prevCorners;
	unsigned int accepted = 0;
	for( unsigned int i = 0; i < entries.size(); i++ )
	{
		std::string path = entries[i].path;
		std::cout << "Reading image " << path << std::endl;
		cv::Mat image = cv::imread( path );
		if( image.empty() )
		{
			std::cerr << "Could not read image at " << path << std::endl;
			return -1;
		}
		cv::Mat gray;
		cv::cvtColor( image, gray, cv::COLOR_BGR2GRAY);
		
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
		
		if( prevCorners.empty() )
		{
			prevCorners = corners;
			continue;
		}
		
		double movement = ComputeMovement( prevCorners, corners );
		prevCorners = corners;
		if( movement > movementThreshold )
		{
			std::cout << "Movement of " << movement << " exceeds threshold. Continuing..." << std::endl;
			continue;
		}
		else
		{
			std::cout << "Movement of " << movement << " accepted!" << std::endl;
			accepted++;
		}
		
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
		
		if( enableVis )
		{
			std::stringstream ss;
			ss << visDir << entries[i].stem.first << "_detection" << entries[i].stem.second;
			imwrite( ss.str(), image );
		}
	}
	
	cv::destroyWindow( "images" );
	
	std::cout << "Accepted " << accepted << " images." << std::endl;
}
	
