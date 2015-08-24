#include <unistd.h>

#include "calotypes/CalibrationLog.h"
#include "calotypes/Shufflers.hpp"

#include <boost/random/random_device.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <boost/algorithm/string.hpp>

using namespace calotypes;

void PrintHelp()
{
	std::cout << "shuffle_log: Shuffles a detection log randomly" << std::endl;
	std::cout << "Arguments:" << std::endl;
	std::cout << "[-i inputPath] Input detection log" << std::endl;
	std::cout << "[-o outputPath] Output detection prefix" << std::endl;
	std::cout << "[-n numShuffles (1)] Number of permutations to create" << std::endl;
}

int main ( int argc, char** argv )
{
	
	std::string inputPath, outputPath;
	unsigned int numCopies = 1;
	char c;
	
	if( argc == 1 )
	{
		PrintHelp();
		return 0;
	}
	
	while( (c = getopt( argc, argv, "hi:o:n:")) != -1 )
	{
		switch( c )
		{
			case 'h':
				PrintHelp();
				return 0;
			case 'i':
				inputPath.assign( optarg );
				break;
			case 'o':
				outputPath.assign( optarg );
				break;
			case 'n':
				numCopies = strtol( optarg, NULL, 10 );
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
	
	CalibrationLogReader reader( inputPath );
	
	std::vector< std::string > splits;
	boost::split( splits, inputPath, boost::is_any_of( "." ) );
	std::string extension = splits[ splits.size()-1 ];
	
	struct DetectionPoint
	{
		CameraTrainingData data;
		std::string path;
		cv::Size imageSize;
	};
	
	std::vector<DetectionPoint> points;
	DetectionPoint point;
	while( reader.GetNext( point.path, point.imageSize, point.data ) )
	{
		points.push_back( point );
	}
	

	boost::random::random_device rng;
	boost::random::mt19937 generator;
	generator.seed( rng );
	
	std::vector<unsigned int> inds( points.size() );
	for( unsigned int i = 0; i < numCopies; i++ )
	{
		FisherYatesShuffle( points.size(), inds, generator );
		
		std::stringstream ss;
		ss << outputPath << "_" << i << "." << extension;
		CalibrationLogWriter writer( ss.str() );
		for( unsigned int j = 0; j < points.size(); j++ )
		{
			const DetectionPoint& pt = points[ inds[j] ];
			writer.WriteNext( pt.path, pt.imageSize, pt.data );
		}
	}
	
}
