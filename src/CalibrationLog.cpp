#include "calotypes/CalibrationLog.h"
#include <boost/algorithm/string.hpp>

namespace calotypes
{
	
CalibrationLogReader::CalibrationLogReader( const std::string& path )
: log( path )
{
	if( !log.is_open() ) 
	{ 
		throw std::runtime_error( "Could not open log at path " + path ); 
	}
}

bool CalibrationLogReader::GetNext( std::string& imagePath, cv::Size& imageSize,
									CameraTrainingData& data )
{
	std::string headerLine, objectLine, imageLine;
	if( !std::getline( log, headerLine ) || 
		!std::getline( log, objectLine ) || 
		!std::getline( log, imageLine ) ) { return false; }
	
	std::vector< std::string > headerSplits, objectSplits, imageSplits;
	
	// First read object coordinates
	boost::split( headerSplits, headerLine, boost::is_any_of(" "), boost::token_compress_on );
	boost::split( objectSplits, objectLine, boost::is_any_of(" "), boost::token_compress_on );
	boost::split( imageSplits, imageLine, boost::is_any_of(" "), boost::token_compress_on );
	if( headerSplits[0].size() == 0 ) { headerSplits.erase( headerSplits.begin() ); }
	if( objectSplits[0].size() == 0 ) { objectSplits.erase( objectSplits.begin() ); }
	if( imageSplits[0].size() == 0 ) { imageSplits.erase( imageSplits.begin() ); }
	
	if( headerSplits.size() != 3 || objectSplits.size() % 3 != 0 || imageSplits.size() % 2 != 0 ) 
	{
		throw std::runtime_error( "Malformed log line(s) has incorrect number of items." );
	}
	
	if( objectSplits.size() / 3 != imageSplits.size() / 2 )
	{
		throw std::runtime_error( "Log lines do not have matching number of points." );
	}
	
	imagePath = headerSplits[0];
	imageSize = cv::Size( std::stoi( headerSplits[1] ), std::stoi( headerSplits[2] ) );
	
	unsigned int numItems = imageSplits.size() / 2;
	
	data.imagePoints = ImagePoints( numItems );
	data.objectPoints = ObjectPoints( numItems );
	
	unsigned int objectCount = 0, imageCount = 0;
	for( unsigned int i = 0; i < numItems; i++ )
	{
		cv::Point3f objectPoint( std::stod( objectSplits[ objectCount + 0 ] ),
									std::stod( objectSplits[ objectCount + 1 ] ),
									std::stod( objectSplits[ objectCount + 2 ] ) );
		cv::Point2f imagePoint( std::stod( imageSplits[ imageCount + 0 ] ),
								std::stod( imageSplits[ imageCount + 1 ] ) );
		objectCount += 3;
		imageCount += 2;
		data.objectPoints[i] = objectPoint;
		data.imagePoints[i] = imagePoint;
	}
	return true;
}

CalibrationLogWriter::CalibrationLogWriter( const std::string& path )
: log( path )
{
	if( !log.is_open() )
	{
		throw std::runtime_error( "Could not open log at path " + path );
	}
}

void CalibrationLogWriter::WriteNext( const std::string& imagePath, const cv::Size& imageSize,
									  const CameraTrainingData& data )
{
	log << imagePath << " " << imageSize.width << " " << imageSize.height << std::endl;
	
	unsigned int numItems = data.imagePoints.size();
	log << data.objectPoints[0].x << " " << data.objectPoints[0].y << " " 
		<< data.objectPoints[0].z;
	for( unsigned int i = 1; i < numItems; i++ )
	{
		log << " " << data.objectPoints[i].x << " " << data.objectPoints[i].y
			<< " " << data.objectPoints[i].z;
	}
	log << std::endl;
	
	log << data.imagePoints[0].x << " " << data.imagePoints[0].y;
	for( unsigned int i = 1; i < numItems; i++ )
	{
		log << " " << data.imagePoints[i].x << " " << data.imagePoints[i].y;
	}
	log << std::endl;
}

} // end namespace calotypes
